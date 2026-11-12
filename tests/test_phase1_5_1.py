import json,hashlib
from pathlib import Path
import numpy as np,pytest,mujoco
from src.model import ROOT
from src.phase1.plant import create_plant
from src.visualization.enhanced.style import LEG_COLORS,series,color_display_model,leg_mapping
from src.visualization.enhanced.owner import EnhancedOwner
from src.visualization.enhanced.events import read_events,visible_events,latest_event
from src.visualization.enhanced.overlay import ProjectionOverlay,project_points

@pytest.fixture
def owner(tmp_path):
    x=EnhancedOwner(dict(folder=str(tmp_path/'enhanced'),controller='C0',condition='nominal',record=False,pace=False,max_sim_seconds=.01,max_wall_seconds=10))
    yield x
    if not x.recorder.stream.closed:x.recorder.close(dict(test_only=True))

def test_actuator_order_and_single_palette():
    m=create_plant('nominal');mapping=leg_mapping(m);assert len(set(LEG_COLORS))==6
    for i,row in enumerate(mapping):assert row['actuator_id']==i and row['joint']==f'joint_lu_{i+1}' and row['color']==LEG_COLORS[i]

def test_display_colour_dynamics_unchanged():
    a=create_plant('nominal');b=create_plant('nominal');mapping=color_display_model(b)
    for field in ['body_mass','body_inertia','body_pos','body_quat','jnt_range','jnt_type','geom_contype','geom_conaffinity','geom_friction','eq_data','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']:np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    da=mujoco.MjData(a);db=mujoco.MjData(b)
    for _ in range(10):mujoco.mj_step(a,da);mujoco.mj_step(b,db)
    np.testing.assert_array_equal(da.qpos,db.qpos);np.testing.assert_array_equal(da.qvel,db.qvel)
    for row in mapping:
        for gid in row['geoms']:assert b.geom_matid[gid]==-1

@pytest.mark.parametrize('group,n,field',[('position',3,'p_m'),('legs',6,'slide_m')])
def test_absolute_delta_inverse(group,n,field):
    a=np.arange(n*3).reshape(3,n)*.001;r=a+.0001;h={field:np.linspace(.001,.003,n)};av,rv,u=series(a,r,group,'Absolute',h);dv,dr,_=series(a,r,group,'Delta from Home',h)
    np.testing.assert_allclose(dv+np.array(h[field])*1000,av,atol=1e-12);np.testing.assert_allclose(dr+np.array(h[field])*1000,rv,atol=1e-12);assert u=='mm'

def test_error_sign_and_units():
    a=np.array([.002,-.001,0]);r=np.array([.001,0,0]);v,zero,u=series(a,r,'position','Tracking Error');np.testing.assert_array_equal(v,[1,-1,0]);assert not zero.any();assert u=='mm'

def test_mdeg_exact():
    a=np.array([.1,-.25,0]);r=a+.0001;ad,rd,_=series(a,r,'angles','Absolute',angle_unit='deg');am,rm,u=series(a,r,'angles','Absolute',angle_unit='mdeg');np.testing.assert_array_equal(am,ad*1000);np.testing.assert_array_equal(rm,rd*1000);assert u=='mdeg'

def test_unknown_home_rejected_no_first_sample_substitution():
    with pytest.raises(ValueError,match='UNKNOWN'):series([[0,0,1.1]],[[0,0,1.1]],'position','Delta from Home')
    a,_,_=series([[0,0,1.1]],[[0,0,1.1]],'position','Absolute');assert a[0,2]==1100

def test_fixed_home_saved_before_physics(owner):
    saved=json.loads((Path(owner.config['folder'])/'home_baseline.json').read_text());assert saved['time_s']==0;assert owner.n==0;np.testing.assert_array_equal(saved['slide_m'],owner.data.qpos[owner.qadr]);before=json.dumps(owner.home_baseline,sort_keys=True);owner.command(dict(kind='home'));owner.command(dict(kind='pause'));owner.command(dict(kind='pause'));assert json.dumps(owner.home_baseline,sort_keys=True)==before

def test_rejected_is_not_executed(owner):
    owner.command(dict(kind='target',payload={'z_mm':999},source='test command queue'));owner.maybe_plan();events=read_events(owner.config['folder']);assert any(e['label']=='REJECTED' for e in events);assert not any(e['type'] in ('plan_accepted','motion_started') for e in events)

def test_queued_is_not_preflighted_and_superseding(owner):
    owner.command(dict(kind='increment',payload={'z_mm':.25},source='test mapping W'));owner.command(dict(kind='increment',payload={'roll_deg':.05},source='test mapping A'));events=read_events(owner.config['folder']);assert any(e['type']=='target_superseded' and e['superseded_command_id']==1 for e in events);assert not any(e['type']=='motion_started' for e in events)
    owner.maybe_plan();owner.step();events=read_events(owner.config['folder']);p=next(e for e in events if e['type']=='plan_accepted');s=next(e for e in events if e['type']=='motion_started');assert p['command_id']==s['command_id']==2;assert p['preflight']=='PASSED';assert s['time_s']>p['time_s']

def test_hold_aborts_and_does_not_rebase(owner):
    baseline=dict(owner.home_baseline);owner.command(dict(kind='increment',payload={'z_mm':.25}));owner.maybe_plan();owner.step();q=owner.data.qpos.copy();owner.command(dict(kind='hold'));np.testing.assert_array_equal(q,owner.data.qpos);assert owner.home_baseline==baseline;assert any(e['type']=='motion_aborted' for e in read_events(owner.config['folder']))

def test_visual_command_has_no_physical_effect(owner):
    old=owner.data.qpos.copy();ctrl=owner.data.ctrl.copy();owner.command(dict(kind='visual_labels',payload={'enabled':False}));np.testing.assert_array_equal(owner.data.qpos,old);np.testing.assert_array_equal(owner.data.ctrl,ctrl);assert not owner.snapshot().data['visual_labels']

def test_event_no_future_or_inferred_data(tmp_path):
    p=tmp_path/'events.jsonl';p.write_text('\n'.join(json.dumps(e) for e in [dict(type='input',kind='increment',payload={'z_mm':.25},time_s=1),dict(type='target_rejected',time_s=2),dict(type='plan_accepted',time_s=3)]));events=read_events(tmp_path);assert latest_event(events,2)['label']=='REJECTED';assert all(e['time_s']<=2 for e in visible_events(events,2));assert len(events)==3

def test_projection_actual_camera_geometry():
    o=ProjectionOverlay();mujoco.mj_forward(o.model,o.data)
    from src.rendering import camera
    mujoco.mjv_updateScene(o.model,o.data,mujoco.MjvOption(),None,camera(),mujoco.mjtCatBit.mjCAT_ALL,o.scene);a,b=o.scene.camera;center=(a.pos+b.pos)/2;forward=(a.forward+b.forward)/2;right=np.cross(forward,a.up);p=project_points(o.scene,[center+forward,center+forward+right*.1,center+forward+a.up*.1],1216,912)
    np.testing.assert_allclose(p[0],[608,456],atol=1e-4);assert p[1][0]>608;assert p[2][1]<456

def test_no_legacy_file_changed():
    old=json.loads((ROOT/'results/phase1_5_1_frozen_history.json').read_text(encoding='utf-8'));changed=[]
    for path,meta in old['files'].items():
        p=ROOT/path
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:changed.append(path)
    assert changed==[]

def test_completion_and_protection_same_step_not_success(owner,monkeypatch):
    # Explicit software fault injection; not a claimed physical failure experiment.
    import src.visualization.owner as base
    owner.completed=True;monkeypatch.setattr(base,'check_safety',lambda *a,**kw:'test_fault_injection_at_completion');owner.step();events=read_events(owner.config['folder']);assert not any(e['type']=='motion_completed' for e in events);assert any(e['type']=='motion_aborted' for e in events)

def test_preflight_rejection_is_explicit(owner,monkeypatch):
    import src.visualization.owner as base
    def rejected(*a,**kw):raise ValueError('test_preflight_fault_injection')
    monkeypatch.setattr(base,'SmoothPlan',rejected);owner.command(dict(kind='increment',payload={'z_mm':.25}));owner.maybe_plan();events=read_events(owner.config['folder']);assert any(e['type']=='target_rejected' and e['preflight']=='FAILED' and e['command_id']==1 for e in events);assert not any(e['type'] in ('motion_started','plan_accepted') for e in events)
