import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,pytest,mujoco
from scipy.spatial.transform import Rotation
from src.model import ROOT
from src.phase1.plant import create_plant
from src.visualization.enhanced.owner import EnhancedOwner
from src.disturbance.disturbance_profiles import DisturbanceProfile,generalized_external
from src.disturbance.executor import DisturbanceOwner
from src.disturbance.metrics import recovery,error_series,compute

def owner(folder,wrench=(0,0,0,0,0,0),**extra):
    return DisturbanceOwner(dict(folder=str(folder),controller='C0',condition='nominal',record=False,pace=False,trajectory='home',max_sim_seconds=.12,max_wall_seconds=15,exit_on_failure=True,disturbance=dict(start_step=20,stop_step=40,wrench=wrench),**extra))

def test_native_channels_units_and_limits():
    m=create_plant();assert m.nu==12
    assert [m.actuator(i).name for i in range(6)]==[f'leg{i}_ctrl' for i in range(1,7)]
    np.testing.assert_array_equal(m.actuator_ctrlrange[:6],np.tile([-400,400],(6,1)))
    np.testing.assert_array_equal(m.actuator_gear[6:],np.eye(6))
    assert all(m.actuator_trnid[i,0]==m.site('upper_center').id for i in range(6,12))

def test_boundary_and_pulse_and_zero():
    p=DisturbanceProfile(start_step=2,stop_step=4,wrench=(0,0,-2,0,0,0),kind='pulse')
    assert [p.active(i) for i in range(6)]==[False,False,True,True,False,False]
    assert np.all(p.command(4,np.eye(3))==0)
    assert np.all(DisturbanceProfile(kind='zero',wrench=(0,0,-10,0,0,0)).command(2500,np.eye(3))==0)

@pytest.mark.parametrize('kw',[dict(wrench=(1,2)),dict(wrench=(0,0,np.nan,0,0,0)),dict(wrench=(0,0,501,0,0,0)),dict(start_step=2.5),dict(frame='unknown')])
def test_bad_profiles(kw):
    with pytest.raises(ValueError):DisturbanceProfile(**kw)

def test_world_local_transform_and_native_generalized_force():
    m=create_plant();d=mujoco.MjData(m)
    # Independent kinematics-only scratch setup: not live motion.
    joint=m.joint(int(m.body('upper').jntadr[0]));adr=int(joint.qposadr[0]);d.qpos[adr+3:adr+7]=Rotation.from_euler('xyz',[.07,-.05,.11]).as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
    r=d.site_xmat[m.site('upper_center').id].reshape(3,3);p=DisturbanceProfile(wrench=(0,0,-10,.5,.1,0));local=p.command(2000,r);a=generalized_external(m,d,local)
    np.testing.assert_allclose(a['world'],p.wrench,atol=1e-12);assert a['maximum_error']<1e-12
    assert np.linalg.norm(local[:2])>.1

def test_zero_pair_bitwise_and_real_response(tmp_path):
    a=owner(tmp_path/'zero');b=owner(tmp_path/'load',(0,0,-2,0,0,0))
    cfg=dict(a.config,folder=str(tmp_path/'legacy'));cfg.pop('disturbance');c=EnhancedOwner(cfg)
    for i in range(120):
        a.step();b.step();c.step()
        np.testing.assert_array_equal(a.data.qpos,c.data.qpos);np.testing.assert_array_equal(a.data.ctrl,c.data.ctrl)
        if i<20:np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    assert np.linalg.norm(a.data.qpos-b.data.qpos)>1e-9
    for x in (a,b,c):x.recorder.close(dict(test_only=True))
    df=pd.read_csv(tmp_path/'load/physics.csv.gz');active=df.external_ctrl_2!=0
    assert np.flatnonzero(active).tolist()==list(range(20,40));assert df.step.iloc[-1]==120
    assert df.generalized_mapping_error.max()<1e-12 and df.warning_count.max()==0
    assert df.weld_angle_rad.max()<=1e-4 and df.proxy_clearance_m.min()>=.005
    np.testing.assert_allclose(df.time_s,np.arange(1,121)*.001,atol=1e-13)
    events=[json.loads(x) for x in (tmp_path/'load/events.jsonl').read_text().splitlines()]
    assert [(e['type'],e['step']) for e in events if e['type'] in ('load_on','load_off')]==[('load_on',20),('load_off',40)]

def test_recovery_dwell_and_censoring():
    t=np.arange(1,1001)*.001;good=np.zeros(1000,bool);good[99:399]=True
    assert recovery(t,good,0,1)['time_s']==pytest.approx(.1)
    good[:]=False;assert recovery(t,good,0,1)['time_s'] is None
    r=recovery(t[:200],good[:200],0,1,complete=False);assert r['right_censored'] and r['time_s'] is None
    good[:]=True;assert recovery(t[:299],good[:299],0,1)['time_s'] is None

def test_inband_response_latency_is_not_recovery():
    t=np.arange(1,2001)*.001;good=np.ones(2000,bool);good[700:1300]=False
    r=recovery(t,good,0,2);assert r['time_s']==pytest.approx(1.301)
    good[700:]=False;assert recovery(t,good,0,2)['time_s'] is None
    r=recovery(t,np.ones(2000,bool),0,2);assert r['time_s'] is None and r['status'].startswith('NO EXCURSION')

def test_so3_error_not_euler_subtraction():
    df=pd.DataFrame({**{f'p_{j}':[0.] for j in range(3)},**{f'p_ref_{j}':[0.] for j in range(3)},'rpy_0':[0.],'rpy_1':[0.],'rpy_2':[179.],'rpy_ref_0':[0.],'rpy_ref_1':[0.],'rpy_ref_2':[-179.]})
    p,a,_=error_series(df);assert a[0]==pytest.approx(2.)

def test_reactive_hold_actual_archive(tmp_path):
    x=owner(tmp_path/'hold');x.hold('TEST_ONLY_INJECTED_PROTECTIVE_EVENT',failed=True);x.run()
    s=json.loads((tmp_path/'hold/summary.json').read_text());df=pd.read_csv(tmp_path/'hold/physics.csv.gz')
    assert s['failure']=='TEST_ONLY_INJECTED_PROTECTIVE_EVENT' and s['physics_steps']==200
    assert df.failure.notna().all() and (tmp_path/'hold/integration_states.npz').exists()

def test_recomputed_metrics_and_integral(tmp_path):
    x=owner(tmp_path/'metrics');x.run();p=dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3);r=compute(tmp_path/'metrics',p)
    df=pd.read_csv(tmp_path/'metrics/physics.csv.gz');pe,ae,_=error_series(df);mask=(df.input_step>=20)&(df.input_step<40)
    assert r['load_position_iae_mm_s']==pytest.approx(pe[mask].sum()*.001)
    assert r['position_rmse_mm']==pytest.approx(np.sqrt(np.mean(pe**2)))

def test_frozen_history_unchanged():
    saved=json.loads((ROOT/'results/phase2a_frozen_history.json').read_text(encoding='utf-8'))
    assert saved['count']>=1017
    for name,row in saved['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['sha256'],name

def test_no_live_qpos_assignment_or_rl():
    text=(ROOT/'src/disturbance/executor.py').read_text(encoding='utf-8')
    assert 'mujoco.mj_step(m,d)' in text and 'd.ctrl[:6]=force' in text
    assert 'qpos[:]=' not in text and 'qvel[:]=' not in text

def test_controller_rejection_gets_real_restricted_hold(tmp_path):
    x=owner(tmp_path/'controller_exception')
    def reject(*a,**kw):raise ValueError('TEST_ONLY_REJECTION')
    x.controller.command=reject;x.run()
    s=json.loads((tmp_path/'controller_exception/summary.json').read_text());assert 'controller_rejected' in s['failure'] and s['physics_steps']>=200

def test_bounded_runtime_preserves_saved_real_load(tmp_path):
    source=ROOT/'results/runs/phase2a_20261028_v1/formal/A_home_force/C0'
    if not source.exists():pytest.skip('Saved physical trial not available')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'));cfg.update(folder=str(tmp_path/'schedule_equivalence'),record=False,max_sim_seconds=2.1)
    x=DisturbanceOwner(cfg);x.run()
    with np.load(source/'integration_states.npz') as a,np.load(tmp_path/'schedule_equivalence/integration_states.npz') as b:np.testing.assert_array_equal(a['states'][:2100],b['states'])

def test_final_raw_and_legacy_acceptance():
    p=ROOT/'results/phase2a_validation.json'
    if not p.exists():pytest.skip('Full archive acceptance not executed yet')
    r=json.loads(p.read_text(encoding='utf-8'));assert r['physical_trial_count']==51 and r['formal_trials']==33
    assert r['historical_files_changed']==[] and r['maximum_independent_mapping_error']<1e-10
    assert all(x['external_commands_bitwise_equal'] and x['qpos_bitwise_equal'] for x in r['legacy_reproduction'])
    assert r['observer']['bitwise_equal'] and r['video']['all_state_hashes_match']

def test_final_video_time_and_actual_state_mapping():
    p=ROOT/'assets/videos/stewart_phase2a_disturbance_rejection.validation.json'
    if not p.exists():pytest.skip('Video has not been composed yet')
    r=json.loads(p.read_text(encoding='utf-8'));assert r['frames']==750 and r['duration_s']==30 and r['all_frames_decoded']
    assert r['no_future_curves'] and r['mj_step_calls']==0
    for f in r['frame_mapping']:
        assert f['physical_step']==f['state_index']+1 and f['input_step']==f['state_index']
        assert f['load_active']==(2000<=f['input_step']<4000)
