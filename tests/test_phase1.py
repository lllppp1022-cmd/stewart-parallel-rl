import json,time,hashlib,subprocess
from pathlib import Path
import numpy as np,pandas as pd,mujoco,pytest
from src.model import ROOT,UPSTREAM,load_model,ids,sha
from src.phase1.controllers import Controller
from src.phase1.trajectory import Trajectory,Reference,protocol
from src.phase1.plant import create_plant,platform_state,external_command
from src.phase1.collision import CollisionQuery,original_mesh_vertices_local
from src.phase1.executor import check_safety,run_trial

@pytest.fixture(scope='module')
def nominal():
    m=load_model();m.opt.timestep=.001;return m
@pytest.fixture(scope='module')
def home(nominal):return Trajectory(nominal,'home',ROOT/'results/phase1_preflight/references')
@pytest.fixture(scope='module')
def query():return CollisionQuery()

@pytest.mark.parametrize('channel',list(range(6)))
def test_input_sort_order_sign_newtons(nominal,channel):
    d=mujoco.MjData(nominal);d.ctrl[channel]=1.;mujoco.mj_forward(nominal,d);_,dof=ids(nominal);expected=np.zeros(6);expected[channel]=1
    assert np.allclose(d.actuator_force[:6],expected) and np.allclose(d.qfrc_actuator[dof],expected)
def test_native_force_clamp():
    m=create_plant('asymmetric_gain');d=mujoco.MjData(m);d.ctrl[:6]=1000;mujoco.mj_forward(m,d)
    assert np.all(abs(d.actuator_force[:6])<=400) and np.all(m.actuator_forcerange[:6]==[-400,400])
def test_slide_is_not_installation_chord(nominal,home):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);assert np.max(abs(d.actuator_length[:6]))<1e-10 and np.min(home.geometry.attachment_chord_lengths(home.geometry.home_position,home.geometry.home_rotation))>.5
def test_virtual_work_and_mujoco_force_map():
    df=pd.read_csv(ROOT/'results/phase1_force_mapping_validation.csv');assert len(df)==12 and df.virtual_work_abs_error_W.max()<1e-10 and df.wrench_max_abs_error.max()<1e-5 and df.mujoco_power_abs_error_W.max()<1e-6
def test_c0_original_equivalence():
    r=json.loads((ROOT/'results/phase1_preflight/c0_phase0_equivalence.json').read_text());assert r['max_force_command_difference_N']==r['max_qpos_difference']==0 and r['bitwise_equal']
def test_nominal_deterministic_real_integration(home,nominal):
    sequences=[]
    for repeat in range(2):
        m=create_plant();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=Controller(nominal,'C0',.001);forces=[]
        for k in range(300):
            p,r,v=platform_state(m,d);u,_=c.command(d.qpos,d.qvel,p,r,v,home.at(d.time+.001),d.time+.001);d.ctrl[:6]=u;mujoco.mj_step(m,d);mujoco.mj_forward(m,d);forces.append(u.copy())
        sequences.append((d.qpos.copy(),np.asarray(forces)))
    assert np.array_equal(sequences[0][0],sequences[1][0]) and np.array_equal(sequences[0][1],sequences[1][1])
def test_pid_antiwindup_and_reset(nominal,home):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);c=Controller(nominal,'C1',.001);ref=home.at(0);ref.q[:]=.2
    p,r,v=platform_state(nominal,d)
    for k in range(100):u,info=c.command(d.qpos,d.qvel,p,r,v,ref,k*.001)
    assert c.antiwindup_count>0 and max(abs(u))<=400 and max(abs(c.integral))<.005
    c.integral[:]=.003;c.reset();assert np.all(c.integral==0)
def test_invalid_reference_stroke_rejected(nominal,home):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);c=Controller(nominal,'C1',.001);ref=home.at(0);ref.q[:]=.36;p,r,v=platform_state(nominal,d)
    with pytest.raises(ValueError,match='stroke'):c.command(d.qpos,d.qvel,p,r,v,ref,0)
@pytest.mark.parametrize('name',['z_roll','roll_pitch','periodic'])
def test_combinations_preflight_full_chain(nominal,name):
    tr=Trajectory(nominal,name,ROOT/'results/phase1_preflight/references');qs=tr.spline(np.linspace(0,6,1201));assert np.all(qs>-.1) and np.all(qs<.35)
    audit=json.loads((ROOT/f'results/phase1_preflight/{name}_interpolation.json').read_text());assert audit['max_offgrid_stroke_error_m']<1e-7
def test_nominal_perturbed_parameters_isolated(nominal):
    mass=nominal.body_mass.copy();inertia=nominal.body_inertia.copy();gain=nominal.actuator_gainprm.copy();pert=create_plant('mass110');assert pert.body('upper').mass[0]==pytest.approx(nominal.body('upper').mass[0]*1.1)
    assert np.array_equal(nominal.body_mass,mass) and np.array_equal(nominal.body_inertia,inertia) and np.array_equal(nominal.actuator_gainprm,gain)
def test_external_channel_policy():
    assert np.all(external_command('nominal',3)==0) and np.all(external_command('mass110',3)==0)
    assert np.array_equal(external_command('external_load',3),[0,0,-10,.5,0,0]) and np.all(external_command('external_load',5)==0)
def test_proxy_encloses_mesh_vertices(nominal,query):
    assert query.description['native_mass_max_change']==0
    for spec in query.description['proxies']:
        v=original_mesh_vertices_local(nominal,spec['body']);assert np.linalg.norm(v[:,:2],axis=1).max()<=spec['radius_m']+1e-9
        assert v[:,2].min()>=spec['z_min_m']-1e-9 and v[:,2].max()<=spec['z_max_m']+1e-9
def test_proxy_home_and_no_live_state_modification(nominal,query):
    d=mujoco.MjData(nominal);q=d.qpos.copy();r=query.evaluate(q);assert r['native_contacts']==0 and r['minimum_clearance_m']>.005 and np.array_equal(d.qpos,q) and query.data is not d
def test_collision_filter_cross_legs_preserved(query):
    names=[(query.model.geom(a).name,query.model.geom(b).name) for a,b in query.pairs]
    assert any('legbot_1' in a and 'legup_2' in b or 'legbot_1' in b and 'legup_2' in a for a,b in names)
    assert not any(set([a,b])==set(['proxy_legbot_1','proxy_legup_1']) for a,b in names)
def test_true_signed_distance_and_truncation_primitive():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="a" type="capsule" fromto="0 0 0 0 0 1" size=".05"/><body pos=".2 0 .5"><freejoint/><geom name="b" type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert mujoco.mj_geomDistance(m,d,0,1,.5,None)==pytest.approx(.1)
    d.qpos[0]=.05;mujoco.mj_forward(m,d);assert mujoco.mj_geomDistance(m,d,0,1,.5,None)<0 and d.ncon>0
    d.qpos[0]=1.;mujoco.mj_forward(m,d);assert mujoco.mj_geomDistance(m,d,0,1,.2,None)==pytest.approx(.2)
def test_safety_stop_invalid_target_has_real_hold_log(nominal,home,query):
    class InvalidTarget:
        geometry=home.geometry;name='invalid_target_unit_test'
        def at(self,t):
            ref=home.at(t);ref.q[:]=.36;return ref
    folder=ROOT/'results/phase1_test_runs'/str(time.time_ns());summary,df=run_trial(nominal,InvalidTarget(),'C1','nominal',.001,folder,query)
    assert not summary['success'] and 'controller_rejected' in summary['failure_reason'] and .19<summary['simulation_time_s']<.3
    assert df.phase.eq('protect_hold').all() and (folder/'events.json').exists() and df[[f'integral_m_s_{j}' for j in range(1,7)]].eq(0).all().all()
def test_weld_gate_not_relaxed(nominal,home,query):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);p,r,v=platform_state(nominal,d);proxy=query.evaluate(d.qpos)
    assert check_safety(nominal,d,p,r,home.geometry,0,1.0001e-4,5.3,proxy)=='weld_angle'
def test_phase0_and_upstream_freeze_unchanged():
    r=json.loads((ROOT/'results/phase1_frozen_phase0.json').read_text(encoding='utf-8'));assert r['count']>=100
    for path,info in r['files'].items():assert sha(ROOT/path)==info['sha256'],path
    assert subprocess.check_output(['git','status','--porcelain'],cwd=UPSTREAM,text=True)==''
def test_final_nominal_and_fine_constraint_records():
    path=ROOT/'results/phase1_controller_comparison.csv'
    if not (ROOT/'results/phase1_suite_manifest.json').exists():pytest.skip('Formal suite still running; final acceptance reruns this check')
    df=pd.read_csv(path);assert len(df)==42
    nominal=df[(df.condition=='nominal')&(df.timestep_s==.001)];fine=df[df.timestep_s==.0005]
    assert len(nominal)==21 and len(fine)==6 and nominal.success.all() and fine.success.all()
    assert df.max_weld_angle_rad.max()<1e-4 and df.max_weld_translation_m.max()<1e-4
def test_fine_keeps_control_period_1ms():
    if not (ROOT/'results/phase1_suite_manifest.json').exists():pytest.skip('Formal suite still running')
    df=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');assert df.controller_period_s.eq(.001).all()
def test_video_full_decode_state_and_hash_validation():
    record=json.loads((ROOT/'results/phase1_video_validation.json').read_text(encoding='utf-8'));assert record['frames']==record['state_matches']==540 and record['actual_capture'] and record['duration_s']==54
    assert sha(ROOT/'assets/videos/phase1_dynamic_control.mp4')==record['sha256']
def test_metrics_independent_xyz_quaternion_recalculation():
    df=pd.read_csv(ROOT/'results/phase1_metric_recomputation.csv');assert len(df)==42 and df.position_rmse_difference_mm.max()<1e-8 and df.orientation_rmse_difference_deg.max()<1e-8
def test_gui_real_close_archive():
    r=json.loads((ROOT/'results/phase1_gui_validation.json').read_text(encoding='utf-8'));assert r['same_live_model_and_data'] and not r['per_physics_viewer_lock'] and any(x['exit']['user_closed'] for x in r['normal_python_exits'])
def test_no_hidden_perturbed_parameters_in_controller_inputs(nominal,home):
    a=mujoco.MjData(nominal);mujoco.mj_forward(nominal,a);p,r,v=platform_state(nominal,a)
    first=Controller(nominal,'C0',.001);u1,_=first.command(a.qpos,a.qvel,p,r,v,home.at(.001),.001)
    pert=create_plant('mass110');b=mujoco.MjData(pert);mujoco.mj_forward(pert,b);pb,rb,vb=platform_state(pert,b)
    second=Controller(nominal,'C0',.001);u2,_=second.command(b.qpos,b.qvel,pb,rb,vb,home.at(.001),.001)
    assert np.array_equal(u1,u2)
def test_display_repeats_do_not_replace_scientific_results():
    r=json.loads((ROOT/'results/phase1_video_delivery_manifest.json').read_text(encoding='utf-8'));assert r['formal_42_trials_unchanged'] and len(r['equivalence'])==9 and all(x['full_state_bitwise_equal'] for x in r['equivalence'])
    table=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');assert len(table)==42 and not table.success.all()
