import json,hashlib,inspect
from pathlib import Path
import numpy as np,pytest,mujoco
from src.model import ROOT
from src.phase1.plant import create_plant,platform_state
from src.phase1.controllers import Controller
from src.phase1.trajectory import Trajectory
from src.control.leaky_pid import LeakyPID
from src.control.nominal_task_model import build_model,pose_coordinates,tangent
from src.control.disturbance_observer import MomentumDisturbanceObserver
from src.control.robust_controller_c3 import RobustControllerC3
from src.control.phase2b1_experiment import RobustOwner,trial_metrics
from src.disturbance.metrics import recovery

@pytest.fixture(scope='module')
def model():return build_model()
@pytest.fixture(scope='module')
def nominal():return create_plant()
@pytest.fixture(scope='module')
def home(nominal):return Trajectory(nominal,'home',ROOT/'results/phase1_preflight/references')

def test_nominal_constraint_reduction(model,nominal):
    assert model['constraint_tangent_error']<1e-12 and model['site_tangent_error']<1e-12
    assert min(model['mass_eigenvalues'])>0 and model['gravity_balance_error']<1e-4
    m=np.array(model['mass']);np.testing.assert_allclose(m,m.T,atol=1e-12)
    assert abs(m[0,4])>.1 # Retain cross-axis inertia, not a fictitious diagonal plant.

def test_lambda_zero_exact_calls(nominal,home):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);c=Controller(nominal,'C1',.001);l=LeakyPID(nominal,0)
    p,r,v=platform_state(nominal,d);ref=home.at(0);ref.q=ref.q+1e-4
    for k in range(200):
        a,ia=c.command(d.qpos,d.qvel,p,r,v,ref,(k+1)*.001);b,ib=l.command(d.qpos,d.qvel,p,r,v,ref,(k+1)*.001);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c.integral,l.integral)

@pytest.mark.parametrize('leak',[.5,1,2,5])
def test_leak_formula_and_antiwindup(leak,nominal,home):
    d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);p,r,v=platform_state(nominal,d);l=LeakyPID(nominal,leak);l.integral[:]=.001
    ref0=home.at(0);error0=ref0.q-d.qpos[l.qadr]
    u,_=l.command(d.qpos,d.qvel,p,r,v,ref0,.001);np.testing.assert_allclose(l.integral,.001*(1-leak*.001)+.001*error0,atol=1e-15,rtol=1e-12)
    ref=home.at(0);ref.q[:]=.2;l.integral[:]=.004
    for k in range(100):u,_=l.command(d.qpos,d.qvel,p,r,v,ref,.002+k*.001)
    assert max(abs(u))<=400 and l.antiwindup_count>0 and max(abs(l.integral))<=.004 and max(abs(l.integral))<=.005
    l.reset();assert not np.any(l.integral)

def synthetic_observer(model,axis,amplitude,omega=15):
    obs=MomentumDisturbanceObserver(model,omega);mass=np.array(model['mass']);v=np.zeros(6);est=[];w0=np.array(model['equilibrium_wrench'])
    for k in range(3000):
        est.append(obs.update(np.zeros(6),v));obs.remember_applied_command(np.zeros(6),v,w0+np.array(model['damping'])@v)
        f=np.zeros(6)
        if 500<=k<1500:f[axis]=amplitude
        v+=.001*np.linalg.solve(mass,f)
    return obs,np.array(est)

@pytest.mark.parametrize('axis,amplitude',[(2,-5),(3,.25),(4,-.15)])
def test_causal_observer_direction_and_decay(model,axis,amplitude):
    obs,est=synthetic_observer(model,axis,amplitude)
    np.testing.assert_array_equal(est[:501],0.)
    assert est[1400,axis]*amplitude>0 and abs(est[1400,axis]-amplitude)<abs(amplitude)*.01
    assert abs(est[-1,axis])<abs(amplitude)*.001
    obs.reset();assert obs.previous_velocity is None and not np.any(obs.estimate)

def test_observer_bounds_and_nonfinite(model):
    obs=MomentumDisturbanceObserver(model);obs.update(np.zeros(6),np.zeros(6));obs.remember_applied_command(np.zeros(6),np.zeros(6),np.zeros(6));r=obs.update(np.zeros(6),np.ones(6)*100)
    assert np.all(abs(r)<=obs.limits) and obs.clipped
    with pytest.raises(ValueError):obs.update(np.ones(6)*np.nan,np.zeros(6))
    obs.reset();obs.update(np.zeros(6),np.zeros(6))
    with pytest.raises(RuntimeError):obs.update(np.zeros(6),np.zeros(6))

def test_invalid_nominal_model_rejected(model):
    broken=dict(model);broken['mass']=np.full((6,6),np.nan).tolist()
    with pytest.raises(ValueError):MomentumDisturbanceObserver(broken)

def test_c3_allowed_inputs_and_no_truth():
    sig=inspect.signature(RobustControllerC3.command);assert list(sig.parameters)==['self','qpos','qvel','p','r','twist','reference','t']
    source=inspect.getsource(RobustControllerC3)+inspect.getsource(MomentumDisturbanceObserver)
    for term in ('qfrc_constraint','qfrc_applied','actuator_force','external_ctrl','external_world','start_step','stop_step','qacc'):
        assert term not in source

def test_actual_hidden_parameters_not_read(model,nominal,home):
    actual=create_plant('mass110');d=mujoco.MjData(actual);mujoco.mj_forward(actual,d);p,r,v=platform_state(actual,d)
    a=RobustControllerC3(nominal,model);b=RobustControllerC3(nominal,model)
    u,_=a.command(d.qpos,d.qvel,p,r,v,home.at(0),.001)
    actual.body_mass[actual.body('upper').id]*=1.2
    w,_=b.command(d.qpos,d.qvel,p,r,v,home.at(0),.001);np.testing.assert_array_equal(u,w)

def test_c3_mapping_limits_and_reset(model,nominal,home):
    c=RobustControllerC3(nominal,model);d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);p,r,v=platform_state(nominal,d)
    c.observer.estimate[[2,3,4]]=[-10,.5,-.2]
    u,_=c.command(d.qpos,d.qvel,p,r,v,home.at(0),.001);diag=c.diagnostics
    np.testing.assert_allclose(c.jacobian.T@diag['compensation_N'],diag['realized_compensation_wrench'],atol=1e-12)
    assert np.max(abs(diag['compensation_N']))<=.02+1e-12 and max(abs(u))<=400
    np.testing.assert_array_equal(diag['compensation_target_wrench'][[0,1,5]],0.)
    c.reset();assert not np.any(c.previous_compensation) and c.observer.previous_velocity is None

def test_c3_invalid_jacobian_rejected(model,nominal,home):
    c=RobustControllerC3(nominal,model);d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);p,r,v=platform_state(nominal,d)
    c.base.condition=100;c.base.last_jacobian_time=0
    with pytest.raises(ValueError,match='Jacobian'):c.command(d.qpos,d.qvel,p,r,v,home.at(0),.001)

def test_incompatible_headroom_and_rate_rejected(model,nominal,home):
    c=RobustControllerC3(nominal,model);d=mujoco.MjData(nominal);mujoco.mj_forward(nominal,d);p,r,v=platform_state(nominal,d);c.previous_compensation[:]=1.
    c.base.command=lambda *args:(np.full(6,400.),dict(raw=np.full(6,400.),integral=np.zeros(6),condition=c.condition,clipped=False))
    with pytest.raises(ValueError,match='headroom'):c.command(d.qpos,d.qvel,p,r,v,home.at(0),.001)

def make_owner(folder,method,model,**params):
    return RobustOwner(dict(folder=str(folder),controller=method,condition='nominal',record=False,pace=False,trajectory='home',max_sim_seconds=.15,max_wall_seconds=20,exit_on_failure=True,stage='unit_test',disturbance=dict(start_step=20,stop_step=80,wrench=[0,0,-2,.1,0,0]),task_model=model,bandwidth_rad_s=15,compensation_gain=0.,**params))

def test_c3_zero_gain_real_physics_equivalence(tmp_path,model):
    a=make_owner(tmp_path/'C0','C0',model);b=make_owner(tmp_path/'C3','C3',model)
    a.run();b.run()
    with np.load(tmp_path/'C0/integration_states.npz') as x,np.load(tmp_path/'C3/integration_states.npz') as y:np.testing.assert_array_equal(x['states'],y['states'])

def test_safety_unchanged_and_new_controller_reject(tmp_path,model):
    x=make_owner(tmp_path/'failure','C3',model)
    def reject(*args):raise ValueError('TEST_ONLY_C3_REJECT')
    x.controller.command=reject;x.run();summary=json.loads((tmp_path/'failure/summary.json').read_text());assert 'controller_rejected' in summary['failure'] and summary['physics_steps']>=200
    assert x.controller.config['weld_angle_limit_rad']==1e-4 and x.controller.config['partial_proxy_min_clearance_m']==.005

def test_split_frozen_before_tuning():
    p=json.loads((ROOT/'configs/phase2b1_protocol.json').read_text());dev=p['development_tasks'];test=p['independent_test_proposals'];assert {t['name'] for t in dev}.isdisjoint(t['name'] for t in test)
    assert all((t['wrench'],t['start_step'],t['stop_step']) not in [(x['wrench'],x['start_step'],x['stop_step']) for x in dev] for t in test)

def test_recovery_keeps_no_excursion_and_censored():
    t=np.arange(1,1001)*.001;r=recovery(t,np.ones(1000,bool),0,1);assert r['time_s'] is None and 'NO EXCURSION' in r['status']
    r=recovery(t[:200],np.zeros(200,bool),0,1,complete=False);assert r['time_s'] is None and r['right_censored']

def test_history_frozen():
    saved=json.loads((ROOT/'results/phase2b1_frozen_history.json').read_text(encoding='utf-8'))
    for name,row in saved['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['sha256'],name

def test_completed_protocol_and_raw_acceptance():
    path=ROOT/'results/phase2b1_validation.json'
    if not path.exists():pytest.skip('Final raw acceptance not executed yet')
    r=json.loads(path.read_text(encoding='utf-8'));assert r['research_trials']==76 and r['final_trials']==45 and r['independent_final_trials']==20
    assert r['historical_files_changed']==[] and r['same_initial_state_count']==1
    assert all(x['bitwise_equal'] for x in r['baseline_full_state_equivalence'])
    assert all(x['maximum_offline_causal_estimate_difference']<1e-8 for x in r['observer_causal_recomputation'])
    assert r['lambda_zero_equivalence']['bitwise_equal'] and r['current_implementation_reproduction']['bitwise_equal']

def test_gui_recording_and_actual_video_acceptance():
    path=ROOT/'results/phase2b1_validation.json'
    if not path.exists():pytest.skip('Final GUI/video acceptance not executed yet')
    r=json.loads(path.read_text(encoding='utf-8'));assert r['gui_nonintrusivity']['bitwise_equal'] and r['gui_nonintrusivity']['physics_steps_each']==2048
    v=json.loads((ROOT/'assets/videos/stewart_phase2b1_robust_control_comparison.validation.json').read_text(encoding='utf-8'));assert v['frames']==600 and v['duration_s']==24 and v['all_frames_decoded'] and v['all_state_hashes_match']
    for f in v['frame_mapping']:assert f['last_curve_time_s']<=f['time_s']+1e-10 and f['input_step']==f['state_index']

def test_no_final_test_retuning():
    root=ROOT/'results/runs/phase2b1_20261028_v1';p=root/'controller_parameters_frozen.json'
    if not p.exists():pytest.skip('Parameters not frozen yet')
    final=json.loads((root/'final_protocol_frozen.json').read_text(encoding='utf-8'));assert final['parameters_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert p.stat().st_mtime_ns<=(root/'final_protocol_frozen.json').stat().st_mtime_ns
    for cfg in (root/'final').glob('*/*/config.json'):
        c=json.loads(cfg.read_text(encoding='utf-8'));assert c['final_protocol_sha256']==hashlib.sha256((root/'final_protocol_frozen.json').read_bytes()).hexdigest()
