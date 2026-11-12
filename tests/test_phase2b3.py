import inspect,json
import numpy as np,pandas as pd,pytest,mujoco
from scipy.spatial.transform import Rotation
from src.model import ROOT,sha
from src.phase1.plant import create_plant,platform_state
from src.phase1.controllers import Controller
from src.phase1.trajectory import Trajectory
from src.control.robust_controller_c3 import RobustControllerC3
from src.control.residual_action import task_to_leg,ActionObservation
from src.control.task_residual_4d import task4_to_leg,Task4Controller,ZeroOrProbe4,FourDAdapter
from src.control.yaw_feedback_controller import YawFeedback,CausalFourD
from src.control.yaw_disturbance_compensation import YawObserverCompensation,C4WithLead
from src.control.phase2b3_experiment import YawOwner,ScaledReference,policy

@pytest.fixture(scope='module')
def sample():
    m=create_plant();d=mujoco.MjData(m);mujoco.mj_forward(m,d);ref=Trajectory(m,'home',ROOT/'results/phase1_preflight/references').at(0);task=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'));return m,d,ref,task

def observation(sample,angle=0.,rate=0.,estimate=0.):
    m,d,ref,_=sample;p,r,v=platform_state(m,d);r=Rotation.from_rotvec([0,0,angle]).as_matrix()@ref.r;v[5]=rate;est=np.zeros(6);est[5]=estimate
    return ActionObservation(p.copy(),r.copy(),v.copy(),ref.p.copy(),ref.r.copy(),ref.twist.copy(),est)

@pytest.mark.parametrize('axis',range(4))
def test_four_dimensional_world_order_unit_and_mapping(sample,axis):
    c=Controller(sample[0],'C0',.001);a=np.zeros(4);a[axis]=.01;f,w=task4_to_leg(a,c.jhome,c.condition);assert w[axis+2]==.01;np.testing.assert_allclose(c.jhome.T@f,w,atol=1e-12)
    if axis<3:np.testing.assert_allclose(f,task_to_leg(a[:3],c.jhome,c.condition)[0],atol=1e-12)

@pytest.mark.parametrize('bad',[np.zeros(3),np.full(4,np.nan),np.full(4,np.inf)])
def test_invalid_4d_rejected(sample,bad):
    c=Controller(sample[0],'C0',.001)
    with pytest.raises(ValueError):task4_to_leg(bad,c.jhome,c.condition)

def test_pd_world_so3_sign_and_damping(sample):
    p=YawFeedback(100,40)
    assert p.request(observation(sample,.001,0))[3]<0
    assert p.request(observation(sample,-.001,0))[3]>0
    assert p.request(observation(sample,0,.001))[3]<0
    o=observation(sample,.001);o.reference_rotation[:]=Rotation.from_rotvec([.001,.002,.003]).as_matrix()@o.reference_rotation;o.rotation[:]=Rotation.from_rotvec([0,0,.001]).as_matrix()@o.reference_rotation
    assert p.request(o)[3]==pytest.approx(-.1,abs=1e-10)

def test_observer_uses_causal_estimate_and_compensating_sign(sample):
    c=YawObserverCompensation(1);assert c.request(observation(sample,estimate=-.2))[3]==.2;assert c.request(observation(sample,estimate=1))[3]==-.3

def test_policies_no_hidden_gain_load_or_future_inputs():
    for cls in [YawFeedback,CausalFourD,YawObserverCompensation,C4WithLead]:
        text=inspect.getsource(cls)
        for prohibited in ['actuator_gainprm','external_world','profile.command','owner.data','start_step','stop_step','qacc']:assert prohibited not in text
    assert 'gain' not in ActionObservation.__dataclass_fields__

def test_zero4_single_observer_update_and_previous_real_command(sample):
    m,d,ref,model=sample;c=RobustControllerC3(m,model,30,1);w=Task4Controller(c,ZeroOrProbe4());p,r,v=platform_state(m,d);cmd,_=w.command(d.qpos,d.qvel,p,r,v,ref,.001);assert c.observer.steps==1
    np.testing.assert_array_equal(cmd,c.diagnostics['base_force_N']+c.diagnostics['compensation_N'])
    c=RobustControllerC3(m,model,30,1);w=Task4Controller(c,YawObserverCompensation(1));c.observer.estimate[5]=-.2;cmd,_=w.command(d.qpos,d.qvel,p,r,v,ref,.001);x=c.diagnostics['position_coordinates'];o=c.observer
    np.testing.assert_allclose(o.previous_drive+o.equilibrium+o.stiffness@x+o.damping@v,c.jacobian.T@cmd,atol=1e-12);assert c.observer.steps==1
    assert max(abs(w.trace['total_add']))<=.02+1e-10

def test_20hz_held_wrench_but_current_mapping(sample):
    p=CausalFourD(100,40);a=FourDAdapter(p);j=Controller(sample[0],'C0',.001).jhome
    o=observation(sample,.001)
    for k in range(101):a.request(o,j);assert a.decision==(k%50==0)
    assert a.policy_version==3

def test_scaled_reference_validated_time_and_velocity(sample):
    base=Trajectory(sample[0],'periodic',ROOT/'results/phase1_preflight/references');scaled=ScaledReference(base,.85);a=scaled.at(2);b=base.at(1.7)
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.p,b.p);np.testing.assert_array_equal(a.dq,b.dq*.85);np.testing.assert_array_equal(a.twist,b.twist*.85);assert scaled.duration==base.duration/.85

def test_gain_actual_only_native_limits_and_channels(tmp_path,sample):
    cfg=dict(folder=str(tmp_path/'gain'),controller='C3',condition='nominal',record=False,pace=False,trajectory='home',max_sim_seconds=.025,max_wall_seconds=20,exit_on_failure=True,stage='unit_test',yaw_policy=dict(kind='observer',gain=1),task_model=sample[3],bandwidth_rad_s=30,compensation_gain=1,actual_perturbation=dict(gain_scales=[.995,1.005]*3),disturbance=dict(start_step=10,stop_step=20,wrench=[0,0,-1,0,0,0]))
    x=YawOwner(cfg);np.testing.assert_array_equal(x.nominal.actuator_gainprm[:6,0],np.ones(6));np.testing.assert_array_equal(x.model.actuator_gainprm[:6,0],[.995,1.005]*3);before=x.model.actuator_ctrlrange.copy();x.run();np.testing.assert_array_equal(x.model.actuator_ctrlrange,before)
    f=pd.read_csv(x.recorder.folder/'physics.csv.gz');assert len(f)==25;assert f.task4_target_3.abs().max()<=.30
    np.testing.assert_allclose(f[[f'force_{j}' for j in range(6)]],f[[f'f_cmd_{j}' for j in range(6)]].to_numpy()*np.array([.995,1.005]*3),atol=1e-10)
    np.testing.assert_array_equal(f[[f'ctrl_{j}' for j in range(6,12)]].to_numpy(),f[[f'external_ctrl_{j}' for j in range(6)]].to_numpy())
    # World Z load is rotated into site-local actuators: local Fx/Fy need not vanish.
    np.testing.assert_allclose(f[[f'external_world_{j}' for j in [0,1,3,4,5]]],0,atol=1e-12)
    expected=np.where((f.input_step>=10)&(f.input_step<20),-1.,0.)
    np.testing.assert_allclose(f.external_world_2,expected,atol=1e-12)

def test_mixed_uncertainty_guard(tmp_path,sample):
    with pytest.raises(ValueError,match='gain-only'):YawOwner(dict(measurement={'delay_steps':1}))

def test_original_yaw_safety_threshold_and_old_failure_unchanged():
    cfg=json.loads((ROOT/'configs/phase1_protocol.json').read_text(encoding='utf-8'));assert cfg['timestep_s']==.001
    old=ROOT/'results/runs/phase2b2_20261029_v1/robustness/robust_force10/asymmetric_0.01_C3/events.jsonl';events=[json.loads(x) for x in old.read_text(encoding='utf-8').splitlines()];r=next(x for x in events if x['type']=='protect_hold');assert r['reason']=='restricted_orientation_envelope' and r['time_s']==pytest.approx(.384)

def test_phase2b3_full_history_hashes():
    manifest=json.loads((ROOT/'results/phase2b3_frozen_history.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():assert sha(ROOT/name)==info['sha256'],name
