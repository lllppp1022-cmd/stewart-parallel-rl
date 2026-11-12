import json,inspect,hashlib
from pathlib import Path
import numpy as np,pandas as pd,pytest,mujoco
from src.model import ROOT
from src.phase1.plant import create_plant,platform_state
from src.phase1.controllers import Controller
from src.phase1.trajectory import Trajectory
from src.control.robust_controller_c3 import RobustControllerC3
from src.control.residual_governor import SharedBudgetGovernor
from src.control.residual_action import ResidualController,CausalAction,task_to_leg,ActionObservation
from src.control.phase2b2_experiment import MeasurementAdapter,ResidualOwner,actual_force_map

@pytest.fixture(scope='module')
def setup():
    m=create_plant();d=mujoco.MjData(m);mujoco.mj_forward(m,d);ref=Trajectory(m,'home',ROOT/'results/phase1_preflight/references').at(0);model=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'));return m,d,ref,model

def test_real_zero_equivalence_gate():
    f=ROOT/'results/phase2b2_zero_residual_equivalence.csv'
    if not f.exists():pytest.skip('Actual gate not yet executed')
    d=pd.read_csv(f);assert len(d)==2 and d.full_integration_state_bitwise_equal.all() and d.all_physical_signals_equal.all() and d.same_terminal_reason.all() and d.maximum_state_difference.max()==0

@pytest.mark.parametrize('axis',range(3))
def test_task_units_direction_and_full_map(setup,axis):
    m,d,ref,model=setup;c=Controller(m,'C0',.001);action=np.zeros(3);action[axis]=.01;f,w=task_to_leg(action,c.jhome,c.condition);np.testing.assert_allclose(c.jhome.T@f,w,atol=1e-12);assert w[[2,3,4][axis]]==.01
    np.testing.assert_allclose(actual_force_map(m,d),c.jhome,atol=1e-6)

def test_shared_single_budget_not_double():
    g=SharedBudgetGovernor();g.previous[:]=5;cmd,r=g.apply(np.zeros(6),np.full(6,5),np.full(6,5));np.testing.assert_array_equal(cmd,5);assert abs(r['total_add']).max()==5

def test_shared_rate_and_native_headroom():
    g=SharedBudgetGovernor();cmd,r=g.apply(np.zeros(6),np.zeros(6),np.full(6,1));np.testing.assert_allclose(cmd,.02);cmd,r=g.apply(np.zeros(6),np.zeros(6),np.full(6,-1));np.testing.assert_allclose(cmd,0)
    g.reset();cmd,r=g.apply(np.full(6,399.99),np.full(6,399.99),np.ones(6));assert max(cmd)<=400 and max(abs(r['total_add']))<=.02+1e-12

@pytest.mark.parametrize('bad',[np.full(6,np.nan),np.ones(5),np.full(6,np.inf)])
def test_invalid_requests_rejected(bad):
    with pytest.raises(ValueError):SharedBudgetGovernor().apply(np.zeros(6),np.zeros(6),bad)

@pytest.mark.parametrize('frequency,stride',[(20,50),(50,20)])
def test_decision_holds_and_no_extra_updates(setup,frequency,stride):
    m,d,ref,_=setup;core=Controller(m,'C0',.001);action=CausalAction('zero','leg',frequency);wrapper=ResidualController(core,action);p,r,v=platform_state(m,d)
    for k in range(101):wrapper.command(d.qpos,d.qvel,p,r,v,ref,(k+1)*.001);assert wrapper.trace['decision']==(k%stride==0)
    assert action.calls==(100//stride+1)

def test_c3_remembers_actual_composed_command(setup):
    m,d,ref,model=setup;core=RobustControllerC3(m,model,30,1);a=CausalAction('probe','task',50,[.5,0,0]);a.calls=40;wrapper=ResidualController(core,a);p,r,v=platform_state(m,d);cmd,_=wrapper.command(d.qpos,d.qvel,p,r,v,ref,.001)
    x=core.diagnostics['position_coordinates'];obs=core.observer;known=obs.previous_drive+obs.equilibrium+obs.stiffness@x+obs.damping@v;np.testing.assert_allclose(known,core.jacobian.T@cmd,atol=1e-12);assert wrapper.trace['observer_command_corrected']

def test_zero_c3_does_not_add_observer_updates(setup):
    m,d,ref,model=setup;core=RobustControllerC3(m,model,30,1);calls=[];old=core.observer.remember_applied_command
    def count(*a):calls.append(1);return old(*a)
    core.observer.remember_applied_command=count;w=ResidualController(core,CausalAction());p,r,v=platform_state(m,d);w.command(d.qpos,d.qvel,p,r,v,ref,.001);assert len(calls)==1 and core.observer.steps==1

def test_known_command_not_mistaken_for_disturbance(setup):
    from src.control.disturbance_observer import MomentumDisturbanceObserver
    model=setup[3];o=MomentumDisturbanceObserver(model,30);v=np.zeros(6);force=np.array(model['equilibrium_wrench']);mass=np.array(model['mass']);res=np.array([0,0,.5,0,0,0.])
    for _ in range(200):
        o.update(np.zeros(6),v);known=force+np.array(model['damping'])@v+res;o.remember_applied_command(np.zeros(6),v,known);v+=.001*np.linalg.solve(mass,res)
    assert abs(o.estimate).max()<1e-10

def test_privileged_action_blocked_in_standard_interface(setup):
    class Oracle:privileged=True;space='leg';frequency=1000
    with pytest.raises(ValueError,match='Privileged'):ResidualController(Controller(setup[0],'C0',.001),Oracle())

class Capture:
    name='capture'
    def __init__(self):self.rows=[]
    def command(self,q,dq,p,r,v,ref,t):self.rows.append((p.copy(),r.copy(),v.copy()));return np.zeros(6),{}
    def reset(self):pass

def test_noise_does_not_mutate_live_state(setup):
    m,d,ref,_=setup;before=d.qpos.copy();p,r,v=platform_state(m,d);a=MeasurementAdapter(Capture(),dict(position_std_m=1e-6,seed=42));a.command(d.qpos,d.qvel,p,r,v,ref,.001);np.testing.assert_array_equal(d.qpos,before);np.testing.assert_array_equal(p,d.site('upper_center').xpos);assert np.any(a.sensor_trace['measured_p']!=p)

@pytest.mark.parametrize('delay',[1,2,5])
def test_delay_queue_has_no_future(delay):
    a=MeasurementAdapter(Capture(),dict(delay_steps=delay))
    for k in range(12):
        a.command(np.zeros(43),np.zeros(42),np.ones(3)*k,np.eye(3),np.ones(6)*k,None,k*.001);assert a.sensor_trace['capture_step']==max(0,k-delay) and a.sensor_trace['read_step']==k

def test_noise_pairing_is_local_seeded_and_identical():
    a=MeasurementAdapter(Capture(),dict(velocity_std=1e-5,seed=42));b=MeasurementAdapter(Capture(),dict(velocity_std=1e-5,seed=42))
    for k in range(20):
        for adapter in (a,b):adapter.command(np.zeros(43),np.zeros(42),np.zeros(3),np.eye(3),np.zeros(6),None,k*.001)
        np.testing.assert_array_equal(a.sensor_trace['noise'],b.sensor_trace['noise'])

def test_single_factor_guard():
    with pytest.raises(ValueError):MeasurementAdapter(Capture(),dict(position_std_m=1e-6,delay_steps=1))

def test_policy_has_no_external_truth_fields():
    assert set(ActionObservation.__dataclass_fields__)=={'p','rotation','velocity','reference_p','reference_rotation','reference_velocity','equivalent_disturbance'}
    text=inspect.getsource(CausalAction)
    for key in ('external_world','start_step','stop_step','qacc','body_mass'):assert key not in text

def test_actual_parameter_initialization_isolated(tmp_path,setup):
    cfg=dict(folder=str(tmp_path/'actual'),controller='C3',condition='nominal',record=False,pace=False,trajectory='home',max_sim_seconds=.05,max_wall_seconds=20,exit_on_failure=True,stage='unit_test',task_model=setup[3],bandwidth_rad_s=30,compensation_gain=1,actual_perturbation=dict(mass_scale=1.01),disturbance=dict(start_step=10,stop_step=20,wrench=[0]*6,kind='zero'))
    x=ResidualOwner(cfg);b=x.model.body('upper').id;assert x.model.body_mass[b]==pytest.approx(x.nominal.body_mass[b]*1.01);x.run();assert (Path(cfg['folder'])/'safety_events.json').exists()

def test_all_history_frozen():
    r=json.loads((ROOT/'results/phase2b2_frozen_history.json').read_text(encoding='utf-8'))
    for name,entry in r['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==entry['sha256'],name
