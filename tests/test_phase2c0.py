import ast,inspect,textwrap,json
import numpy as np,pandas as pd,pytest,mujoco
from src.model import ROOT,sha
from src.dynamics.actuator_lag import ActuatorLag
from src.dynamics.execution_hook import INSTRUMENTED_AST
from src.disturbance.executor import DisturbanceOwner
from src.control.phase2c0_experiment import PhysicalOwner,PregeneratedNoise
from src.control.phase2c0_actions import TaskDiagnostic
from src.control.phase2b2_experiment import MeasurementAdapter

@pytest.mark.parametrize('tau',[0,.002,.005,.01])
def test_exact_zoh_step_and_release(tau):
    initial=np.array([80,81,82,83,84,85.]);x=ActuatorLag(tau,initial);expect=initial.copy()
    for k in range(300):
        command=initial+(.2 if 20<=k<100 else -.1 if 150<=k<200 else 0);expect=command.copy() if not tau else command+(expect-command)*np.exp(-.001/tau);np.testing.assert_allclose(x.advance(command),expect,atol=2e-13,rtol=0)
    assert x.steps==300

def test_differential_tau_and_zero_element():
    tau=np.array([0,.002,.004,.006,.008,.01]);x=ActuatorLag(tau,np.ones(6)*80);out=x.advance(np.ones(6)*81);assert out[0]==81 and np.all(np.diff(out)<0)

def test_disabled_and_zero_tau_bitwise_passthrough():
    c=np.array([80.0000000000001,81,82,83,84,85]);np.testing.assert_array_equal(ActuatorLag(.01,np.zeros(6),False).advance(c),c);np.testing.assert_array_equal(ActuatorLag(0,np.zeros(6),True).advance(c),c)

def test_force_bounds_and_nonfinite():
    x=ActuatorLag(.005,np.ones(6)*399);assert max(x.advance(np.ones(6)*1000))<=400
    with pytest.raises(ValueError):x.advance(np.full(6,np.nan))

@pytest.mark.parametrize('tau',[-.001,.1,np.nan])
def test_bad_tau_rejected(tau):
    with pytest.raises(ValueError):ActuatorLag(tau,np.ones(6)*80)

def test_only_one_explicit_hook_added_to_frozen_physics():
    source=ast.parse(textwrap.dedent(inspect.getsource(DisturbanceOwner.step)));original=ast.dump(source,include_attributes=False)
    from src.dynamics.execution_hook import instrumented_step
    import src.dynamics.execution_hook as hook
    # Rebuild source tree and remove exactly the one plant-input insertion.
    tree=ast.parse(textwrap.dedent(inspect.getsource(DisturbanceOwner.step)));body=tree.body[0].body;anchor=next(i for i,s in enumerate(body) if isinstance(s,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='last_force' for t in s.targets));body.insert(anchor,ast.parse('force = self.apply_actuator_command(force)').body[0]);assert ast.dump(tree,include_attributes=False)==INSTRUMENTED_AST
    body.pop(anchor);assert ast.dump(tree,include_attributes=False)==original
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='mj_step' for n in ast.walk(tree))==1

def cfg(folder,tau=.005):
    model=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'))
    return dict(folder=str(folder),controller='C3',yaw_policy=dict(kind='observer',gain=1),task_model=model,bandwidth_rad_s=30,compensation_gain=1,condition='nominal',record=False,pace=False,max_sim_seconds=.08,max_wall_seconds=20,exit_on_failure=True,stage='unit_test',trajectory='home',disturbance=dict(start_step=10,stop_step=40,wrench=[0,0,-1,0,0,0]),actuator_dynamics=dict(enabled=bool(tau),tau_s=tau))

def test_lag_real_physics_three_forces_and_known_command_memory(tmp_path):
    x=PhysicalOwner(cfg(tmp_path/'lag'));initial=x.lag.state.copy();assert initial.min()>50;before=x.data.qpos.copy();x.run();d=pd.read_csv(x.recorder.folder/'physics.csv.gz');assert len(d)==80 and x.lag.steps==80
    cmd=d[[f'F_command_{j}' for j in range(6)]].to_numpy();drive=d[[f'F_drive_{j}' for j in range(6)]].to_numpy();force=d[[f'force_{j}' for j in range(6)]].to_numpy();prev=d[[f'F_drive_previous_{j}' for j in range(6)]].to_numpy()
    np.testing.assert_allclose(drive,prev+(1-np.exp(-.001/.005))*(cmd-prev),atol=1e-12);np.testing.assert_allclose(drive,force,atol=1e-12);assert abs(cmd-drive).max()>1e-6
    np.testing.assert_allclose(d[[f'observer_memory_wrench_{j}' for j in range(6)]],d[[f'known_command_wrench_{j}' for j in range(6)]].to_numpy(),atol=1e-11)
    assert abs(d[[f'known_command_wrench_{j}' for j in range(6)]].to_numpy()-d[[f'nominal_drive_wrench_{j}' for j in range(6)]].to_numpy()).max()>1e-6
    np.testing.assert_array_equal(d[[f'ctrl_{j}' for j in range(6)]],drive);np.testing.assert_array_equal(d[[f'ctrl_{j}' for j in range(6,12)]],d[[f'external_ctrl_{j}' for j in range(6)]].to_numpy());assert not np.array_equal(x.data.qpos,before)

def test_lag_changes_physical_trajectory_and_no_state_zero_start(tmp_path):
    a=PhysicalOwner(cfg(tmp_path/'off',0));b=PhysicalOwner(cfg(tmp_path/'on',.005));np.testing.assert_array_equal(a.data.qpos,b.data.qpos);np.testing.assert_array_equal(a.data.qvel,b.data.qvel);a.run();b.run();assert max(abs(a.data.qpos-b.data.qpos))>1e-10

def test_lag_continues_through_original_protect_hold(tmp_path):
    c=cfg(tmp_path/'hold');c['max_sim_seconds']=.02;x=PhysicalOwner(c);x.step();x.lag.state+=.1;x.hold('test_latched_failure',failed=True);start=x.lag.steps;x.run();assert x.lag.steps>x.n-2 and x.data.time>=.201-1e-10 and x.lag.steps>start

def test_pregenerated_noise_same_seed_and_budget():
    a=PregeneratedNoise(42);b=PregeneratedNoise(42)
    for _ in range(10):np.testing.assert_array_equal(a.normal(12),b.normal(12))

def test_nonprivileged_diagnostic_cannot_read_tau_gain_drive():
    text=inspect.getsource(TaskDiagnostic)
    for word in ['.lag','actual_tau','actuator_gainprm','profile.command','F_drive','F_actual','owner.data']:assert word not in text

def test_original_models_controllers_and_history_unchanged():
    manifest=json.loads((ROOT/'results/phase2c0_frozen_history.json').read_text(encoding='utf-8'))
    for path,info in manifest['files'].items():assert sha(ROOT/path)==info['sha256'],path
