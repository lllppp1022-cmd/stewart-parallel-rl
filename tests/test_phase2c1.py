import json,inspect
import numpy as np,pytest,pandas as pd
from src.model import ROOT,sha
from src.rl.stewart_residual_env import StewartResidualEnv,env_config
from src.rl.observations import MeasuredState,HistoryObservation
from src.rl.action_adapter import HeldFzAction
from src.rl.rewards import physical_cost,failure_penalty

def make(tmp_path,name='env',history=3):return StewartResidualEnv(tmp_path/name,tasks=[env_config()['training_tasks'][0]],history_length=history,mode='Unit Test')

def test_one_decision_exact50_and_correct_api(tmp_path):
    e=make(tmp_path);obs,info=e.reset(seed=42);assert obs.shape==(255,) and obs.dtype==np.float32
    obs,r,t,c,i=e.step(np.array([.5],np.float32));assert i['physical_steps']==50 and e.owner.n==50 and not t and not c and np.isfinite(r) and e.observation_space.contains(obs);e.close()

def test_reset_restores_all_states_and_rng(tmp_path):
    e=make(tmp_path);a,_=e.reset(seed=123);q=e.owner.data.qpos.copy();drive=e.owner.lag.state.copy();task=e.current_task.copy();e.step(np.array([.8],np.float32));b,_=e.reset(seed=123)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,e.owner.data.qpos);np.testing.assert_array_equal(drive,e.owner.lag.state);assert e.current_task==task and e.owner.residual.base.observer.steps==0 and e.owner.residual.held_action.normalized==0;e.close()

def test_current_only_observation_variant(tmp_path):
    e=make(tmp_path,history=1);o,_=e.reset(seed=42);assert o.shape==(85,);e.close()

def test_action_nonfinite_rejected_without_step(tmp_path):
    e=make(tmp_path);e.reset(seed=42)
    for a in [np.array([np.nan]),np.ones(4)]:
        with pytest.raises(ValueError):e.step(a)
    assert e.owner.n==0;e.close()

def test_physical_mapping_only_fz_in_existing_task_interface(tmp_path):
    e=make(tmp_path);e.reset(seed=42);e.step(np.array([1.],np.float32));d=e.owner.residual.held_action.task
    assert d[0]==.5 and d[1]==0 and d[2]==0;np.testing.assert_allclose(e.owner.residual.base.jacobian.T@e.owner.residual.trace['requested'],np.r_[0,0,d],atol=1e-10);e.close()

def test_failure_immediate_decision_and_original200_hold(tmp_path):
    e=make(tmp_path);e.reset(seed=42);owner=e.owner;original=owner.step
    def fault():
        original()
        if owner.n==1:owner.hold('unit_injected_protection',failed=True)
    owner.step=fault;o,r,t,c,i=e.step(np.array([0.],np.float32));assert t and not c and i['action_physical_steps']==1 and i['hold_physical_steps']==200 and i['physical_steps']==201 and r<=-2400;e.close()

def test_reward_scales_cap_and_early_failure_ranking():
    c=env_config()['reward'];p,f=physical_cost([0,0,.00005],0,np.zeros(6),np.zeros(6),0,.5,c);assert p['z']==1 and f==1
    p,f=physical_cost([1,1,1],1,np.ones(6),np.zeros(6),.5,.5,c);assert abs(sum(p.values())-20)<1e-10
    for time in [0.,1.,5.9]:assert failure_penalty(time,c)<-20*np.ceil((6-time)/.05)

def test_observation_schema_no_hidden_truth():
    allowed=set(MeasuredState.__dataclass_fields__)
    for x in ['tau','drive','actual_force','external_wrench','load_on_time','time_remaining']:assert x not in allowed
    text=inspect.getsource(HistoryObservation)
    for x in ['actuator_gainprm','.lag','external_world','task_id']:assert x not in text

def test_real_zero_and_random_prechecks_already_passed():
    v=json.loads((ROOT/'results/phase2c1_env_validation.json').read_text(encoding='utf-8'));assert v['environment_gate_passed'] and v['api']['zero_complete_state_bitwise_equal'] and v['precheck_episodes']==27

def test_fixed_reward_no_external_flag():
    signature=inspect.signature(physical_cost);assert 'load_active' not in signature.parameters and 'tau' not in signature.parameters

def test_history_all_frozen():
    h=json.loads((ROOT/'results/phase2c1_frozen_history.json').read_text(encoding='utf-8'))
    for name,row in h['files'].items():assert sha(ROOT/name)==row['sha256'],name
