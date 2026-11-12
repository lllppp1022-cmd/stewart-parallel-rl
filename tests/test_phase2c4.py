import json,inspect
from pathlib import Path
import numpy as np,pandas as pd,pytest
from src.model import ROOT,sha
from src.rl.stewart_large_motion_env import RUN,StewartLargeMotionResidualEnv
from src.rl.large_motion_observations import FIELDS,LargeObservation
from src.rl.large_motion_rewards import failure_penalty,costs
from src.rl.large_motion_action_adapter import LargeMotionAction

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def test_zero_full_state_and_reset_evidence():
    r=load(RUN/'validation/zero.json');assert r['passed'] and r['paired_physical_steps']==15000 and r['reset_first50_bitwise_equal'];a=np.load(RUN/'validation/zero_env/episode_0000/integration_states.npz')['states'];b=np.load(RUN/'validation/zero_frozen/integration_states.npz')['states'];np.testing.assert_array_equal(a,b)

def test_gym_interface_and_normalisation_contract():
    r=load(ROOT/'results/phase2c4_env_validation.json');assert r['passed'] and r['feature_dim']==53 and r['decision_steps']==50 and r['horizon_steps']==15000;assert sum(n for _,n in FIELDS)==53
    d=np.load(RUN/'validation/uniform/episode_0000/decisions.npz');assert d['observations'].shape==(300,53) and np.isfinite(d['observations']).all();assert (abs(d['observations'])<10).all()

def test_53_frame_no_hidden_inputs_and_history_is_causal():
    source=inspect.getsource(LargeObservation.frame)
    for term in ['.profile','.lag','actuator_gainprm','actuator_force','external_world','F_drive']:assert term not in source
    h=LargeObservation(3);h.history.extend([np.ones(53)*j for j in [1,2,3]]);np.testing.assert_array_equal(h.value(),np.repeat([1,2,3],53))

@pytest.mark.parametrize('pattern',['positive','negative','uniform','alternating'])
def test_live_mapping_budget_slew_and_external_channels(pattern):
    d=pd.read_csv(RUN/f'validation/{pattern}/episode_0000/physics.csv.gz');f=d[[f'F_command_{i}' for i in range(6)]].to_numpy();c=d[[f'F_C0_{i}' for i in range(6)]].to_numpy();a=f-c;assert abs(a).max()<=5+1e-8 and abs(np.diff(a,axis=0)/.001).max()<=20+1e-6;assert abs(d[[f'force_{i}' for i in range(6)]].to_numpy()).max()<400;assert abs(d.rl_requested_Fz_N).max()<=.1+1e-9;expected=np.where((d.input_step>=5200)&(d.input_step<11000),-5.5,0);np.testing.assert_allclose(d.external_world_2,expected,atol=1e-9)
    np.testing.assert_allclose(d.ff_probe_wrench_2,d.rl_requested_Fz_N,atol=1e-12)
    for i in range(6):np.testing.assert_allclose(d[f'F_drive_{i}'],d[f'ctrl_{i}'],atol=1e-12);np.testing.assert_allclose(d[f'force_{i}'],d[f'ctrl_{i}'],atol=1e-12)

def test_reward_independent_recalculation_and_failure_ordering():
    r=load(RUN/'validation/reward_independent.json');assert r['passed'] and max(x['maximum_component_difference'] for x in r['trials'])<1e-9
    for n in [0,1,50,300]:assert failure_penalty(n)<-sum(100*.99**j for j in range(n))

def test_physical_counting_and_termination():
    d=pd.read_csv(RUN/'validation/zero_env/training_samples.csv');assert (d.action_physical_steps==50).all() and d.iloc[299].truncated and not d.iloc[299].terminated;assert d.iloc[299].physical_time_s==pytest.approx(15)

def test_fault_injected_hold_semantics(tmp_path):
    e=StewartLargeMotionResidualEnv(tmp_path/'failure',mode='Unit fault injection, not a real safety failure');e.reset(seed=9);e.owner.hold('unit_injected_fault',failed=True);_,reward,t,u,info=e.step([0]);assert t and not u and not info['is_success'] and reward<0 and info['physical_steps']>=199;e.close()

def test_fixed_seed_short_stochastic_sampling_reproducible(tmp_path):
    import torch
    from stable_baselines3 import PPO
    traces=[]
    for j in range(2):
        e=StewartLargeMotionResidualEnv(tmp_path/str(j),mode='Seed reproducibility unit test');obs,_=e.reset(seed=888);m=PPO('MlpPolicy',e,seed=888,n_steps=256,batch_size=64,device='cpu',policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),log_std_init=-1))
        with torch.no_grad():m.policy.action_net.weight.zero_();m.policy.action_net.bias.zero_()
        actions=[]
        for _ in range(4):
            with torch.no_grad():raw=m.policy(torch.as_tensor(obs.reshape(1,-1)))[0].cpu().numpy()[0]
            actions.append(raw.copy());obs,*_=e.step(raw)
        traces.append((np.asarray(actions),np.array(e.owner.recorder.states)));e.close()
    np.testing.assert_array_equal(traces[0][0],traces[1][0]);np.testing.assert_array_equal(traces[0][1],traces[1][1])

def test_ppo_smoke_actual_update_and_reload():
    r=load(RUN/'smoke_validation.json');assert r['passed'] and r['decisions']==256 and r['physical_steps']==12800 and r['reload_parameter_equal'];assert r['update']['actor_change_norm']>0 and r['update']['critic_change_norm']>0

def test_saved_SB3_samples_equal_executed_actions():
    for seed in [42,123,2026]:
        folder=RUN/f'training/seed{seed}';d=pd.read_csv(folder/'training_samples.csv')
        for file in sorted(folder.glob('rollout_*.npz')):
            with np.load(file) as r:
                end=int(r['global_decision_end']);np.testing.assert_allclose(r['actions'].reshape(-1),d.raw_action.iloc[end-256:end],atol=1e-7);np.testing.assert_allclose(r['physical_rewards'].reshape(-1),d.reward.iloc[end-256:end],atol=1e-8)

@pytest.mark.parametrize('seed',[42,123,2026])
def test_seed_budget_checkpoint_and_real_optimisation(seed):
    r=load(RUN/f'training/seed{seed}/training_summary.json');assert r['decisions']==4096 and r['physical_steps']==204800 and r['updates']==16 and r['actor_updated'] and r['critic_updated'];assert sha(r['model'])==r['model_sha256']

def test_viewer_invariance_and_videos():
    assert load(RUN/'validation/viewer.json')['bitwise_equal']
    for kind in ['training','evaluation']:assert load(RUN/f'video_{kind}_validation.json')['passed']

def test_baselines_and_historical_hashes_frozen():
    from phase2c6_history_audit import check_hashes
    check_hashes(ROOT, load(RUN/'history_freeze.json'))
    p=load(ROOT/'results/phase2c3b_feedforward_parameters.json');assert p['kv_N_s_m']==570.4530241151673 and p['ka_kg']==50.1853287954833

def test_final_tasks_frozen_and_evaluation_deterministic():
    protocol=load(RUN/'protocol_frozen.json');assert protocol['evaluation']['protocol_frozen_before_training'];d=pd.read_csv(ROOT/'results/phase2c4_controller_comparison.csv');assert len(d)==24 and d[d.method.str.startswith('PPO')].source.eq('Deterministic Checkpoint Evaluation').all()
