"""Acceptance against actually produced Pilot artifacts, no substitute training."""
import json
from pathlib import Path
import numpy as np,pandas as pd,pytest,torch
from src.model import ROOT,sha
from src.rl.ppo_trace import TracedPPO,parameter_hash
from src.rl.evaluation import preserve_rng
from src.rl.stewart_residual_env import env_config

RUN=ROOT/'results/runs/phase2c1_20261029_v1'

@pytest.mark.parametrize('seed',[42,123,2026])
def test_actual_counts_updates_samples_and_checkpoint(seed):
    folder=RUN/f'seed{seed}';r=json.loads((folder/'training_result.json').read_text());assert r['actual_decisions']==4096 and r['actual_physical_steps']==204800 and r['updates']==16
    assert sha(r['final_checkpoint'])==r['final_sha256'];data=pd.read_csv(folder/'actual_training/training_samples.csv');u=pd.read_csv(folder/'ppo_update_events.csv');assert np.array_equal(data.global_decision,np.arange(1,4097));assert np.array_equal(u.global_decisions,np.arange(256,4097,256));assert (u.actor_change_norm>0).all() and (u.critic_change_norm>0).all() and u.simulator_paused_during_update.all()
    for idx,p in enumerate(sorted(folder.glob('rollout_*.npz'))):
        with np.load(p) as z:
            sl=data.iloc[idx*256:(idx+1)*256];np.testing.assert_allclose(z['actions'].reshape(-1),sl.raw_action,atol=1e-7);np.testing.assert_array_equal(np.clip(z['actions'].reshape(-1),-1,1).astype(np.float32),sl.clipped_action.to_numpy(np.float32));np.testing.assert_allclose(z['physical_rewards'],sl.reward,atol=1e-12)
    # Episode2 actually spans the first update; it must not be relabelled fixed-policy.
    assert data[data.episode==2].policy_version.nunique()==2
    episodes=json.loads((folder/'actual_training/episode_summary.json').read_text());assert sum(x['complete'] for x in episodes)==34;assert episodes[-1]['reason']=='budget_cutoff' and episodes[-1]['decisions']==16 and not episodes[-1]['complete']
    with preserve_rng():model=TracedPPO.load(r['final_checkpoint'],device='cpu');assert parameter_hash(model.policy)==r['final_parameter_hash']

def test_new_tasks_and_fixed_normalization():
    c=env_config();train={tuple(x[k] for k in ['tau_s','Fz_N','on_step','off_step']) for x in c['training_tasks']};final={tuple(x[k] for k in ['tau_s','Fz_N','on_step','off_step']) for x in c['final_tasks']};assert not train&final
    frozen=json.loads((RUN/'final_evaluation_frozen.json').read_text());assert not frozen['VecNormalize'];assert all(x['decisions']==4096 for x in frozen['models'])

def test_independent_reward_gae_and_history_preservation():
    r=pd.read_csv(ROOT/'results/phase2c1_training_reward_audit.csv');assert len(r)==12288 and r.error.max()<1e-8
    gae=pd.read_csv(ROOT/'results/phase2c1_gae_audit.csv');assert len(gae)==48 and gae.max_GAE_recompute_error.max()<2e-4
    h=json.loads((ROOT/'results/phase2c1_final_acceptance.json').read_text());assert h['historical_files_unchanged'] and h['original_distribution_versions_unchanged']

@pytest.mark.parametrize('kind',['training','evaluation'])
def test_videos_are_actual_RGB_and_causal(kind):
    f=ROOT/f'assets/videos/stewart_phase2c1_ppo_{kind}.mp4';v=json.loads(f.with_suffix('.validation.json').read_text());assert sha(f)==v['sha256'] and v['frames']>100 and v['unique_frames']>1 and v['all_frames_decoded'] and v['all_qpos_hashes_match'] and v['no_future_curves'];assert v['physics_steps_during_render']==v['policy_calls_during_render']==0

def test_initial_deterministic_physics_equals_C4_on_each_final_task():
    for task in env_config()['final_tasks']:
        with np.load(RUN/'evaluation/classical'/task['id']/'C4/integration_states.npz') as baseline:
            for seed in [42,123,2026]:
                with np.load(RUN/f'evaluation/seed{seed}/initial_deterministic'/task['id']/'episode_0000/integration_states.npz') as initial:np.testing.assert_array_equal(baseline['states'],initial['states'])
