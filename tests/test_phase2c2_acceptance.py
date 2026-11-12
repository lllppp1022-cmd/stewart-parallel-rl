"""Tests against produced bounded research evidence, not synthetic results."""
import json
import numpy as np,pandas as pd
from src.model import ROOT,sha
R=ROOT/'results/runs/phase2c2_20261030_v1'

def test_amplitude_and_action_ladders_actual():
    a=json.loads((R/'amplitude_ladder.json').read_text(encoding="utf-8"));b=json.loads((R/'action_diagnostics.json').read_text(encoding="utf-8"));assert len(a)==6 and len(b)==24
    assert all(x['success'] and x['physical_steps']==6000 and x['warning_count']==0 and x['native_contact_observations']==0 for x in a+b)
    assert all(x['z_ptp_mm']>2.9 for x in b)

def test_recording_does_not_change_actual_ppo():
    r=json.loads((R/'observer_equivalence_v2.json').read_text(encoding="utf-8"));assert r['passed'] and r['episode_endings_exact'] and all(r['rollout_arrays_exact'].values());assert r['branches'][0]['parameter_hash']==r['branches'][1]['parameter_hash']

def test_seed_budgets_physics_and_updates():
    rs=[json.loads((R/f'seed{s}/training_result.json').read_text(encoding="utf-8")) for s in [42,123]];assert rs[0]['decisions']==rs[1]['decisions'] and rs[0]['decisions'] in [2048,4096]
    for r in rs:
        u=pd.read_csv(R/f'seed{r["seed"]}/ppo_update_events.csv');assert len(u)==r['decisions']//256 and r['physical_steps']==r['decisions']*50
        assert (u.actor_change_norm>0).all() and (u.critic_change_norm>0).all();assert np.array_equal(u.global_decisions,np.arange(256,r['decisions']+1,256))

def test_raw_rewards_actions_and_counts_recompute():
    r=json.loads((R/'training_journal_audit.json').read_text(encoding="utf-8"));assert r['passed'];assert all(t['reward_cap_fraction']==0 for t in r['totals'])

def test_final_freeze_and_evaluation():
    f=json.loads((R/'final_evaluation_frozen.json').read_text(encoding="utf-8"));assert sha(R/'protocol_frozen.json')==f['protocol_sha256']
    for x in f['models']:assert sha(x['checkpoint'])==x['sha256']
    rows=json.loads((R/'final_evaluation_results.json').read_text(encoding="utf-8"));assert len(rows)==10 and len({r['label'] for r in rows})==5

def test_history_and_actual_video():
    h=json.loads((R/'history_preservation.json').read_text(encoding="utf-8"));v=json.loads((ROOT/'results/phase2c2_recording_validation.json').read_text(encoding="utf-8"));assert h['passed'] and h['files_checked']>=2989;assert v['passed'] and v['decoded_frames']>0
    assert all(x['all_qpos_hashes_match'] and x['maximum_slide_displacement_m']>.002 for x in v['sources'])
