"""Evidence-based acceptance: these tests require genuinely generated artifacts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,POINTS,NAMES,read,sha,checkpoint

@pytest.fixture(scope='module')
def data():return pd.read_csv(ROOT/'results/phase2c7v_reward_components.csv')
@pytest.fixture(scope='module')
def episodes():return pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv')
@pytest.fixture(scope='module')
def evaluation():return pd.read_csv(ROOT/'results/phase2c7v_checkpoint_evaluation.csv')

def test_01_frozen_weights():
    inventory=read(RUN/'checkpoint_inventory.json');assert len(inventory)==201
    assert all(sha(r['path'])==r['sha256'] for r in inventory)

def test_02_original_training_sha():
    manifest=read(RUN/'history_before.json')
    for s in SEEDS:
        for phase in ('pilot','continue'):
            p=OLD/f'training/seed{s}/{phase}/training_samples.csv';assert sha(p)==manifest[str(p.relative_to(ROOT))]

def test_03_complete_vs_fragments(episodes):
    assert len(episodes)==135 and episodes.complete.sum()==126
    assert (episodes[episodes.complete].physical_seconds==episodes[episodes.complete].reference_seconds).all()
    assert (episodes[~episodes.complete].physical_seconds<episodes[~episodes.complete].reference_seconds).all()

def test_04_stage_alignment(data):
    expected=np.where(data.global_decision<=4096,'A',np.where(data.global_decision<=8192,'B','C'))
    np.testing.assert_array_equal(data.stage,expected)

def test_05_decision_and_physical_counts(data):
    assert len(data)==98304 and data.physical_steps.sum()==1966080
    for _,g in data.groupby('seed'):np.testing.assert_array_equal(g.global_decision,np.arange(1,32769))

def test_06_return_is_sum_not_mean(data,episodes):
    for r in episodes.itertuples():
        g=data[(data.seed==r.seed)&(data.phase==r.phase)&(data.episode==r.episode)]
        assert abs(g.reward.sum()-r.episode_return)<1e-8

def test_07_independent_physical_reward_recomputation():
    a=read(RUN/'reward_audit_summary.json');assert a['physical_steps']==1966080 and a['max_abs_difference']<1e-12 and a['anomalies']==0

def test_08_bootstrap_is_separate(data):
    b=pd.read_csv(ROOT/'results/phase2c7v_bootstrap_audit.csv');assert len(b)==126
    assert (abs(b.bootstrap_adjustment)>0).all()
    for r in b.itertuples():
        g=data[(data.seed==r.seed)&(data.global_decision==r.decision)].iloc[0]
        assert g.truncated and abs(g.reward-r.environment_reward)<1e-10

def test_09_five_components_sum(data):
    np.testing.assert_allclose(data[[f'reward_{n}' for n in NAMES]].sum(axis=1)+data.terminal_penalty,data.reward,atol=1e-12)

def test_10_no_fabricated_missing_points(evaluation):
    for s in SEEDS:assert tuple(evaluation[evaluation.seed==s].decisions)==POINTS
    assert len(evaluation)==28

def test_11_raw_data_retained_and_trends_optional():
    g=read(RUN/'analysis_gui_validation.json');assert g['combinations']==144 and g['read_only']
    source=(ROOT/'scripts/view_phase2c7v_training.py').read_text();assert 'self.raw.isChecked()' in source and 'self.trend.isChecked()' in source

def test_12_checkpoint_sha_matches_evaluation(evaluation):
    for r in evaluation[evaluation.seed!=0].itertuples():assert sha(r.model_path)==r.model_sha

def test_13_no_gradient_updates(evaluation):
    assert evaluation.parameters_unchanged.all() and evaluation.gradient_updates.eq(0).all()
    source=(ROOT/'scripts/evaluate_phase2c7v_checkpoints.py').read_text();assert '.learn(' not in source and '.backward(' not in source

def test_14_identical_reference_load(evaluation):
    columns=[f'p_ref_{i}' for i in range(3)]+[f'external_world_{i}' for i in range(6)]+['time_s']
    baseline=None
    for r in evaluation.itertuples():
        frame=pd.read_csv(Path(r.folder)/'physics.csv.gz',usecols=columns).to_numpy()
        if baseline is None:baseline=frame
        # World/body conversion roundoff depends on the actual posture (~5e-15).
        # The prescribed trajectory/load is identical; no physical tolerance is relaxed.
        np.testing.assert_allclose(frame,baseline,rtol=0,atol=1e-12)

def test_15_metrics_from_full_physics(evaluation):
    for r in evaluation.itertuples():
        d=pd.read_csv(Path(r.folder)/'physics.csv.gz',usecols=['orientation_error_deg','p_2','p_ref_2'])
        assert len(d)==r.physical_steps==15000
        assert abs(d.orientation_error_deg.max()-r.so3_peak_deg)<1e-10
        assert abs(np.sqrt(np.mean((d.p_2-d.p_ref_2)**2))*1000-r.z_rmse_mm)<1e-9

def test_16_development_final_separation(evaluation):
    assert set(evaluation.task_id)=={'Dev_mix'}
    protocol=read(RUN/'protocol.json');assert protocol['original_final_selection_unchanged']==sha(OLD/'final_selection.json')

def test_17_observation_order():
    groups=read(RUN/'observation_groups.json');assert sum(g['dimension'] for g in groups)==58
    assert [i for g in groups for i in range(g['start'],g['end_exclusive'])]==list(range(58))

def test_18_observation_units():
    g=read(RUN/'observation_groups.json');assert g[0]['unit']=='rad' and g[0]['scale']==[.001]*3
    assert g[1]['unit']=='rad/s' and g[1]['scale']==[.02]*3

def test_19_request_accept_not_identical():
    a=pd.read_csv(ROOT/'results/phase2c7v_action_audit.csv');assert len(a)==12 and (a.request_accept_error_rms_Nm>0).all()
    assert a[(a.seed==42)&(a.task_id=='F_roll')].requested_accepted_rms_ratio.iloc[0]<.7

def test_20_force_counterfactual_mapping():
    a=pd.read_csv(RUN/'governor_counterfactual_audit.csv');assert len(a)==12
    assert a.command_max_error_N.max()<1e-10 and a.counter_base_max_error_N.max()<1e-10
    b=pd.read_csv(ROOT/'results/phase2c7v_action_audit.csv');assert b.actual_vs_nominal_mapping_max_Nm.max()<1e-6

def test_21_no_privileged_observation_inputs():
    import inspect
    from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv
    source=inspect.getsource(StewartDynamicAttitudeEnv.observe)
    assert all(k not in source for k in ('external_world','actual_gain','true_mass','tau_s','load_on','load_off','drive.'))

def test_22_gui_matches_headless_physics():
    rows=[read(p) for p in (RUN/'viewer').glob('*/viewer_validation.json')]
    assert len(rows)>=2 and any(r['classical'] for r in rows) and any(not r['classical'] for r in rows)
    assert all(r['bitwise_equal_headless'] and r['window_closed'] and r['single_step'][0]['after']==20 for r in rows)

@pytest.mark.parametrize('name,frames', [('training',2250),('comparison',1400)])
def test_23_video_time_state_proof(name,frames):
    r=read(RUN/f'{name}_video_validation.json');assert r['frames']==frames and r['seconds']==frames/25
    assert sha(r['path'])==r['sha'] and r['resolution']==[1920,1080]
    assert all(c['state_timestamp_hashes_match'] and c['peak_rgb_change']>0 and c['qpos_range']>0 for c in r['clips'])

def test_24_source_labels():
    train=read(RUN/'training_video_validation.json');evaluation=read(RUN/'comparison_video_validation.json')
    assert train['future_history_gray'] and all('TRAINING' in c['source_label'] for c in train['clips'])
    assert all('EVALUATION' in c['source_label'] for c in evaluation['clips'])

def test_25_history_manifest():
    h=read(RUN/'history_after.json');assert h['passed'] and h['checked']==len(read(RUN/'history_before.json')) and not h['differences']

def test_26_real_gae_and_value_targets():
    a=pd.read_csv(RUN/'value_recomputation.csv');assert len(a)==192
    assert a.gae_internal_max_error.max()<1e-4 and a.return_value_plus_advantage_max_error.max()<1e-4

def test_27_derived_tensorboard_complete_episode_count():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for s in SEEDS:
        acc=EventAccumulator(str(ROOT/f'results/phase2c7v_tensorboard/seed{s}'),size_guidance={'scalars':0});acc.Reload()
        assert len(acc.Scalars('training/complete_episode_return'))==42
        assert len(acc.Scalars('training/reward_per_decision'))==32768
        assert len(acc.Scalars('training/incomplete_fragment_return'))==3

def test_28_all_actor_critic_updates_real():
    u=pd.read_csv(RUN/'updates.csv');assert len(u)==192
    assert u.actor_delta.gt(0).all() and u.critic_delta.gt(0).all() and u.adam_steps.sum()==15360
