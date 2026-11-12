"""Phase8 acceptance, grounded in actual physical/training/evaluation artifacts."""
import inspect
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read,sha
from src.rl.phase2c8_reward import costs
from src.rl.phase2c8_env import Phase2C8Env,TaskSampler
from src.rl.ppo_trace import parameter_hash

@pytest.fixture(scope='module')
def training():return pd.read_csv(ROOT/'results/phase2c8_training_summary.csv')
@pytest.fixture(scope='module')
def final():return pd.read_csv(ROOT/'results/phase2c8_controller_comparison.csv')

def test_01_history_unchanged():
    h=read(RUN/'history_after.json');assert h['passed'] and not h['changes']
def test_02_rng_restore():assert read(RUN/'resume_validation.json')['action_observation_reward_rollout_physics_bitwise_equal']
def test_03_stage_continuity():
    for method in ('v2_control','v3'):
        for seed in SEEDS:
            a=read(RUN/f'training/{method}/seed{seed}/A/task_sequence.json');b=read(RUN/f'training/{method}/seed{seed}/B/task_sequence.json')
            assert a[-1]['sampler_after']==b[0]['sampler_before'] and b[0]['requested_gym_seed'] is None
def test_04_boundary_resume_equal():
    r=read(RUN/'resume_validation.json');assert r['passed'] and r['training_decisions_test_only']==1536 and r['stage_boundary_reset_explicit']
def test_05_matched_task_sequence():
    for seed in SEEDS:
        for stage in ('A','B'):
            assert read(RUN/f'training/v2_control/seed{seed}/{stage}/task_sequence.json')==read(RUN/f'training/v3/seed{seed}/{stage}/task_sequence.json')
def zero_row():return dict(orientation_error_deg=0.,**{f'p_{i}':0. for i in range(3)},**{f'p_ref_{i}':0. for i in range(3)},**{f'twist_actual_{i}':0. for i in (3,4,5)},**{f'twist_ref_{i}':0. for i in (3,4,5)})
def test_06_v3_orientation_formula():
    r=dict(zero_row(),orientation_error_deg=.006);a=costs(r,np.zeros(2),np.zeros(2))[0][0];assert a==pytest.approx(.5*(np.hypot(1,.01)-.01)+.5)
def test_07_scale_and_zero():
    assert cfg('reward_v3')['orientation_scale_deg']==.006 and costs(zero_row(),np.zeros(2),np.zeros(2))[0].sum()==0
def test_08_all_physical_rewards_recompute():
    a=pd.read_csv(ROOT/'results/phase2c8_reward_recomputation.csv');assert a.physical_steps.sum()==983040 and a.reward_error.max()<1e-10 and a.component_error.max()<1e-10
def test_09_reward_no_privileged_fields():
    s=inspect.getsource(costs);assert all(t not in s for t in ('external','torque','load_time','on_step','off_step','mass'))
def test_10_observation_still_58():
    for m in ('v2_control','v3'):
        p=PPO.load(ROOT/f'models/phase2c8/{m}/seed42/checkpoint_008192.zip',device='cpu');assert p.observation_space.shape==(58,)
def test_11_two_action_dimensions():
    for m in ('v2_control','v3'):
        p=PPO.load(ROOT/f'models/phase2c8/{m}/seed42/checkpoint_008192.zip',device='cpu');assert p.action_space.shape==(2,) and np.all(p.action_space.high==1)
def test_12_jacobian_marginal_mapping():
    d=pd.read_csv(ROOT/'results/phase2c8_action_audit.csv');assert len(d)==24 and d.command_recompute_error_N.max()<1e-9 and d.counterfactual_recompute_error_N.max()<1e-9
def test_13_shared_amplitude(final):assert final.add_peak_N.max()<=5+1e-8
def test_14_shared_slew(final):assert final.add_rate_N_s.max()<=20+1e-6
def test_15_native_force(final):assert final.force_peak_N.max()<=400+1e-8
def test_16_zero_equivalence():assert read(RUN/'zero_equivalence.json')['v2_v3_zero_match_historical_baseline_bitwise']
def test_17_reset_no_leak():assert read(RUN/'zero_equivalence.json')['reset_equal']
def test_18_termination_and_failure_penalty():
    d=pd.read_csv(ROOT/'results/phase2c8_training_samples.csv');assert not d.terminated.any() and d.truncated.sum()==60
    assert read(RUN/'reward_gate.json')['failure_discount_bound_checked_remaining']==[0,800]
def test_19_actor_critic_update():
    d=pd.read_csv(ROOT/'results/phase2c8_critic_diagnostics.csv');assert len(d)==96 and d.actor_delta.gt(0).all() and d.critic_delta.gt(0).all() and d.adam_steps.sum()==7680
def test_20_save_reload_matched_initials(training):
    for seed in SEEDS:
        initial=[]
        for method in ('v2_control','v3'):
            p=PPO.load(ROOT/f'models/phase2c8/{method}/seed{seed}/checkpoint_008192.zip',device='cpu');expected=training[(training.method==method)&(training.seed==seed)].iloc[0].final_parameter_hash;assert parameter_hash(p.policy)==expected
            initial.append(parameter_hash(PPO.load(ROOT/f'models/phase2c8/{method}/seed{seed}/checkpoint_000000.zip',device='cpu').policy))
        assert initial[0]==initial[1]
def test_21_normalization_physics():assert read(RUN/'normalization_validation.json')['physics_bitwise_equal']
def test_22_normalization_statistics():
    d=read(RUN/'normalization_validation.json');assert d['statistics_reload_equal'] and d['evaluation_statistics_frozen']
def test_23_no_double_observation_normalization():assert not cfg('return_norm')['norm_obs'] and not read(RUN/'normalization_validation.json')['obs_normalization']
def test_24_matched_budget(training):assert len(training)==6 and training.new_decisions.eq(8192).all() and training.physical_steps.eq(163840).all()
def test_25_final_isolation():
    selection=read(RUN/'final_selection.json');assert selection['evaluation_sha']==sha(ROOT/'configs/phase2c8_evaluation.yaml') and selection['no_best_seed_or_checkpoint_selection']
    assert read(RUN/'pilot_gate.json')['no_final_policy_evaluated']
def test_26_iae_independent(final):
    for r in final.itertuples():
        m=read(Path(r.folder).parent/'metrics.json');d=pd.read_csv(Path(r.folder)/'physics.csv.gz',usecols=['input_step','orientation_error_deg'])
        for window,edge in [('load','on'),('unload','off')]:
            n=round(m['task'][edge]*1000);value=float(d.loc[d.input_step.between(n,n+999),'orientation_error_deg'].sum()*.001);assert value==pytest.approx(m[window+'_iae_deg_s'],abs=1e-12)
def test_27_peak_and_velocity(final):
    assert np.isfinite(final[['so3_peak_deg','omega_peak_rad_s','roll_error_peak_deg','pitch_error_peak_deg']]).all().all()
    assert final.gradient_updates.eq(0).all() and final.parameters_unchanged.all()
def test_28_actual_force_and_xyz(final):assert np.isfinite(final[['xyz_rmse_mm','force_squared_integral_N2_s','force_rate_N_s']]).all().all()
def test_29_protection_unchanged():
    gate=read(RUN/'training_gate.json');assert gate['new_joint_preflight']==14 and gate['control_limits']==[.001,20,58,2,.05,5,20,400]
def test_30_all_raw_physical_cost_rows(training):assert training.physical_steps.sum()==pd.read_csv(ROOT/'results/phase2c8_reward_recomputation.csv').physical_steps.sum()
def test_31_gui_equivalence():
    proofs=[read(p) for p in (RUN/'viewer').glob('*/viewer_validation.json')];assert proofs and all(p['bitwise_equal'] and p['window_closed'] and p['single_step']==[[0,20]] for p in proofs)
@pytest.mark.parametrize('kind',['training','reward_comparison','attitude_stability'])
def test_32_video_provenance(kind):
    v=read(RUN/f'video_{kind}.json');assert sha(v['path'])==v['sha'] and v['frames']>100 and v['no_policy_resampling']
    assert all(c['timestamp_state_hash_match'] and c['pixel_change']>0 and c['qpos_range']>0 for c in v['clips'])
    assert all(('TRAINING' if kind=='training' else 'EVALUATION') in c['source_label'] for c in v['clips'])
def test_33_old_video_sha():
    before=read(RUN/'history_before.json')
    for name,h in before.items():
        if name.startswith('assets') and 'phase2c7' in name and name.endswith('.mp4'):assert sha(ROOT/name)==h
def test_34_negative_gate_preserved():
    g=read(RUN/'pilot_gate.json');assert not g['passed'] and not read(RUN/'normalization_admission.json')['passed']
    assert read(RUN/'final_selection.json')['budget']==8192
