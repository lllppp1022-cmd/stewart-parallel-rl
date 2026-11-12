"""Actual experiment acceptance, including raw 1 ms physical evidence."""
import inspect
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,sha
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv


def test_history_preserved():
    assert read(RUN/'history_verification.json')['changes']==[]


def test_config_frozen():
    for path,digest in read(RUN/'config_freeze.json').items():assert sha(path)==digest


def test_mass_inertia_isolated():
    p=next((RUN/'preflight').glob('Train_m22.0_6/baseline/seed0/0/episode_*/phase2c9_plant.json'))
    m=read(p);assert m['nominal']['platform_mass_kg']==20 and m['actual']['platform_mass_kg']==22
    np.testing.assert_allclose(m['actual']['platform_inertia_kg_m2'],np.array(m['nominal']['platform_inertia_kg_m2'])*1.1)


@pytest.mark.parametrize('forbidden',['mass_kg','plant','external','tau','last_drive'])
def test_no_privileged_observation(forbidden):
    assert re.search(r'\b'+re.escape(forbidden)+r'\b',inspect.getsource(StewartDynamicAttitudeEnv.observe)) is None


def test_history_order_reset_equivalence():
    d=read(RUN/'interface_validation.json');assert d['single_history_state_reward_bitwise_equal'] and d['reset_equal'] and d['causal_history_order']


def test_rng_and_boundary_restore():
    assert read(RUN/'resume_validation.json')['boundary_resume_bitwise']


@pytest.mark.parametrize('mass',[20.,21.2,22.])
def test_action_causality(mass):
    d=read(RUN/'authority'/str(mass)/'audit.json');assert d['same_prefix'] and d['measurable_effect']


def test_physical_admission():
    d=read(RUN/'admission.json');assert d['passed'] and d['trials']==60 and not d['rejected']


@pytest.mark.parametrize('method,dimension',[('single_frame',58),('history_5frame',290)])
def test_policy_dimensions_and_parameters(method,dimension):
    selection=read(RUN/'final_selection.json')
    for seed in SEEDS:
        item=selection['models'][f'{method}/seed{seed}'];model=PPO.load(item['path'],device='cpu')
        assert sha(item['path'])==item['sha'] and model.observation_space.shape==(dimension,) and model.action_space.shape==(2,)


def test_training_audited_and_matched():
    d=read(RUN/'training_audit.json');assert d['passed'] and d['matched_task_sequences'] and d['all_gae_recomputed']
    n=read(RUN/'final_selection.json')['common_budget'];assert d['formal_decisions']==6*n and d['formal_physics']==6*n*20


def test_actor_critic_updated():
    d=pd.read_csv(ROOT/'results/phase2c9b_update_events.csv')
    assert d.actor_delta.gt(0).all() and d.critic_delta.gt(0).all()
    assert np.isfinite(d[['train/value_loss','train/approx_kl']]).all().all()


def test_final_isolation():
    d=read(RUN/'final_selection.json');assert d['frozen'] and d['no_best_seed_selection']
    assert read(RUN/'pilot_gate.json')['no_final_policy_used']


def test_final_physical_limits():
    d=pd.read_csv(ROOT/'results/phase2c9b_generalization.csv');assert len(d)==72
    assert d.add_peak_N.max()<=5+1e-8 and d.add_rate_N_s.max()<=20+1e-6 and d.force_peak_N.max()<=400+1e-8
    assert set(d[d.method.isin(METHODS)].seed)==set(SEEDS)
    assert d.evaluation_gradient_updates.eq(0).all() and d.parameters_unchanged.all()


def test_reward_recomputation():
    d=pd.read_csv(ROOT/'results/phase2c9b_reward_recomputation.csv');assert d.reward_error.max()<1e-10 and d.component_error.max()<1e-10


def test_final_iae_recomputation():
    for path in (RUN/'final').rglob('metrics.json'):
        m=read(path);d=pd.read_csv(Path(m['folder'])/'physics.csv.gz',usecols=['input_step','orientation_error_deg'])
        for label,key in [('load','on'),('unload','off')]:
            n=round(m['task'][key]*1000);value=d.loc[d.input_step.between(n,n+999),'orientation_error_deg'].sum()*.001
            assert value==pytest.approx(m[label+'_iae_deg_s'],abs=1e-12)


def test_old_failure_retained():
    assert 'find_spec' in (ROOT/'tests/test_phase0.py').read_text(encoding='utf8')


def test_viewer_preserves_physics():
    proofs=[read(p) for p in (RUN/'viewer').glob('*/viewer_validation.json')]
    assert {p['method'] for p in proofs}==set(METHODS)
    assert all(p['headless_bitwise_equal'] and p['window_closed'] for p in proofs)


@pytest.mark.parametrize('kind',['unknown_mass','history_training','single_vs_history'])
def test_actual_video_frames(kind):
    v=read(RUN/f'video_{kind}.json');assert sha(v['path'])==v['sha'] and v['no_policy_resampling'] and v['frames']>100
    assert all(c['state_time_sha_match'] and c['pixel_change']>0 and c['qpos_range']>0 for c in v['clips'])
    assert all(c['training']==(kind=='history_training') for c in v['clips'])
