"""Post-run acceptance: these require actual completed artifacts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase2c7_protocol import ROOT,RUN,SEEDS,read
from src.rl.ppo_trace import parameter_hash


@pytest.mark.parametrize('seed',SEEDS)
def test_real_updates_actions_and_reload(seed):
    root=RUN/'training'/f'seed{seed}'
    summary=read(root/'continue/summary.json')
    assert summary['decisions']==32768 and summary['physical_steps']==655360
    assert summary['updates']==64 and summary['failures']==0 and summary['error'] is None
    model=PPO.load(ROOT/f'models/phase2c7/seed{seed}/final.zip',device='cpu')
    assert parameter_hash(model.policy)==summary['final_hash']
    assert model.observation_space.shape==(58,)
    initial=PPO.load(ROOT/f'models/phase2c7/seed{seed}/initial.zip',device='cpu')
    np.testing.assert_array_equal(initial.predict(np.zeros(58,np.float32),deterministic=True)[0],[0,0])
    for phase in ('pilot','continue'):
        samples=pd.read_csv(root/phase/'training_samples.csv')
        assert samples.physical_steps.eq(20).all() and not samples.terminated.any()
        for npz in (root/phase).glob('rollout_*.npz'):
            update=int(npz.stem.split('_')[-1]);part=samples[(samples.global_decision>(update-1)*512)&(samples.global_decision<=update*512)]
            with np.load(npz) as b:
                    # csv.writer uses NumPy float32's shortest round-trip text.
                    # Restore the recorded sampling dtype, then require exact equality.
                    np.testing.assert_array_equal(part[['raw_roll','raw_pitch']].to_numpy(dtype=np.float32),b['actions'].reshape(-1,2))
    updates=read(root/'continue/updates.json')
    assert all(u['actor_delta']>0 and u['critic_delta']>0 and u['physical_steps']==u['decisions']*20 for u in updates)


def test_zero_and_window_do_not_change_dynamics():
    gate=read(RUN/'training_gate.json')
    assert gate['passed'] and gate['zero_residual_bitwise'] and gate['equivalence_physical_steps']==15000
    assert read(RUN/'qt_equivalence.json')['bitwise_equal']


@pytest.mark.parametrize('kind',['dynamic_ppo_training','classical_vs_ppo','dynamic_disturbance_demo'])
def test_live_video_source_and_motion(kind):
    v=read(RUN/f'video_{kind}.json');assert v['frames']>100
    assert all(c['matched_state_hashes'] and c['frame_change_peak']>0 for c in v['clips'])
    assert any(c['z_motion_mm']>49 for c in v['clips'])


def test_all_old_files_preserved():
    result=read(RUN/'history_after.json')
    assert result['passed'] and not result['errors'] and result['matched']==16395
