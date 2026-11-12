"""Real physics and training-boundary correctness before formal PPO admission."""
import argparse
import numpy as np
import pandas as pd
import torch
from phase2c9b_protocol import ROOT,RUN,SEEDS,cfg,read,write_json,sha
from src.rl.phase2c8_env import TaskSampler
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation


def interfaces():
    task=cfg('tasks')['training'][-1];results=[]
    for mode in ('single','history'):
        base=StewartAdaptiveEnv(RUN/'validation'/mode,[task],seed=42,version='v2_control')
        env=HistoryObservation(base,5) if mode=='history' else base
        try:
            obs,_=env.reset(seed=12);initial=obs.copy();body=base.owner.model.body('upper').id
            assert base.owner.nominal.body_mass[body]==20 and base.owner.model.body_mass[body]==22
            np.testing.assert_allclose(base.owner.model.body_inertia[body],base.owner.nominal.body_inertia[body]*1.1)
            frames=[initial[-58:].copy()]*5;rows=[]
            for i in range(50):
                obs,reward,term,trunc,_=env.step(np.array([.2,-.2]) if i>=25 else np.zeros(2))
                frames=frames[1:]+[obs[-58:].copy()]
                if mode=='history':np.testing.assert_array_equal(obs,np.concatenate(frames))
                assert not term and not trunc and np.isfinite(reward)
                rows.append(np.r_[obs[-58:],reward,base.owner.data.qpos,base.owner.data.qvel,base.owner.data.ctrl])
            obs,_=env.reset(seed=99);np.testing.assert_array_equal(obs,initial)
            results.append(np.asarray(rows))
            try:env.step([np.nan,0]);raise AssertionError('NaN accepted')
            except ValueError:pass
        finally:env.close()
    np.testing.assert_array_equal(*results)
    for seed in SEEDS:
        a=TaskSampler(seed);b=TaskSampler(seed)
        assert [a.choose(cfg('tasks')['training'])[0] for _ in range(20)]==[b.choose(cfg('tasks')['training'])[0] for _ in range(20)]
        state=a.snapshot();expected=[a.choose(cfg('tasks')['training'])[0] for _ in range(20)]
        restored=TaskSampler(seed,state);assert expected==[restored.choose(cfg('tasks')['training'])[0] for _ in range(20)]
    write_json(RUN/'interface_validation.json',dict(passed=True,physical_steps=2000,
        single_history_state_reward_bitwise_equal=True,reset_equal=True,causal_history_order=True,nominal_plant_separate=True,task_rng_resume=True))


def resume():
    from train_phase2c9b_ppo import train
    torch.set_num_threads(1);root=RUN/'validation/resume'
    previous=train('history_5frame',42,512,custom_root=root/'prefix')
    a=train('history_5frame',42,512,previous,custom_root=root/'branch_a')
    b=train('history_5frame',42,512,previous,custom_root=root/'branch_b')
    assert a['updates'][-1]['policy_hash']==b['updates'][-1]['policy_hash']
    pa=root/'branch_a/from_000512';pb=root/'branch_b/from_000512'
    pd.testing.assert_frame_equal(pd.read_csv(pa/'training_samples.csv'),pd.read_csv(pb/'training_samples.csv'),check_exact=True)
    with np.load(next(pa.glob('episode_*/integration_states.npz'))) as x,np.load(next(pb.glob('episode_*/integration_states.npz'))) as y:
        np.testing.assert_array_equal(x['states'],y['states'])
    write_json(RUN/'resume_validation.json',dict(passed=True,test_decisions=1536,physical_steps=30720,
        boundary_resume_bitwise=True,arbitrary_mid_episode_resume=False))


def admission():
    import time
    deadline=time.monotonic()+1800
    while len(list((RUN/'preflight_z50').rglob('metrics.json')))<8:
        if time.monotonic()>deadline:raise TimeoutError('Explicit pure 50 mm Z supplemental preflight incomplete')
        time.sleep(10)
    extra=[read(p) for p in (RUN/'preflight_z50').rglob('metrics.json')]
    assert len(extra)==8 and all(r['success'] and not r['failure'] for r in extra)
    assert read(RUN/'interface_validation.json')['passed'] and read(RUN/'resume_validation.json')['passed']
    rows=[read(p) for p in (RUN/'preflight').rglob('metrics.json')]
    expected=len(cfg('tasks')['layers'])+2*sum(len(cfg('tasks')[g]) for g in ('training','development','final'))
    assert len(rows)==expected,(len(rows),expected)
    rejected=[r for r in rows if not r['success'] or r['failure']]
    write_json(RUN/'admission.json',dict(passed=not rejected,trials=len(rows),supplemental_pure_z50=len(extra),rejected=rejected))
    assert not rejected,[(r['task_id'],r['method'],r['failure']) for r in rejected]
    for p,h in read(RUN/'config_freeze.json').items():assert sha(p)==h
    probes=[read(p) for p in (RUN/'authority').glob('*/audit.json')]
    assert len(probes)==3 and all(p['same_prefix'] and p['measurable_effect'] for p in probes)
    assert (RUN/'history.jsonl').exists()
    write_json(RUN/'training_gate.json',dict(passed=True,preflight_trials=len(rows),ceiling_effect_not_blocking=True,
        reward='frozen V2',authority_audited=True,safety_limits=[.001,20,58,290,.05,5,20,400]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['interfaces','resume','admission'])
    globals()[parser.parse_args().mode]()
