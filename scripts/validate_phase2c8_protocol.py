"""Gates for finite physical tests and exact curriculum-boundary resume."""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from phase2c8_protocol import ROOT,RUN,OLD,SEEDS,cfg,read,write_json,sha
from src.rl.phase2c8_env import TaskSampler,Phase2C8Env
from src.rl.ppo_trace import parameter_hash
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv

def sampler_test():
    tasks=cfg('training')['curriculum'];rows=[]
    for seed in SEEDS:
        one=TaskSampler(seed);first=[one.choose(tasks['A'])[0]['id'] for _ in range(7)];state=one.snapshot();expected=[one.choose(tasks['B'])[0]['id'] for _ in range(9)]+[one.choose(tasks['C'])[0]['id'] for _ in range(12)]
        restored=TaskSampler(seed,state);actual=[restored.choose(tasks['B'])[0]['id'] for _ in range(9)]+[restored.choose(tasks['C'])[0]['id'] for _ in range(12)]
        assert expected==actual;rows.append(dict(seed=seed,A=first,continued=expected,resumed=actual,state=state))
    write_json(RUN/'sampler_validation.json',dict(passed=True,results=rows))

def resume_test():
    from train_phase2c8_reward_ablation import train_stage
    torch.set_num_threads(1);root=RUN/'validation/resume'
    model,state=train_stage('v2_control',42,'A_probe',512,custom_root=root/'prefix')
    continuous,cs=train_stage('v2_control',42,'B_probe',512,state,model,custom_root=root/'continuous')
    hash_continuous=parameter_hash(continuous.policy)
    restored,rs=train_stage('v2_control',42,'B_probe',512,state,custom_root=root/'restored')
    assert hash_continuous==parameter_hash(restored.policy)
    a=pd.read_csv(root/'continuous/B_probe/training_samples.csv');b=pd.read_csv(root/'restored/B_probe/training_samples.csv')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    for pa,pb in zip(sorted((root/'continuous/B_probe').glob('episode_*/integration_states.npz')),sorted((root/'restored/B_probe').glob('episode_*/integration_states.npz'))):
        with np.load(pa) as x,np.load(pb) as y:np.testing.assert_array_equal(x['states'],y['states'])
    with np.load(root/'continuous/B_probe/rollout_002.npz') as x,np.load(root/'restored/B_probe/rollout_002.npz') as y:
        for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    assert cs['rng']['tasks']==rs['rng']['tasks']
    write_json(RUN/'resume_validation.json',dict(passed=True,physical_steps=30720,training_decisions_test_only=1536,stage_boundary_reset_explicit=True,arbitrary_mid_episode_resume_supported=False,final_parameter_hash=hash_continuous,action_observation_reward_rollout_physics_bitwise_equal=True))

def zero_test():
    task=read(ROOT/'configs/phase2c7_eval.yaml')['development'][0];root=RUN/'validation/zero'
    for version in ('v2_control','v3'):
        env=Phase2C8Env(root/version,[task],seed=42,version=version)
        try:
            obs,_=env.reset(seed=987);assert obs.shape==(58,)
            for _ in range(750):
                obs,r,term,trunc,_=env.step(np.zeros(2));assert not term and np.isfinite(r)
            assert trunc and env.episodes[0]['complete']
        finally:env.close()
        source=OLD/'preflight/Dev_mix/baseline/episode_0000/integration_states.npz'
        with np.load(source) as x,np.load(root/version/'episode_0000/integration_states.npz') as y:np.testing.assert_array_equal(x['states'],y['states'])
    # Two resets of the same physical task must match, independent of task RNG draws.
    env=Phase2C8Env(root/'reset',[task],seed=42,version='v3');obs=[];rewards=[]
    try:
        for n in range(2):
            obs.append(env.reset(seed=111+n)[0]);rewards.append(env.step(np.zeros(2))[1])
        np.testing.assert_array_equal(obs[0],obs[1]);assert rewards[0]==rewards[1]
        try:env.step(np.array([np.nan,0]));raise AssertionError('NaN action accepted')
        except ValueError:pass
    finally:env.close()
    write_json(RUN/'zero_equivalence.json',dict(passed=True,physical_steps=30040,v2_v3_zero_match_historical_baseline_bitwise=True,reset_equal=True,nan_rejected=True))

def collect():
    assert read(RUN/'sampler_validation.json')['passed'] and read(RUN/'resume_validation.json')['passed'] and read(RUN/'zero_equivalence.json')['passed'] and read(RUN/'reward_gate.json')['passed']
    rows=[read(p) for p in (RUN/'preflight').rglob('metrics.json')];assert len(rows)==14
    assert all(r['success'] and r['failure'] is None for r in rows)
    prior=read(OLD/'admission.json')['admitted'];training=sum([cfg('training')['curriculum'][s] for s in ('A','B','C')],[])
    assert all(t in prior for t in training)
    for path,digest in read(RUN/'config_freeze.json').items():assert sha(path)==digest
    assert (RUN/'history_before.json').exists()
    write_json(RUN/'training_gate.json',dict(passed=True,new_joint_preflight=len(rows),historical_exact_joint_tasks_reused=len(training),no_final_policy_used_for_selection=True,zero_equivalent=True,resume_bitwise_equal=True,reward_gate=True,optional_2p4_not_executed=True,control_limits=[.001,20,58,2,.05,5,20,400]))
    print('TRAINING GATE PASSED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sampler','resume','zero','collect']);mode=p.parse_args().mode
    {'sampler':sampler_test,'resume':resume_test,'zero':zero_test,'collect':collect}[mode]()
