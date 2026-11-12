"""Frozen policy evaluation only. No learn/train/optimizer step calls."""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,POINTS,read,write_json,sha,checkpoint,stage
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv
from src.rl.ppo_trace import parameter_hash,optimizer_steps
from src.phase1.plant import platform_state
from validate_phase2c7_dynamic_tasks import metrics

def evaluate(task,seed,n,comparison=False):
    method='classical50' if seed==0 else f'seed{seed}'
    root=RUN/('comparison' if comparison else 'evaluation')/task['id']/method/(str(n) if not comparison else 'final')
    if (root/'result.json').exists():return read(root/'result.json')
    file=checkpoint(seed,n) if seed else None
    model=PPO.load(file,device='cpu') if file else None
    if model:
        model.policy.set_training_mode(False)
        before=parameter_hash(model.policy);adam=optimizer_steps(model.policy);digest=sha(file)
    env=StewartDynamicAttitudeEnv(root,[task],[0] if comparison else [],source='FROZEN DETERMINISTIC CHECKPOINT EVALUATION / PHASE 2C-7V')
    obs,_=env.reset(seed=6401);inputs=[];values=[];start=time.perf_counter()
    try:
        for _ in range(round(env.reference.duration*50)):
            inputs.append(obs.copy())
            if model:
                with torch.no_grad():
                    tensor=torch.as_tensor(obs[None],device='cpu');dist=model.policy.get_distribution(tensor).distribution
                    mean=dist.mean[0].numpy();std=dist.stddev[0].numpy();value=float(model.policy.predict_values(tensor).item())
                action=model.predict(obs,deterministic=True)[0]
                env.context.update(raw=action,mean=mean,std=std,version=n//512)
            else:
                p,r,v=platform_state(env.owner.model,env.owner.data);ref=env.reference.at(env.owner.data.time)
                action=(-25*Rotation.from_matrix(r@ref.r.T).as_rotvec()[:2]-(v[3:5]-ref.twist[3:5]))/.05
                value=np.nan
            values.append(value)
            obs,reward,term,trunc,info=env.step(action)
            if term or trunc:break
    finally:env.close()
    ep=env.episodes[0];result=metrics(__import__('pathlib').Path(ep['folder']),ep)
    unchanged=model is None or (before==parameter_hash(model.policy) and adam==optimizer_steps(model.policy) and digest==sha(file))
    assert unchanged
    result.update(seed=seed,decisions=n,stage=stage(n),task_id=task['id'],method=method,
        evaluation_return=ep['episode_return'],model_sha=sha(file) if file else None,
        model_path=str(file) if file else None,parameters_unchanged=unchanged,gradient_updates=0,
        physical_steps=env.total_physical_steps,wall_s=time.perf_counter()-start)
    np.savez_compressed(root/'decision_observations.npz',observations=inputs,values=values)
    write_json(root/'result.json',result)
    print(task['id'],seed,n,result['success'],result['evaluation_return'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['curve','comparison','collect']);p.add_argument('--worker',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(1);protocol=read(RUN/'protocol.json')
    if a.mode=='curve':
        work=[(seed,n) for seed in SEEDS for n in POINTS]+[(0,0)]
        for i,(seed,n) in enumerate(work):
            if i%3==a.worker:evaluate(protocol['task'],seed,n)
    elif a.mode=='comparison':
        task=next(t for t in read(ROOT/'configs/phase2c7_eval.yaml')['final'] if t['id']==protocol['comparison_video_task'])
        for i,seed in enumerate((0,*SEEDS)):
            if i%3==a.worker:evaluate(task,seed,32768 if seed else 0,True)
    else:
        expected=[RUN/'evaluation'/protocol['task']['id']/f'seed{s}'/str(n)/'result.json' for s in SEEDS for n in POINTS]
        expected.append(RUN/'evaluation'/protocol['task']['id']/'classical50/0/result.json')
        assert all(p.exists() for p in expected)
        pd.DataFrame([read(p) for p in expected]).to_csv(ROOT/'results/phase2c7v_checkpoint_evaluation.csv',index=False)
if __name__=='__main__':main()
