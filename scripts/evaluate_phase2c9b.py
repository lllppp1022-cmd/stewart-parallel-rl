"""Matched real dynamics trials for admission, tuning and frozen policy evaluation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from phase2c9b_protocol import ROOT,RUN,cfg,read,write_json,sha
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation
from validate_phase2c7_dynamic_tasks import metrics
from evaluate_phase2c7_ppo import enrich
from evaluate_phase2c8 import rescore
from src.rl.ppo_trace import parameter_hash


def trial(task, method, group, seed=0, decisions=0, gains=None, record=False, model_override=None):
    folder=RUN/group/task['id']/method/f'seed{seed}'/str(decisions)
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    model_path=Path(model_override) if model_override else ROOT/f'models/phase2c9b/{method}/seed{seed}/checkpoint_{decisions:06d}.zip'
    model=PPO.load(model_path,device='cpu') if method in ('single_frame','history_5frame') else None
    if model:assert model.num_timesteps==decisions
    policy_before=parameter_hash(model.policy) if model else None
    base=StewartAdaptiveEnv(folder,[task],seed=seed,version='v2_control',record_episodes=[0] if record else [],
                            source=f'PHASE2C9B {group} / FROZEN DYNAMIC EVALUATION / {method}')
    env=HistoryObservation(base,5) if method=='history_5frame' else base
    rng=np.random.default_rng(6401);integral=np.zeros(2)
    observations=[]
    try:
        obs,_=env.reset(seed=6401);observations.append(obs)
        for k in range(round(base.reference.duration*50)):
            action=np.zeros(2)
            if model:action=model.predict(obs,deterministic=True)[0]
            elif method=='random':
                action=rng.uniform(-1,1,2) if k%5==0 else action_previous
            elif method.startswith('adaptive') or method=='classical50':
                g=gains or dict(kp=25.,kd=1.,ki=0.,leak=1.,imax=.025)
                current=obs[-58:];error=current[:2]*.001;velocity=(current[3:5]-current[6:8])*.02
                pd_action=-g['kp']*error-g['kd']*velocity
                proposed=np.clip(np.exp(-g['leak']*.02)*integral-g['ki']*.02*error,-g['imax'],g['imax'])
                integral=np.where(abs(pd_action+proposed)<=.05,proposed,np.exp(-g['leak']*.02)*integral)
                action=np.clip((pd_action+integral)/.05,-1,1)
            action_previous=action.copy();base.context.update(raw=action,version=decisions//512 if model else -1)
            obs,_,terminated,truncated,_=env.step(action);observations.append(obs)
            if terminated or truncated:break
    finally:env.close()
    episode=base.episodes[0];result=metrics(Path(episode['folder']),episode)
    if np.linalg.norm(task['torque']):result=enrich(result)
    d=pd.read_csv(Path(episode['folder'])/'physics.csv.gz')
    rotation=Rotation.from_euler('xyz',d[[f'rpy_{j}' for j in range(3)]],degrees=True)
    reference_rotation=Rotation.from_euler('xyz',d[[f'rpy_ref_{j}' for j in range(3)]],degrees=True)
    error=(rotation*reference_rotation.inv()).as_rotvec()*180/np.pi
    result.update(roll_error_peak_deg=float(abs(error[:,0]).max()),pitch_error_peak_deg=float(abs(error[:,1]).max()))
    result.update(load_window_complete=len(d)>=round(task['on']*1000)+1000,
                  unload_window_complete=len(d)>=round(task['off']*1000)+1000)
    np.save(folder/'observations.npy',np.asarray(observations))
    result.update(task_id=task['id'],mass_kg=task['mass_kg'],method=method,seed=seed,decisions=decisions,
                  episode_return=episode['episode_return'],model_sha=sha(model_path) if model else None,
                  common_v2_return=float(-rescore(d)['v2'][0].sum()/20),gains=gains)
    result.update(evaluation_gradient_updates=0,parameters_unchanged=not model or policy_before==parameter_hash(model.policy))
    assert result['parameters_unchanged']
    write_json(folder/'metrics.json',result)
    print(group,task['id'],method,result['success'],result['failure'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['preflight','collect','tune','development','final'])
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--method',default='single_frame')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--decisions',type=int,default=8192)
    args=parser.parse_args();torch.set_num_threads(1);tasks=cfg('tasks')
    if args.mode=='preflight':
        work=[(t,'baseline') for t in tasks['layers']]+[(t,m) for t in tasks['training']+tasks['development']+tasks['final'] for m in ('baseline','random')]
        for i,(t,m) in enumerate(work):
            if i%6==args.worker:trial(t,m,'preflight')
    elif args.mode=='tune':
        for t in tasks['development']:
            for i,g in enumerate(cfg('evaluation')['adaptive_candidates']):
                if i%3==args.worker:trial(t,f'adaptive_{i}','tuning',gains=g)
    elif args.mode in ('development','final'):
        if args.mode=='final':assert read(RUN/'final_selection.json')['frozen']
        gains=read(RUN/'adaptive_selection.json')['gains'] if args.method=='adaptive' else None
        for t in tasks[args.mode]:
            trial(t,args.method,args.mode,seed=args.seed,decisions=args.decisions,gains=gains,
                  record=args.mode=='final' and t['id'].startswith('Final_m21.5_6'))
    else:
        for group,name in [('preflight','mass_preflight'),('development','checkpoint_evaluation'),('final','generalization')]:
            rows=[read(p) for p in (RUN/group).rglob('metrics.json')]
            if rows:pd.DataFrame(rows).to_csv(ROOT/f'results/phase2c9b_{name}.csv',index=False)


if __name__=='__main__':main()
