"""Bounded real dynamics for safety, development and frozen final evaluation."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from phase2c8_protocol import ROOT,RUN,OLD,SEEDS,cfg,read,write_json,sha
from src.rl.phase2c8_env import Phase2C8Env
from src.rl.ppo_trace import parameter_hash
from src.phase1.plant import platform_state
from validate_phase2c7_dynamic_tasks import metrics
from evaluate_phase2c7_ppo import enrich

def rescore(d,alpha=.5,beta=.5):
    x=d.orientation_error_deg.to_numpy()/.006
    omega=d[[f'twist_actual_{i}' for i in (3,4,5)]].to_numpy()-d[[f'twist_ref_{i}' for i in (3,4,5)]].to_numpy()
    pos=(d[[f'p_{i}' for i in range(3)]].to_numpy()-d[[f'p_ref_{i}' for i in range(3)]].to_numpy())*1000
    a=d[['accepted_mx_Nm','accepted_my_Nm']].to_numpy();slew=np.diff(np.vstack([np.zeros(2),a]),axis=0)*1000
    other=np.column_stack([.2*np.sum((omega/.003)**2,axis=1),.1*np.sum((pos/[.2,.2,.05])**2,axis=1),.001*np.sum((a/.05)**2,axis=1),.000025*np.sum((slew/2.5)**2,axis=1)])
    result={}
    for name,orientation in [('v2',x*x),('v3',alpha*(np.hypot(x,.01)-.01)+beta*x*x)]:
        raw=np.column_stack([orientation,other]);factor=np.minimum(1.,100/np.maximum(raw.sum(axis=1),1e-30));result[name]=(raw*factor[:,None],factor)
    return result

def evaluate(task,method,group,seed=0,decisions=0,model_path=None,record=False,version=None):
    root=RUN/group/task['id']/method/f'seed{seed}'/str(decisions)
    if (root/'metrics.json').exists():return read(root/'metrics.json')
    version=version or ('v2_control' if method in ('v2_control','baseline','classical50','random','positive','negative','historical42') else 'v3')
    model=PPO.load(model_path,device='cpu') if model_path else None
    if model:model.policy.set_training_mode(False);before=parameter_hash(model.policy)
    env=Phase2C8Env(root,[task],seed=6401,version=version,record_episodes=[0] if record else [],source=f'FROZEN DETERMINISTIC EVALUATION / phase2c8 / {method}')
    obs,_=env.reset(seed=6401);rng=np.random.default_rng(6401);start=time.perf_counter()
    try:
        for _ in range(round(env.reference.duration*50)):
            action=np.zeros(2)
            if model:action=model.predict(obs,deterministic=True)[0]
            elif method=='random':action=rng.uniform(-1,1,2)
            elif method in ('positive','negative'):action=np.ones(2)*(.5 if method=='positive' else -.5)
            elif method=='classical50':
                _,r,v=platform_state(env.owner.model,env.owner.data);ref=env.reference.at(env.owner.data.time)
                action=(-25*Rotation.from_matrix(r@ref.r.T).as_rotvec()[:2]-(v[3:5]-ref.twist[3:5]))/.05
            env.context.update(raw=action,version=decisions//512)
            obs,_,term,trunc,_=env.step(action)
            if term or trunc:break
    finally:env.close()
    ep=env.episodes[0];result=enrich(metrics(Path(ep['folder']),ep));d=pd.read_csv(Path(ep['folder'])/'physics.csv.gz');scores=rescore(d)
    result.update(method=method,seed=seed,decisions=decisions,group=group,reward_version=version,episode_return=ep['episode_return'],common_v2_return=float(-scores['v2'][0].sum()/20),common_v3_return=float(-scores['v3'][0].sum()/20),model_sha=sha(model_path) if model else None,model_path=str(model_path) if model else None,parameters_unchanged=not model or before==parameter_hash(model.policy),gradient_updates=0,wall_s=time.perf_counter()-start)
    actual=Rotation.from_euler('xyz',d[[f'rpy_{i}' for i in range(3)]],degrees=True);ref=Rotation.from_euler('xyz',d[[f'rpy_ref_{i}' for i in range(3)]],degrees=True);err=(actual*ref.inv()).as_rotvec()*180/np.pi
    result.update(roll_error_peak_deg=float(abs(err[:,0]).max()),pitch_error_peak_deg=float(abs(err[:,1]).max()),native_force_saturation_steps=int(d.native_force_saturated.sum()))
    samples=pd.read_csv(root/'training_samples.csv');penalty=float(samples.terminal_penalty.sum());result['common_v2_return']+=penalty;result['common_v3_return']+=penalty
    assert result['parameters_unchanged'];write_json(root/'metrics.json',result);print(group,task['id'],method,seed,decisions,result['success'],flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','reward_audit','development','final','collect']);p.add_argument('--worker',type=int,default=0);p.add_argument('--method',default='v3');p.add_argument('--seed',type=int,default=42);p.add_argument('--decisions',type=int,default=8192);a=p.parse_args();torch.set_num_threads(1)
    ev=cfg('evaluation')
    if a.mode=='preflight':
        work=[(t,m) for t in ev['development']+ev['final'] for m in ('baseline','random')]
        for i,(t,m) in enumerate(work):
            if i%3==a.worker:evaluate(t,m,'preflight')
    elif a.mode=='reward_audit':
        t=read(ROOT/'configs/phase2c7_eval.yaml')['development'][0]
        methods=['baseline','classical50','historical42','positive','negative']
        for i,m in enumerate(methods):
            if i%3==a.worker:evaluate(t,m,'reward_audit',model_path=ROOT/'models/phase2c7/seed42/final.zip' if m=='historical42' else None)
    elif a.mode in ('development','final'):
        if a.mode=='final':assert read(RUN/'final_selection.json')['frozen']
        model=ROOT/f'models/phase2c8/{a.method}/seed{a.seed}/checkpoint_{a.decisions:06d}.zip'
        if a.method in ('baseline','classical50'):model=None
        for t in ev[a.mode]:evaluate(t,a.method,a.mode,seed=a.seed,decisions=a.decisions,model_path=model,record=a.mode=='final' and t['id'] in ('T8_mix','T8_combined50'))
    else:
        for name in ('development','final'):
            rows=[read(p) for p in (RUN/name).rglob('metrics.json')];pd.DataFrame(rows).to_csv(ROOT/('results/phase2c8_checkpoint_evaluation.csv' if name=='development' else 'results/phase2c8_controller_comparison.csv'),index=False)
if __name__=='__main__':main()
