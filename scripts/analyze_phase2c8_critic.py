"""Raw physical training samples and stored value targets, without updating models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read,write_json
from evaluate_phase2c8 import rescore

def analyze(method,seed):
    root=RUN/'training'/method/f'seed{seed}';out=RUN/'analysis'/method/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    samples=[];episode_rows=[];rollouts=[];summaries=[];physical_checks=[]
    for stage in ('A','B','C'):
        folder=root/stage
        if not (folder/'summary.json').exists():continue
        summary=read(folder/'summary.json');summaries.append(summary);s=pd.read_csv(folder/'training_samples.csv');s['method']=method;s['seed']=seed;samples.append(s)
        for ep in read(folder/'episodes.json'):
            d=pd.read_csv(Path(ep['folder'])/'physics.csv.gz');score=rescore(d)['v2' if method=='v2_control' else 'v3'][0]
            logged=d[[f'dyn_cost_{k}' for k in ('orientation','omega','xyz','action','slew')]].to_numpy();err=float(abs(score-logged).max())
            part=s[s.episode==ep['episode']];rewards=-pd.DataFrame(score).groupby(d.global_decision.to_numpy()).sum().sum(axis=1).to_numpy()/20+part.terminal_penalty.to_numpy()
            reward_error=float(abs(rewards-part.reward.to_numpy()).max())
            assert err<1e-10 and reward_error<1e-10
            physical_checks.append(dict(method=method,seed=seed,stage=stage,episode=ep['episode'],physical_steps=len(d),component_error=err,reward_error=reward_error,source=ep['folder']))
            episode_rows.append(dict(method=method,seed=seed,stage=stage,episode=ep['episode'],task_id=ep['task']['id'],end_decision=int(part.global_decision.max()),complete=ep['complete'],reason=ep['reason'],seconds=len(d)*.001,return_raw=ep['episode_return'],failure=ep['failure'],load_iae=ep['diagnostics']['load_window_IAE'],unload_iae=ep['diagnostics']['unload_window_IAE'],source=ep['folder']))
        for path in folder.glob('rollout_*.npz'):
            update=int(path.stem.split('_')[-1]);rows=read(folder/'updates.json');u=next(x for x in rows if x['update']==update)
            with np.load(path) as b:
                returns=b['returns'].reshape(-1);values=b['values'].reshape(-1);adv=b['advantages'].reshape(-1)
                raw=b['raw_environment_rewards'].reshape(-1);effective=b['rewards'].reshape(-1)
                finite=np.r_[returns,values,adv,effective];assert np.isfinite(finite).all()
                result=dict(method=method,seed=seed,stage=stage,update=update,decisions=update*512,physical_steps=u['physical_steps'],actor_delta=u['actor_delta'],critic_delta=u['critic_delta'],adam_steps=u['adam_steps'],value_loss=u['train/value_loss'],explained_variance=u['train/explained_variance'],approx_kl=u['train/approx_kl'],clip_fraction=u['train/clip_fraction'],policy_loss=u['train/policy_gradient_loss'],actor_std=u['train/std'],pre_value_mse=float(np.mean((returns-values)**2)),value_bias_prediction_minus_target=float((values-returns).mean()),raw_reward_mean=float(raw.mean()),effective_reward_mean=float(effective.mean()),task_composition=str(u['task_composition']),model_sha=u['sha256'])
                for label,arr in [('target',returns),('value',values),('advantage',adv)]:
                    for key,val in [('mean',arr.mean()),('std',arr.std()),('min',arr.min()),('max',arr.max()),('p05',np.quantile(arr,.05)),('p95',np.quantile(arr,.95))]:result[label+'_'+key]=float(val)
                rollouts.append(result)
    pd.concat(samples).to_csv(out/'samples.csv',index=False);pd.DataFrame(episode_rows).to_csv(out/'episodes.csv',index=False);pd.DataFrame(rollouts).to_csv(out/'critic.csv',index=False);pd.DataFrame(physical_checks).to_csv(out/'reward_checks.csv',index=False)
    write_json(out/'summary.json',dict(method=method,seed=seed,new_decisions=sum(s['new_decisions'] for s in summaries),physical_steps=sum(s['new_physical_steps'] for s in summaries),complete_episodes=sum(s['complete_episodes'] for s in summaries),fragments=sum(s['fragments'] for s in summaries),failures=sum(s['failures'] for s in summaries),updates=len(rollouts),adam_steps=sum(r['adam_steps'] for r in rollouts),wall_s=sum(s['wall_s'] for s in summaries),error=[s['error'] for s in summaries if s['error']],final_parameter_hash=summaries[-1]['final_hash']))
    print('Audited',method,seed,len(rollouts),flush=True)

def collect():
    rows=[read(p) for p in (RUN/'analysis').glob('*/*/summary.json')];pd.DataFrame(rows).to_csv(ROOT/'results/phase2c8_training_summary.csv',index=False)
    for name,target in [('critic','critic_diagnostics'),('episodes','episode_returns'),('samples','training_samples'),('reward_checks','reward_recomputation')]:
        pd.concat([pd.read_csv(p) for p in (RUN/'analysis').glob(f'*/*/{name}.csv')],ignore_index=True).to_csv(ROOT/f'results/phase2c8_{target}.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['v2_control','v3','v3_return_norm']);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    analyze(a.method,a.seed) if a.method else collect()
