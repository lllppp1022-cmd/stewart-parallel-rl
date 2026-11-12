"""Recompute every original environment reward from archived 1 ms telemetry."""
import argparse
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from phase2c7v_common import ROOT,OLD,RUN,SEEDS,NAMES,read,write_json

def analyze(seed):
    out=RUN/f'audit_seed{seed}';out.mkdir(exist_ok=False)
    all_samples=[];episodes=[];audit=[];updates=[];bootstrap=[]
    columns=['global_decision','decision','orientation_error_deg','reward_cap_factor',
       *[f'{p}_{i}' for p in ('p','p_ref') for i in range(3)],
       *[f'{p}_{i}' for p in ('twist_actual','twist_ref') for i in (3,4,5)],
       'accepted_mx_Nm','accepted_my_Nm',*[f'dyn_cost_{n}' for n in NAMES]]
    for phase in ('pilot','continue'):
        folder=OLD/f'training/seed{seed}'/phase
        samples=pd.read_csv(folder/'training_samples.csv');samples['seed']=seed;samples['phase']=phase
        for ep in read(folder/'episodes.json'):
            g=samples[samples.episode==ep['episode']].copy()
            d=pd.read_csv(__import__('pathlib').Path(ep['folder'])/'physics.csv.gz',usecols=columns)
            assert len(d)==int(g.physical_steps.sum())==ep['physical_steps']
            assert d.groupby('global_decision').size().eq(20).all()
            accepted=d[['accepted_mx_Nm','accepted_my_Nm']].to_numpy()
            slew=np.diff(np.vstack([np.zeros((1,2)),accepted]),axis=0)/.001
            xyz=(d[[f'p_{i}' for i in range(3)]].to_numpy()-d[[f'p_ref_{i}' for i in range(3)]].to_numpy())*1000
            w=d[[f'twist_actual_{i}' for i in (3,4,5)]].to_numpy()-d[[f'twist_ref_{i}' for i in (3,4,5)]].to_numpy()
            costs=np.column_stack([(d.orientation_error_deg.to_numpy()/.006)**2,.2*np.sum((w/.003)**2,1),
              .1*np.sum((xyz/[.2,.2,.05])**2,1),.001*np.sum((accepted/.05)**2,1),.000025*np.sum((slew/2.5)**2,1)])
            factor=np.minimum(1.,100/np.maximum(costs.sum(1),1e-30));costs*=factor[:,None]
            component_error=float(abs(costs-d[[f'dyn_cost_{n}' for n in NAMES]].to_numpy()).max())
            sums=pd.DataFrame(costs,columns=NAMES).groupby(d.global_decision.to_numpy()).sum()/20
            sums=sums.loc[g.global_decision]
            recomputed=-sums.sum(1).to_numpy()+g.terminal_penalty.to_numpy()
            error=abs(recomputed-g.reward.to_numpy())
            for i,name in enumerate(NAMES):
                g['cost_'+name]=sums[name].to_numpy()
                g['fraction_'+name]=sums[name].to_numpy()/np.maximum(sums.sum(1).to_numpy(),1e-30)
            g['recomputed_reward']=recomputed;g['reward_error']=error
            g['cap_fraction']=pd.Series(factor<1).groupby(d.global_decision.to_numpy()).mean().loc[g.global_decision].to_numpy()
            all_samples.append(g)
            ep_return=float(g.reward.sum());assert abs(ep_return-ep['episode_return'])<1e-7
            episodes.append(dict(seed=seed,phase=phase,episode=ep['episode'],stage=g.stage.iloc[0],task_id=ep['task']['id'],
                complete=ep['complete'],reason=ep['reason'],failure=ep['failure'],start_decision=int(g.global_decision.min()),
                end_decision=int(g.global_decision.max()),decisions=len(g),physical_steps=len(d),physical_seconds=len(d)*.001,
                reference_seconds=ep['duration_s'],episode_return=ep_return,mean_reward=ep_return/len(g),source=ep['folder']))
            audit.append(dict(seed=seed,phase=phase,episode=ep['episode'],decisions=len(g),physical_steps=len(d),
                max_abs_difference=float(error.max()),mean_abs_difference=float(error.mean()),anomalies=int((error>1e-8).sum()),
                physical_component_error=component_error,component_reward_error=float(abs(-sums.to_numpy()-g[[f'reward_{n}' for n in NAMES]].to_numpy()).max()),
                cap_steps=int((factor<1).sum()),source=str(__import__('pathlib').Path(ep['folder'])/'physics.csv.gz')))
            print(seed,phase,ep['episode'],'reward checked',float(error.max()),flush=True)
        indexed=samples.set_index('global_decision')
        for file in sorted(folder.glob('rollout_*.npz')):
            update=int(file.stem.split('_')[-1]);g=indexed.loc[np.arange((update-1)*512+1,update*512+1)]
            with np.load(file) as b:
                raw=b['rewards'].reshape(-1);delta=raw-g.reward.to_numpy()
                for i in np.flatnonzero(g.truncated.to_numpy()):
                    bootstrap.append(dict(seed=seed,update=update,decision=int(g.index[i]),environment_reward=float(g.reward.iloc[i]),
                        rollout_reward=float(raw[i]),bootstrap_adjustment=float(delta[i]),source=str(file)))
                untouched=~g.truncated.to_numpy();expected=g.reward.to_numpy(dtype=np.float32)
                np.testing.assert_array_equal(raw[untouched],expected[untouched])
                np.testing.assert_array_equal(b['actions'].reshape(-1,2),g[['raw_roll','raw_pitch']].to_numpy(dtype=np.float32))
                stats={}
                for key in ('advantages','returns','values'):
                    vals=b[key].reshape(-1)
                    for suffix,val in zip(('min','p05','median','p95','max'),np.quantile(vals,[0,.05,.5,.95,1])):stats[key+'_'+suffix]=float(val)
                updates.append(dict(seed=seed,update=update,decisions=update*512,**stats,
                    task_composition=';'.join(f'{key}:{len(x)}' for key,x in g.groupby('task_id'))))
    pd.concat(all_samples).to_csv(out/'samples.csv',index=False)
    for name,rows in [('episodes',episodes),('audit',audit),('rollouts',updates),('bootstrap',bootstrap)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)

def collect():
    for name,target in [('samples','reward_components'),('episodes','episode_returns'),('audit','reward_recomputation'),('rollouts','rollout_diagnostics'),('bootstrap','bootstrap_audit')]:
        d=pd.concat([pd.read_csv(RUN/f'audit_seed{s}'/(name+'.csv')) for s in SEEDS],ignore_index=True)
        d.to_csv(ROOT/f'results/phase2c7v_{target}.csv',index=False)
    d=pd.read_csv(ROOT/'results/phase2c7v_reward_components.csv');ep=pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv')
    assert len(d)==98304 and d.physical_steps.sum()==1966080 and ep.complete.sum()==126 and len(ep)==135
    summary=[];lengths=[]
    for (seed,stage),g in d.groupby(['seed','stage']):
        e=ep[(ep.seed==seed)&(ep.stage==stage)&ep.complete]
        row=dict(seed=seed,stage=stage,decisions=len(g),physical_steps=int(g.physical_steps.sum()),complete_episodes=len(e),
             mean_reward=g.reward.mean(),return_mean_same_duration=e.episode_return.mean() if e.physical_seconds.nunique()==1 else None,
             return_std_same_duration=e.episode_return.std() if e.physical_seconds.nunique()==1 else None,
             duration_note='same duration' if e.physical_seconds.nunique()==1 else 'mixed durations: see length-stratified table',
             actor_raw_rms=float(np.sqrt(np.mean(g[['raw_roll','raw_pitch']].to_numpy()**2))),
             accepted_rms_Nm=float(np.sqrt(np.mean(g[['accepted_roll','accepted_pitch']].to_numpy()**2))),
             box_clip_fraction=g.action_box_clipped.mean(),budget_projection_fraction=g.projection_fraction.mean(),
             failures=int(g.terminated.sum()),cap_fraction=g.cap_fraction.mean())
        total=g[[f'cost_{n}' for n in NAMES]].sum().sum()
        row.update({n+'_fraction':float(g['cost_'+n].sum()/total) for n in NAMES});summary.append(row)
        for duration,h in e.groupby('physical_seconds'):lengths.append(dict(seed=seed,stage=stage,seconds=duration,n=len(h),return_mean=h.episode_return.mean(),return_std=h.episode_return.std()))
    pd.DataFrame(summary).to_csv(ROOT/'results/phase2c7v_training_summary.csv',index=False)
    pd.DataFrame(lengths).to_csv(RUN/'duration_stratified_returns.csv',index=False)
    u=pd.concat([pd.DataFrame(read(OLD/f'training/seed{s}/continue/updates.json')).assign(seed=s) for s in SEEDS])
    u.to_csv(RUN/'updates.csv',index=False)
    tags=[]
    for folder in sorted((OLD/'training').glob('*/*/tensorboard/PPO*')):
        a=EventAccumulator(str(folder));a.Reload()
        tags.append(dict(source=str(folder),tags=a.Tags(),scalar_steps={k:[int(e.step) for e in a.Scalars(k)] for k in a.Tags()['scalars']}))
    write_json(RUN/'original_tensorboard_tags.json',tags)
    audit=pd.read_csv(ROOT/'results/phase2c7v_reward_recomputation.csv')
    write_json(RUN/'reward_audit_summary.json',dict(decisions=len(d),physical_steps=int(d.physical_steps.sum()),episodes=len(ep),complete=int(ep.complete.sum()),
       max_abs_difference=float(audit.max_abs_difference.max()),mean_abs_difference=float(np.average(audit.mean_abs_difference,weights=audit.decisions)),
       anomalies=int(audit.anomalies.sum()),physical_component_error=float(audit.physical_component_error.max()),cap_steps=int(audit.cap_steps.sum())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    analyze(a.seed) if a.seed else collect()
