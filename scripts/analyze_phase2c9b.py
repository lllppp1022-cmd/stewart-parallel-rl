"""Read-only physical reward/GAE audit, matched gates and complete seed summaries."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,write_json,sha
from evaluate_phase2c8 import rescore


def training():
    summaries=[];updates=[];episodes=[];rewards=[];samples=[];gae=[]
    for method in METHODS:
        for seed in SEEDS:
            folders=sorted((RUN/'training'/method/f'seed{seed}').glob('from_*'))
            other='history_5frame' if method=='single_frame' else 'single_frame'
            for folder in folders:
                summary=read(folder/'summary.json');summaries.append(summary)
                partner=RUN/'training'/other/f'seed{seed}'/folder.name/'task_sequence.json'
                assert read(folder/'task_sequence.json')==read(partner)
                sample=pd.read_csv(folder/'training_samples.csv');sample['method']=method;sample['seed']=seed;samples.append(sample)
                for episode in read(folder/'episodes.json'):
                    physical=pd.read_csv(Path(episode['folder'])/'physics.csv.gz')
                    components=rescore(physical)['v2'][0]
                    recorded=physical[[f'dyn_cost_{n}' for n in ('orientation','omega','xyz','action','slew')]].to_numpy()
                    error=float(abs(components-recorded).max());assert error<1e-10
                    selected=sample[sample.episode==episode['episode']]
                    recomputed=-pd.DataFrame(components).groupby(physical.global_decision.to_numpy()).sum().sum(axis=1).to_numpy()/20+selected.terminal_penalty.to_numpy()
                    reward_error=float(abs(recomputed-selected.reward.to_numpy()).max());assert reward_error<1e-10
                    rewards.append(dict(method=method,seed=seed,source=episode['folder'],physical_rows=len(physical),component_error=error,reward_error=reward_error))
                    episodes.append(dict(method=method,seed=seed,complete=episode['complete'],failure=episode['failure'],return_raw=episode['episode_return'],task=episode['task'],end_decision=int(selected.global_decision.max()),source=episode['folder']))
                for u in read(folder/'updates.json'):
                    if u['decisions']<=summary['start_decisions']:continue
                    values=dict(method=method,seed=seed,**u);updates.append(values)
                    path=folder/f"rollout_{u['update']:03d}.npz"
                    with np.load(path) as b:
                        r,v,start,adv,target=[b[k].reshape(-1).astype(float) for k in ('rewards','values','episode_starts','advantages','returns')]
                        last=float(b['last_values'].ravel()[0])
                    values['prediction_minus_target_bias']=float((v-target).mean())
                    for label,array in [('target',target),('value',v),('advantage',adv)]:
                        values[label+'_p05']=float(np.quantile(array,.05))
                        values[label+'_p50']=float(np.quantile(array,.50))
                        values[label+'_p95']=float(np.quantile(array,.95))
                    part=sample[sample.global_decision.between(u['decisions']-511,u['decisions'])]
                    values['actual_task_composition']=str(part.task_id.value_counts().to_dict())
                    # Preserve SB3's float32 vector arithmetic, including every
                    # recursive rounding; float64 scalar recurrence drifts at
                    # large learned value magnitudes despite identical inputs.
                    r,v,start=[x.astype(np.float32).reshape(-1,1) for x in (r,v,start)]
                    last=np.array([last],dtype=np.float32)
                    a=np.empty((512,1),dtype=np.float32);previous=0;ppo=cfg('ppo')
                    for i in reversed(range(512)):
                        mask=1-np.array([part.iloc[-1].terminated or part.iloc[-1].truncated],dtype=np.float32) if i==511 else 1-start[i+1]
                        delta=r[i]+ppo['gamma']*(last if i==511 else v[i+1])*mask-v[i]
                        previous=delta+ppo['gamma']*ppo['gae_lambda']*mask*previous;a[i]=previous
                    a=a.ravel();v=v.ravel()
                    np.testing.assert_allclose(a,adv,rtol=3e-6,atol=5e-5)
                    np.testing.assert_allclose(a+v,target,rtol=3e-6,atol=5e-5)
                    gae.append(dict(method=method,seed=seed,update=u['update'],max_error=float(abs(a-adv).max())))
    for name,rows in [('training_summary',summaries),('update_events',updates),('episode_returns',episodes),('reward_recomputation',rewards),('gae_recomputation',gae)]:
        pd.DataFrame(rows).to_csv(ROOT/f'results/phase2c9b_{name}.csv',index=False)
    merged=pd.concat(samples)
    merged.to_csv(ROOT/'results/phase2c9b_training_samples.csv',index=False)
    mass_by_task={t['id']:t['mass_kg'] for t in cfg('tasks')['training']}
    merged['hidden_mass_kg_log_only']=merged.task_id.map(mass_by_task)
    merged.groupby(['method','seed','hidden_mass_kg_log_only']).agg(actual_decisions=('reward','size'),mean_raw_reward=('reward','mean')).reset_index().to_csv(ROOT/'results/phase2c9b_mass_exposure.csv',index=False)
    write_json(RUN/'training_audit.json',dict(passed=True,formal_decisions=sum(s['new_decisions'] for s in summaries),
        formal_physics=sum(s['physical_steps'] for s in summaries),updates=len(updates),matched_task_sequences=True,
        physical_reward_recomputed=True,all_gae_recomputed=True))


def tune():
    rows=[read(p) for p in (RUN/'tuning').rglob('metrics.json')]
    assert len(rows)==6
    data=pd.DataFrame(rows);choices=[]
    for method,g in data.groupby('method'):
        baseline=data[data.method=='adaptive_0'].set_index('task_id')
        paired=g.set_index('task_id')
        if g.success.all() and (paired.xyz_rmse_mm<=baseline.xyz_rmse_mm*1.1).all():
            choices.append((float((g.load_iae_deg_s+g.unload_iae_deg_s).mean()),method,g.iloc[0].gains))
    assert choices
    score,method,gains=min(choices)
    write_json(RUN/'adaptive_selection.json',dict(method=method,gains=gains,development_objective=score,final_test_unseen=True))


def pilot():
    d=pd.DataFrame([read(p) for p in (RUN/'development').rglob('metrics.json')]);summary=pd.read_csv(ROOT/'results/phase2c9b_training_summary.csv')
    safe=bool(summary.failures.sum()==0 and summary.new_decisions.eq(8192).all())
    details=[];families={}
    for method in METHODS:
        evidence=False
        for seed in SEEDS:
            a=d[(d.method==method)&(d.seed==seed)&(d.decisions==0)].set_index('task_id')
            b=d[(d.method==method)&(d.seed==seed)&(d.decisions==8192)].set_index('task_id')
            assert len(a)==len(b)==len(cfg('tasks')['development'])
            ratios={k:float(b[k].mean()/a[k].mean()) for k in ('load_iae_deg_s','unload_iae_deg_s','xyz_rmse_mm')}
            good=bool(b.success.all() and all(x<=1.5 for x in ratios.values()))
            learned=bool(any(ratios[k]<.999 for k in ('load_iae_deg_s','unload_iae_deg_s')) or b.common_v2_return.mean()>a.common_v2_return.mean())
            evidence|=learned;safe&=good
            details.append(dict(method=method,seed=seed,guard=good,learning_evidence=learned,ratios=ratios))
        families[method]=evidence
    passed=safe and all(families.values())
    write_json(RUN/'pilot_gate.json',dict(passed=bool(passed),details=details,family_learning=families,
        rule=cfg('evaluation')['pilot_gate'],no_final_policy_used=True))
    print('PILOT GATE',passed,details,flush=True)


def selection():
    n=32768 if read(RUN/'pilot_gate.json')['passed'] else 8192
    models={f'{m}/seed{s}':dict(path=str(ROOT/f'models/phase2c9b/{m}/seed{s}/checkpoint_{n:06d}.zip'),
        sha=sha(ROOT/f'models/phase2c9b/{m}/seed{s}/checkpoint_{n:06d}.zip')) for m in METHODS for s in SEEDS}
    assert not (RUN/'final_selection.json').exists()
    write_json(RUN/'final_selection.json',dict(frozen=True,common_budget=n,models=models,no_best_seed_selection=True,
        configs=read(RUN/'config_freeze.json'),adaptive=read(RUN/'adaptive_selection.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['training','tune','pilot','selection']);globals()[p.parse_args().mode]()
