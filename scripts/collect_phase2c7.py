"""Gates and evidence tables, computed only from completed real runs."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase2c7_protocol import ROOT,RUN,SEEDS,config,read,write_json
from src.model import sha


def pilot_gate():
    summaries=[read(RUN/'training'/f'seed{s}'/'pilot/summary.json') for s in SEEDS]
    evaluations=[read(RUN/'pilot_evaluation/Dev_mix'/f'PPO_{s}'/'comparison.json') for s in SEEDS]
    passed=all(s['decisions']==8192 and s['physical_steps']==163840 and s['updates']==16 and
               s['failures']==0 and s['error'] is None and s['stop'] is None and s['complete_episodes']>=8 for s in summaries)
    passed=passed and all(e['success'] and e['external_error']<1e-9 for e in evaluations)
    for seed in SEEDS:
        updates=read(RUN/'training'/f'seed{seed}'/'pilot/updates.json')
        passed=passed and all(x['actor_delta']>0 and x['critic_delta']>0 for x in updates)
    write_json(RUN/'pilot_gate.json',dict(passed=passed,summaries=summaries,evaluations=evaluations,
        rule='All three bounded pilots update both networks, contain complete A/B episodes, have no physical failures, and pass matched complete development episodes.'))
    print('PILOT GATE',passed)


def freeze():
    assert not (RUN/'final_selection.json').exists()
    models={f'PPO_{s}':sha(ROOT/'models/phase2c7'/f'seed{s}/final.zip') for s in SEEDS}
    write_json(RUN/'final_selection.json',dict(models=models,eval_sha=sha(ROOT/'configs/phase2c7_eval.yaml'),
        selection='All three cumulative-32768 terminal checkpoints, no best-seed selection'))


def historical():
    errors=[];matched=0
    for name,expected in read(RUN/'history_before.json').items():
        path=ROOT/name
        if not path.exists():errors.append(dict(path=name,error='missing'));continue
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:errors.append(dict(path=name,error='changed',expected=expected,actual=actual))
        else:matched+=1
    write_json(RUN/'history_after.json',dict(passed=not errors,matched=matched,errors=errors))
    print('HISTORICAL',matched,'unchanged',len(errors),'errors')


def collect():
    expected=[RUN/'final'/t['id']/method/'comparison.json' for t in config('eval')['final']
              for method in ('baseline','classical50','PPO_42','PPO_123','PPO_2026','OldCurriculum42')]
    assert all(p.is_file() for p in expected), 'Final evaluation is incomplete; do not publish a partial aggregate'
    summary=[];stage_rows=[];reward_rows=[];update_frames=[];fig,axes=plt.subplots(2,2,figsize=(13,8))
    for seed in SEEDS:
        for phase in ('pilot','continue'):
            directory=RUN/'training'/f'seed{seed}'/phase
            s=read(directory/'summary.json');summary.append(s)
            samples=pd.read_csv(directory/'training_samples.csv')
            for ep in read(directory/'episodes.json'):
                reward_rows.append(dict(seed=seed,phase=phase,episode=ep['episode'],task=ep['task']['id'],
                    physical_steps=ep['physical_steps'],complete=ep['complete'],failure=ep['failure'],
                    cap_fraction=ep['diagnostics']['cap_steps']/max(1,ep['physical_steps']),
                    observation_clip_fraction=ep['clip_fraction'],episode_return=ep['episode_return'],
                    peak_so3_error_deg=ep['diagnostics']['peak_so3_error']))
            for stage,g in samples.groupby('stage'):
                stage_rows.append(dict(seed=seed,phase=phase,stage=stage,decisions=len(g),physical_steps=int(g.physical_steps.sum()),
                   reward_mean=g.reward.mean(),reward_min=g.reward.min(),accepted_rms_Nm=float(np.sqrt(np.mean(g[['accepted_roll','accepted_pitch']].to_numpy()**2))),
                   task_ids=','.join(sorted(g.task_id.unique())),failures=int(g.terminated.sum())))
            axes[0,0].plot(samples.global_decision,samples.reward.rolling(100,min_periods=1).mean(),label=f'{seed} {phase}')
        updates=pd.DataFrame(read(RUN/'training'/f'seed{seed}'/'continue/updates.json'))
        updates['seed']=seed;update_frames.append(updates)
        for ax,field in [(axes[0,1],'train/value_loss'),(axes[1,0],'train/explained_variance'),(axes[1,1],'value_target_std')]:
            ax.plot(updates.decisions,updates[field],label=str(seed))
    pd.DataFrame(summary).to_csv(ROOT/'results/phase2c7_training_summary.csv',index=False)
    pd.DataFrame(stage_rows).to_csv(RUN/'stage_composition.csv',index=False)
    pd.DataFrame(reward_rows).to_csv(ROOT/'results/phase2c7_reward_diagnostics.csv',index=False)
    pd.concat(update_frames,ignore_index=True).to_csv(ROOT/'results/phase2c7_ppo_update_events.csv',index=False)
    for ax,title in zip(axes.flat,['Reward (100-decision mean)','Value loss','Explained variance','Value target standard deviation']):
        ax.set_title(title);ax.set_xlabel('PPO decisions');ax.legend();ax.grid(alpha=.25)
    axes[0,1].set_yscale('symlog',linthresh=1)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c7_learning_curves.png',dpi=140);plt.close(fig)
    rows=[read(p) for p in expected]
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/phase2c7_controller_comparison.csv',index=False)
    attitude=[c for c in df if any(key in c for key in ('iae','so3','omega','overshoot','recovery','yaw'))]
    df[['task_id','method','success',*attitude]].to_csv(ROOT/'results/phase2c7_attitude_metrics.csv',index=False)
    df[['task_id','method','requested_rms_Nm','accepted_rms_Nm','projection_fraction','add_peak_N','add_rate_N_s','force_peak_N','force_rate_N_s','force_squared_integral_N2_s']].to_csv(ROOT/'results/phase2c7_action_audit.csv',index=False)
    fig,axes=plt.subplots(4,2,figsize=(15,14))
    for i,task in enumerate(config('eval')['final']):
        case=df[df.task_id==task['id']].sort_values('method')
        for j,metric in enumerate(('load_iae_deg_s','unload_iae_deg_s')):
            ax=axes[i,j];ax.bar(case.method,case[metric]);ax.set_title(task['id']+' '+metric)
            ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c7_independent_iae.png',dpi=140);plt.close(fig)
    from render_phase2c7_videos import series
    fig,axes=plt.subplots(4,2,figsize=(15,13))
    for method in ('baseline','classical50','PPO_42'):
        d=pd.read_csv(RUN/'final/F_mix'/method/'episode_0000/physics.csv.gz').iloc[::20]
        for ax,(name,values) in zip(axes.flat,series(d)):
            for k in range(values.shape[1]):ax.plot(d.time_s,values[:,k],label=f'{method}:{k+1}',linewidth=.8)
            ax.set_title(name);ax.set_xlabel('physical seconds');ax.grid(alpha=.2)
    for ax in axes.flat:ax.legend(fontsize=5,ncol=3)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c7_control_curves.png',dpi=140);plt.close(fig)
    effects=[];qualifying=[]
    for seed in SEEDS:
        wins=0;costs=True;complete=True
        for task in config('eval')['final']:
            case=df[df.task_id==task['id']].set_index('method');p=case.loc[f'PPO_{seed}'];b=case.loc['classical50']
            effect={metric:100*(b[metric]-p[metric])/max(abs(b[metric]),1e-12) for metric in
                    ('load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','z_rmse_mm','xy_peak_mm','force_rate_N_s')}
            both=effect['load_iae_deg_s']>0 and effect['unload_iae_deg_s']>0
            wins+=int(both);complete=complete and bool(p.success)
            costs=costs and all(p[m]<=1.1*b[m] for m in ('z_rmse_mm','xy_peak_mm','force_rate_N_s'))
            effects.append(dict(seed=seed,task_id=task['id'],both_iae_improved=both,**effect))
        qualifying.append(dict(seed=seed,improved_tasks=wins,complete=complete,cost_guard_passed=costs,qualifies=wins>=2 and complete and costs))
    good=sum(q['qualifies'] for q in qualifying)
    any_gain=any(e['both_iae_improved'] for e in effects)
    decision='SUCCESS' if good>=2 else 'PARTIAL SUCCESS' if any_gain and df.success.all() else 'NEGATIVE'
    pd.DataFrame(effects).to_csv(RUN/'relative_effects_percent.csv',index=False)
    write_json(ROOT/'results/phase2c7_final_decision.json',dict(status='PHASE 2C-7 '+decision,qualifying=qualifying,
        new_training_decisions=sum(s['new_decisions'] for s in summary),new_training_physical_steps=sum(s['physical_steps'] for s in summary if s['phase']=='continue'),
        all_final_complete=bool(df.success.all()),evaluation_count=len(df),evidence=str(RUN),criterion=config('eval')['criterion']))
    write_json(ROOT/'results/phase2c7_safety_audit.json',dict(preflight_rows=len(pd.read_csv(ROOT/'results/phase2c7_joint_preflight.csv')),
        training_failure_episodes=sum(s['failures'] for s in summary),final_failure_episodes=int((~df.success).sum()),
        final_add_peak_N=float(df.add_peak_N.max()),final_add_rate_N_s=float(df.add_rate_N_s.max()),
        final_native_force_peak_N=float(df.force_peak_N.max()),geometry_coverage='Existing partial audited proxy; not continuous/full collision certification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pilot-gate','freeze-final','collect','historical']);a=p.parse_args()
    {'pilot-gate':pilot_gate,'freeze-final':freeze,'collect':collect,'historical':historical}[a.action]()
