"""Frozen evaluation gates, whole trajectories, all seeds, no selection by test."""
import _bootstrap,json,gc
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json,sha
from src.rl.stewart_large_motion_env import RUN

def main():
    summaries=[];actions=[];recompute=[];episode_rows=[]
    for seed in [42,123,2026]:
        folder=RUN/f'training/seed{seed}';s=json.loads((folder/'training_summary.json').read_text(encoding='utf-8'));summaries.append(s);d=pd.read_csv(folder/'training_samples.csv');u=pd.read_csv(folder/'ppo_update_events.csv');assert len(d)==s['decisions'] and int((d.action_physical_steps+d.hold_physical_steps).sum())==s['physical_steps'];assert (u.actor_change_norm>0).all() and (u.critic_change_norm>0).all()
        actions.append(dict(seed=seed,decisions=len(d),raw_action_mean=float(d.raw_action.mean()),raw_action_std=float(d.raw_action.std()),box_clip_fraction=float(d.action_clip_fraction.mean()),budget_projection_fraction=float(d.budget_projection_fraction.mean()),accepted_residual_rms_N=float(np.sqrt(np.mean(d.accepted_residual_rms_N**2))),safety_terminations=s['safety_terminations'],full_sample_reward_mean=float(d.reward.mean()),last_parameter_hash=s['parameter_hash']))
        ep=json.loads((folder/'episode_summary.json').read_text(encoding='utf-8'))
        for row in ep:episode_rows.append(dict(seed=seed,**row))
    pd.DataFrame(summaries).to_csv(ROOT/'results/phase2c4_training_summary.csv',index=False);pd.DataFrame(actions).to_csv(ROOT/'results/phase2c4_action_audit.csv',index=False);pd.DataFrame(episode_rows).to_csv(ROOT/'results/phase2c4_episode_summary.csv',index=False)
    c=pd.read_csv(ROOT/'results/phase2c4_controller_comparison.csv');comparisons=[];checks=[];timing=[];protocol=json.loads((ROOT/'configs/phase2c4_eval.yaml').read_text(encoding='utf-8'));final_ids=[x['id'] for x in protocol['final_tasks']];taskmap={x['id']:x for x in [protocol['training_regression']]+protocol['final_tasks']}
    for r in c.itertuples(index=False):
        d=pd.read_csv(Path(r.folder)/'physics.csv.gz');e=(d.p_2-d.p_ref_2).to_numpy();f=d[[f'force_{j}' for j in range(6)]].to_numpy();a=d[[f'F_command_{j}' for j in range(6)]].to_numpy()-d[[f'F_C0_{j}' for j in range(6)]].to_numpy();rate=np.diff(np.vstack([np.zeros((1,6)),a]),axis=0)/.001;rmse=float(np.sqrt(np.mean(e**2))*1000)
        checks.append(dict(task=r.task_id,method=r.method,metric_error=abs(rmse-r.z_rmse_mm),add_peak_N=float(abs(a).max()),add_rate_N_s=float(abs(rate).max()),force_peak_N=float(abs(f).max()),physical_steps=len(d),safety_terminated=bool(d.failure.notna().any()),warning_count=int(d.warning_count.max())))
        for event,t in [('load',taskmap[r.task_id]['on_s']),('unload',taskmap[r.task_id]['off_s'])]:
            m=(d.time_s>=t)&(d.time_s<t+1);indices=np.flatnonzero(m);i=indices[int(np.argmax(abs(e[m])))];timing.append(dict(task_id=r.task_id,method=r.method,event=event,peak_total_z_error_mm=float(e[i]*1000),peak_after_event_ms=float((d.time_s.iloc[i]-t)*1000),earliest_new_20Hz_feedback_after_event_ms=float((np.floor(t/.05+1e-8)+1)*50-t*1000),note='Total Z error, not isolated disturbance effect; event aligns with decision boundary. Controller1kHz, Actor20Hz.'))
        if r.method.startswith('PPO'):
            b=c[(c.task_id==r.task_id)&(c.method=='FF-VA')].iloc[0];row=dict(task_id=r.task_id,method=r.method,seen_training_task=r.task_id=='train_fixed')
            for k in ['load_IAE_mm_s','unload_IAE_mm_s','z_rmse_mm','position_rmse_mm','xy_peak_mm','so3_rmse_deg','max_force_rate_N_s','force_energy_N2_s']:
                v=float(getattr(r,k));base=float(b[k]);row[k]=v;row['FFVA_'+k]=base;row['absolute_gain_'+k]=base-v;row['relative_gain_'+k]=(base-v)/base
            row['pass_continue']=bool(r.success and row['relative_gain_load_IAE_mm_s']>=.01 and row['relative_gain_unload_IAE_mm_s']>=.01 and r.z_rmse_mm<=b.z_rmse_mm and r.xy_peak_mm<=b.xy_peak_mm*1.05+.005 and r.so3_rmse_deg<=b.so3_rmse_deg*1.05+.0001 and r.max_force_rate_N_s<=b.max_force_rate_N_s*1.1);comparisons.append(row)
        del d;gc.collect()
    pd.DataFrame(timing).to_csv(ROOT/'results/phase2c4_transient_timing.csv',index=False)
    t=pd.DataFrame(comparisons);t.to_csv(ROOT/'results/phase2c4_improvement_vs_FFVA.csv',index=False);seed_pass={seed:bool(t[(t.method==f'PPO{seed}')&t.task_id.isin(final_ids)].pass_continue.all()) for seed in [42,123,2026]};multi=bool(sum(seed_pass.values())>=2 and c.success.all() and not any(s['safety_terminations'] for s in summaries));decision='PILOT SUCCESS — CONTINUE RL RESEARCH' if multi else 'PILOT NEGATIVE — STOP SCALING' if not c.success.all() or any(s['safety_terminations'] for s in summaries) or all(t[(t.method==f'PPO{seed}')&t.task_id.isin(final_ids)].relative_gain_z_rmse_mm.mean()<0 for seed in [42,123,2026]) else 'PILOT INCONCLUSIVE — REVISE TRAINING DESIGN'
    write_json(ROOT/'results/phase2c4_final_decision.json',dict(decision=decision,multi_condition_gate_passed=multi,individual_seed_gate=seed_pass,source='Frozen rule in phase2c4_eval.yaml; all last4096 checkpoints, no best-seed selection',training_decisions=sum(x['decisions'] for x in summaries),training_physical_steps=sum(x['physical_steps'] for x in summaries),untrained_actor='stochastic initial seed42; other final policies deterministic',no_significance_claim=True,larger_budget_authorised=False,multi_executed=False,next='If gate failed, preserve negative/inconclusive result and stop; no additional tuning or budget'))
    passed=all(x['metric_error']<1e-8 and x['add_peak_N']<=5+1e-8 and x['add_rate_N_s']<=20+1e-6 and x['force_peak_N']<=400 and not x['safety_terminated'] and x['warning_count']==0 for x in checks) and not any(x['safety_terminations'] for x in summaries)
    write_json(ROOT/'results/phase2c4_safety_audit.json',dict(passed=passed,evaluation=checks,training_safety_terminations=sum(x['safety_terminations'] for x in summaries),partial_collision_coverage=True,unlimited_passive_hinges=30,training_physical_steps=sum(x['physical_steps'] for x in summaries),no_live_qpos_assignments=True));assert all(x['metric_error']<1e-8 for x in checks)
    fig,ax=plt.subplots(2,3,figsize=(15,8))
    for seed in [42,123,2026]:
        d=pd.read_csv(RUN/f'training/seed{seed}/training_samples.csv');u=pd.read_csv(RUN/f'training/seed{seed}/ppo_update_events.csv');ep=pd.DataFrame([r for r in episode_rows if r['seed']==seed and r['complete']]);ax[0,0].plot((ep.episode+1)*300,ep.return_,label=str(seed));ax[0,1].plot(d.global_decision,d.Z_error_rmse.rolling(300,min_periods=1).mean(),label=str(seed));ax[0,2].plot(d.global_decision,d.XY_error_rmse.rolling(300,min_periods=1).mean(),label=str(seed));ax[1,0].plot(u.global_decisions,u.value_loss,label=str(seed));ax[1,1].plot(u.global_decisions,u.approx_kl,label=str(seed));ax[1,2].plot(d.global_decision,d.accepted_residual_mean_N.rolling(100,min_periods=1).mean(),label=str(seed))
    for a,title in zip(ax.flat,['Complete episode return only','Z error interval RMSE (mm),300decision rolling','XY error interval RMSE (mm),300decision rolling','Value loss','Approx KL','Accepted residual command Fz (N),100decision rolling']):a.set_title(title,fontsize=10);a.set_xlabel('Actual PPO decisions');a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c4_learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(16,5));methods=['C4','fourDlead','FF-VA','PPO42','PPO123','PPO2026'];x=np.arange(len(final_ids))
    for j,m in enumerate(methods):
        q=c[c.method==m].set_index('task_id').loc[final_ids]
        for a,k in zip(ax,['load_IAE_mm_s','unload_IAE_mm_s','z_rmse_mm']):a.bar(x+(j-2.5)*.13,q[k],width=.13,label=m)
    for a,title in zip(ax,['Load1s Z-IAE (mm*s)','Unload1s Z-IAE (mm*s)','Full trajectory Z-RMSE (mm)']):a.set_xticks(x,final_ids);a.set_title(title);a.grid(axis='y',alpha=.2);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c4_independent_evaluation.png',dpi=160);plt.close(fig)
    d=pd.read_csv(RUN/'training/seed42/training_samples.csv');u=pd.read_csv(RUN/'training/seed42/ppo_update_events.csv');fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True);ax[0].plot(d.global_decision,d.raw_action*.1,label='Raw Gaussian request before Box',alpha=.5);ax[0].plot(d.global_decision,d.clipped_action*.1,label='Box request');ax[1].plot(d.global_decision,d.accepted_residual_mean_N,label='Mean accepted command residual')
    for a in ax:
        for x in u.global_decisions:a.axvline(x,color='gray',alpha=.2)
        a.set_ylabel('Task Fz (N)');a.legend();a.grid(alpha=.2)
    ax[1].set_xlabel('Actual global PPO decisions, vertical lines=optimizer events');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c4_action_vs_applied.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True);ax[0].plot(u.global_decisions,u.actor_change_norm,'o-',label='Actor parameter change');ax[0].plot(u.global_decisions,u.critic_change_norm,'o-',label='Critic parameter change');ax[1].plot(u.global_decisions,u.value_loss,label='Value loss');ax[1].set_ylabel('Value loss');a2=ax[1].twinx();a2.plot(u.global_decisions,u.explained_variance,color='orange',label='Explained variance');a2.set_ylabel('Explained variance');ax[0].legend();ax[1].legend();ax[1].set_xlabel('Actual global PPO decisions');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c4_update_timeline.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5));dist=[]
    for k in ['z','xy','rotation','action','slew']:
        v=-d['reward_'+k];ax.plot(d.global_decision,v.rolling(300,min_periods=1).mean(),label=k);dist.append(dict(seed=42,component=k,mean=float(v.mean()),std=float(v.std()),p01=float(v.quantile(.01)),p50=float(v.quantile(.5)),p99=float(v.quantile(.99)),maximum=float(v.max())))
    ax.set_yscale('symlog',linthresh=1e-5);ax.set_xlabel('Actual PPO decisions');ax.set_ylabel('Positive per-decision integrated cost');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c4_reward_components.png',dpi=150);plt.close(fig);pd.DataFrame(dist).to_csv(ROOT/'results/phase2c4_training_reward_distribution.csv',index=False);print(decision,'MULTI GATE',multi,'SEEDS',seed_pass)
if __name__=='__main__':main()
