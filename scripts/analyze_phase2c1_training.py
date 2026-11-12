"""Recompute rewards, GAE/bootstrap audit and paired final metrics from raw data."""
import _bootstrap,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json,sha
from src.rl.stewart_residual_env import env_config
from src.control.phase2b1_experiment import trial_metrics
from validate_phase2c1_env import audit_folder

def main(root,training_only=False,evaluation_only=False):
    root=Path(root);cfg=env_config();hyper=json.loads((ROOT/'configs/phase2c1_ppo.yaml').read_text(encoding='utf-8'));summaries=[];audits=[];updates=[];episode_rows=[];bootstrap=[];gae=[];action_audit=[]
    if evaluation_only:return evaluate_outputs(root,hyper,pd.read_csv(ROOT/'results/phase2c1_training_summary.csv').to_dict('records'))
    for seed in hyper['training_seeds']:
        folder=root/f'seed{seed}';r=json.loads((folder/'training_result.json').read_text(encoding='utf-8'));data=pd.read_csv(folder/'actual_training/training_samples.csv');assert len(data)==r['env_decisions']==r['actual_decisions']<=4096
        assert int((data.action_physical_steps+data.hold_physical_steps).sum())==r['actual_physical_steps'];events=pd.read_csv(folder/'ppo_update_events.csv');assert len(events)==r['updates'] and (events.actor_change_norm>0).all() and (events.critic_change_norm>0).all();events['seed']=seed;updates.append(events)
        for e in json.loads((folder/'actual_training/episode_summary.json').read_text(encoding='utf-8')):
            if e['physical_steps']==0:continue
            ep=Path(e['folder']);d,costs=audit_folder(ep,cfg);sel=data[data.episode==e['episode']]
            for row in sel.itertuples():
                interval=d.iloc[int(row.physical_start_step):int(row.physical_end_step)];action_audit.append(dict(seed=seed,global_decision=row.global_decision,raw_action=row.raw_action,actor_mean=row.actor_mean,actor_std=row.actor_std,clipped_action=row.clipped_action,target_Fz_N=row.mapped_Fz_N,mean_effective_Fz_N=float(interval.rl_applied_command_wrench_2.mean()),mean_abs_effective_Fz_N=float(abs(interval.rl_applied_command_wrench_2).mean()),peak_leg_RL_effect_N=float(abs(interval[[f'rl_applied_command_{j}' for j in range(6)]]).to_numpy().max()),shared_budget_peak_N=float(abs(interval[[f'F_add_total_{j}' for j in range(6)]]).to_numpy().max()),projected_physical_steps=int((interval.action_rejection_reason=='projected_shared_budget').sum()),native_saturated_steps=int(interval.native_force_saturated.sum())))
            for row in sel.itertuples():
                sl=slice(int(row.physical_start_step),int(row.physical_end_step));recomputed=-sum(sum(np.asarray(v)[sl]) for v in costs.values())*.001/.05+row.failure_penalty;assert abs(recomputed-row.reward)<1e-8;audits.append(dict(seed=seed,episode=e['episode'],global_decision=row.global_decision,logged_reward=row.reward,recomputed_reward=recomputed,error=abs(recomputed-row.reward),actual_physical_steps=row.action_physical_steps+row.hold_physical_steps))
            episode_rows.append(dict(seed=seed,**e))
        for i,path in enumerate(sorted(folder.glob('rollout_*.npz'))):
            with np.load(path) as z:
                adv=z['advantages'].reshape(-1);ret=z['returns'].reshape(-1);values=z['values'].reshape(-1);assert np.isfinite(np.r_[adv,ret,values]).all();np.testing.assert_allclose(adv+values,ret,rtol=1e-5,atol=1e-5)
                physical=z['physical_rewards'];effective=z['effective_rewards'].reshape(-1);chunk=data.iloc[i*256:(i+1)*256];np.testing.assert_allclose(physical,chunk.reward,atol=1e-12);delta=effective-physical;nonterminal=~chunk.truncated.to_numpy(bool);assert abs(delta[nonterminal]).max()<1e-5
                for j,x in enumerate(delta):bootstrap.append(dict(seed=seed,global_decision=int(chunk.iloc[j].global_decision),physical_reward=float(physical[j]),SB3_effective_reward=float(effective[j]),bootstrap_delta=float(x),truncated=bool(chunk.iloc[j].truncated)))
                starts=z['episode_starts'].reshape(-1);last=float(z['last_values'].reshape(-1)[0]);a=np.zeros(len(adv));following=0.
                for j in reversed(range(len(adv))):
                    alive=1.-float(chunk.iloc[-1].terminated or chunk.iloc[-1].truncated) if j==len(adv)-1 else 1.-starts[j+1];nextvalue=last if j==len(adv)-1 else values[j+1];td=effective[j]+hyper['gamma']*nextvalue*alive-values[j];following=td+hyper['gamma']*hyper['gae_lambda']*alive*following;a[j]=following
                np.testing.assert_allclose(a,adv,rtol=2e-5,atol=2e-4);gae.append(dict(seed=seed,update=i+1,max_GAE_recompute_error=float(abs(a-adv).max())))
        summaries.append(dict(seed=seed,decisions=r['actual_decisions'],physical_steps=r['actual_physical_steps'],updates=r['updates'],optimizer_steps=r['optimizer_steps_total'],wall_seconds=r['wall_seconds'],protected_episodes=r['protected_episodes'],mean_training_reward=float(data.reward.mean()),std_training_reward=float(data.reward.std()),mean_abs_sampled_action=float(abs(data.raw_action).mean()),mean_abs_clipped_action=float(abs(data.clipped_action).mean()),clipped_action_fraction=float((abs(data.raw_action)>1).mean()),final_checkpoint=r['final_checkpoint'],sha256=r['final_sha256'],stop_reason=r['stop_reason']))
    pd.DataFrame(summaries).to_csv(ROOT/'results/phase2c1_training_summary.csv',index=False);pd.DataFrame(audits).to_csv(ROOT/'results/phase2c1_training_reward_audit.csv',index=False);pd.DataFrame(episode_rows).to_csv(ROOT/'results/phase2c1_training_episodes.csv',index=False);pd.concat(updates).to_csv(ROOT/'results/phase2c1_update_events.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(ROOT/'results/phase2c1_bootstrap_audit.csv',index=False);pd.DataFrame(gae).to_csv(ROOT/'results/phase2c1_gae_audit.csv',index=False)
    pd.DataFrame(action_audit).to_csv(ROOT/'results/phase2c1_action_audit.csv',index=False)
    write_json(ROOT/'results/phase2c1_reward_audit_summary.json',dict(decisions_recomputed=len(audits),maximum_error=max(x['error'] for x in audits),physical_reward_vs_SB3='Rollout effective rewards may include gamma*terminal Critic prediction for TimeLimit truncations; not physical reward or future truth',actual_updates=sum(x['updates'] for x in summaries)))
    if training_only:return
    evaluate_outputs(root,hyper,summaries)

def evaluate_outputs(root,hyper,summaries):
    evaluation=json.loads((root/'classical_evaluation.json').read_text(encoding='utf-8'))
    for seed in hyper['training_seeds']:evaluation+=json.loads((root/f'seed{seed}/final_evaluation.json').read_text(encoding='utf-8'))
    rows=[]
    for row in evaluation:
        m=trial_metrics(row['folder'],dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3));m.update({k:row.get(k) for k in ['method','task_id','training_seed','deterministic','checkpoint','checkpoint_sha256','source']});d=pd.read_csv(Path(row['folder'])/'physics.csv.gz');xy=np.linalg.norm(d[['p_0','p_1']].to_numpy()-d[['p_ref_0','p_ref_1']].to_numpy(),axis=1)*1000;m['xy_peak_mm']=float(xy.max());extra=d[[f'F_add_total_{j}' for j in range(6)]].to_numpy();m['shared_additional_peak_N']=float(abs(extra).max());m['shared_additional_slew_N_s']=float(abs(np.diff(np.r_[np.zeros((1,6)),extra],axis=0)/.001).max());assert m['shared_additional_peak_N']<=5+1e-8 and m['shared_additional_slew_N_s']<=20+1e-8
        profile=json.loads((Path(row['folder'])/'disturbance_profile.json').read_text());load=(d.input_step>=profile['start_step'])&(d.input_step<profile['stop_step']);post=d.input_step>=profile['stop_step'];zerr=abs(d.p_2-d.p_ref_2)*1000;m['load_z_peak_mm']=float(zerr[load].max());m['unload_z_peak_mm']=float(zerr[post].max());rows.append(m)
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/phase2c1_classical_vs_ppo.csv',index=False);plots(root,f,summaries)
    print('ANALYZED',len(rows),'final evaluation episodes',flush=True)

def plots(root,evaluation,summaries):
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    for s in summaries:
        d=pd.read_csv(root/f'seed{s["seed"]}/actual_training/training_samples.csv');u=pd.read_csv(root/f'seed{s["seed"]}/ppo_update_events.csv');axes[0].plot(d.global_decision,d.reward.rolling(120,min_periods=1).mean(),label=f'seed{s["seed"]}');axes[1].plot(u.global_decisions,u.value_loss,label=f'seed{s["seed"]}');axes[2].plot(u.global_decisions,u.actor_change_norm,label=f'Actor{s["seed"]}');axes[2].plot(u.global_decisions,u.critic_change_norm,ls='--',label=f'Critic{s["seed"]}')
    for ax,title in zip(axes,['Actual physical reward, trailing120 decisions','Value loss','Actual parameter update norms']):ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('PPO decisions')
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c1_learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));post=evaluation[evaluation.method.isin(['C4','C4_fourDlead','PPO_final_deterministic'])]
    for row in post.itertuples():ax.scatter(row.max_force_rate_N_s,row.load_position_iae_mm_s,label=f'{row.task_id}/{row.method}/{row.training_seed}')
    ax.set_xlabel('Actual force peak slew N/s');ax.set_ylabel('Load position IAE mm s');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c1_control_comparison.png',dpi=150);plt.close(fig)
    d=pd.read_csv(root/'seed42/actual_training/episode_0002/physics.csv.gz');t=d.time_s
    for kind in ['actions','position_orientation','six_leg_forces']:
        fig,axes=plt.subplots(2,1,figsize=(12,7))
        if kind=='actions':
            for c in ['rl_actor_raw','rl_normalized_action','rl_actor_mean']:axes[0].plot(t,d[c],label=c)
            axes[1].plot(t,d.rl_mapped_target4_0,label='Fz target N');axes[1].plot(t,d.rl_applied_command_wrench_2,label='Actual command-effect Fz N')
        elif kind=='position_orientation':
            for j in range(3):axes[0].plot(t,(d[f'p_{j}']-d[f'p_ref_{j}'])*1000,label='XYZ'[j]);axes[1].plot(t,d[f'rpy_{j}']-d[f'rpy_ref_{j}'],label=['Roll','Pitch','Yaw'][j])
            axes[0].set_ylabel('Error mm');axes[1].set_ylabel('Error deg')
        else:
            for j in range(6):axes[0].plot(t,d[f'force_{j}'],label=f'Leg{j+1}');axes[1].plot(t,d[f'F_add_total_{j}'],label=f'Leg{j+1}')
            axes[0].set_ylabel('Actual force N');axes[1].set_ylabel('Shared additional force N')
        for ax in axes:ax.set_xlabel('Physical time s');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
        fig.suptitle('Actual Training seed42 episode2; crosses first PPO update');fig.tight_layout();fig.savefig(ROOT/f'assets/plots/phase2c1_{kind}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--training-only',action='store_true');p.add_argument('--evaluation-only',action='store_true');a=p.parse_args();main(a.run_root,a.training_only,a.evaluation_only)
