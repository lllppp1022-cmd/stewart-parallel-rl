"""Independent journal checks and plot export; no training or physics execution."""
import _bootstrap
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_phase2c2 import RUN,PROTOCOL,metrics
from src.model import ROOT,write_json,sha
from src.disturbance.metrics import vec

def main():
    plot=ROOT/'assets/plots';plot.mkdir(exist_ok=True);diagnostics=json.loads((RUN/'action_diagnostics.json').read_text(encoding="utf-8"));parts=[]
    for r in diagnostics:
        folder=Path(r['folder']);d=pd.read_csv(folder/'physics.csv.gz');e=(d.p_2-d.p_ref_2).to_numpy()*1000
        for name,a,b in [('rise',.5,2),('top',2,3),('return',3,4.5),('settle',4.5,6)]:
            mask=(d.time_s>=a)&(d.time_s<b);parts.append(dict(tau_s=r['task']['tau_s'],hz=r['decision_hz'],pattern=r['pattern'],segment=name,z_rmse_mm=float(np.sqrt(np.mean(e[mask]**2))),z_mean_error_mm=float(e[mask].mean()),actual_rl_peak_leg_N=r['actual_rl_peak_leg_N'],success=r['success']))
    pd.DataFrame(parts).to_csv(ROOT/'results/phase2c2_action_response.csv',index=False)
    audits=[];updates=[];totals=[]
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for seed in [42,123]:
        f=RUN/f'seed{seed}';s=pd.read_csv(f/'actual_training/training_samples.csv');u=pd.read_csv(f/'ppo_update_events.csv');u['seed']=seed;updates.append(u)
        for idx,p in enumerate(sorted(f.glob('rollout_*.npz'))):
            with np.load(p) as z:
                b=s.iloc[idx*256:(idx+1)*256];audits.append(dict(seed=seed,update=idx+1,action_error=float(abs(z['actions'].ravel()-b.raw_action).max()),reward_error=float(abs(z['physical_rewards']-b.reward).max()),observation_finite=bool(np.isfinite(z['observations']).all()),value_target_finite=bool(np.isfinite(z['returns']).all())))
        errors=[];rows=0;physmax=0;costcaps=0
        for ep in sorted((f/'actual_training').glob('episode_*')):
            if not ep.is_dir():continue
            d=pd.read_csv(ep/'physics.csv.gz');rows+=len(d);costcaps+=int((d.reward_cap_factor<1).sum());physmax=max(physmax,float(abs(vec(d,'ctrl',12)[:,:6]).max()))
            eps=s[s.episode==int(ep.name.split('_')[-1])]
            for _,a in eps.iterrows():
                v=d.iloc[int(a.physical_start_step):int(a.physical_end_step)];re=-v[[f'reward_cost_{k}' for k in ['z','xy','angle','amplitude','slew','request']]].to_numpy().sum()*.001/.05+a.failure_penalty;errors.append(abs(re-a.reward))
        summary=json.loads((f/'actual_training/interaction_totals.json').read_text(encoding="utf-8"));totals.append(dict(seed=seed,decisions=len(s),physical_steps=rows,declared_physical_steps=summary['physical_steps'],updates=len(u),optimizer_steps=int(u.optimizer_steps_total.iloc[-1]),complete_episodes=sum(x['complete'] for x in summary['episodes']),protected_episodes=sum(bool(x['failure']) for x in summary['episodes']),max_reward_recompute_error=max(errors),reward_cap_fraction=costcaps/rows,max_ctrl_drive_N=physmax))
        er=s.groupby('episode').reward.sum();finished={x['episode'] for x in summary['episodes'] if x['reason'] in ['time_limit','protected_failure']};er=er[er.index.isin(finished)];ax[0,0].plot(er.index,er,label=str(seed));ax[0,1].plot(u.global_decisions,u.value_loss,label=str(seed));ax[1,0].plot(u.global_decisions,u.approx_kl,label=str(seed));ax[1,1].plot(u.global_decisions,u.action_std,label=str(seed))
    for a,title in zip(ax.flat,['Finished episode reward (budget cutoff excluded)','Value loss','Approx KL','Action standard deviation']):a.set_title(title);a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(plot/'phase2c2_learning_curves.png',dpi=160);plt.close(fig)
    pd.DataFrame(totals).to_csv(ROOT/'results/phase2c2_training_comparison.csv',index=False);pd.concat(updates).to_csv(ROOT/'results/phase2c2_update_events.csv',index=False);write_json(RUN/'training_journal_audit.json',dict(totals=totals,rollouts=audits,passed=all(a['action_error']<1e-7 and a['reward_error']<1e-9 for a in audits) and all(t['physical_steps']==t['declared_physical_steps'] and t['max_reward_recompute_error']<1e-8 for t in totals)))
    f=RUN/'seed42/actual_training/episode_0002';d=pd.read_csv(f/'physics.csv.gz');fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True);home=json.loads((f/'home_baseline.json').read_text(encoding="utf-8"))['p_m'][2]
    ax[0].plot(d.time_s,(d.p_2-home)*1000,label='Actual');ax[0].plot(d.time_s,(d.p_ref_2-home)*1000,'--',label='Reference');ax[0].set_ylabel('Z from Home (mm)');ax[1].plot(d.time_s,(d.p_2-d.p_ref_2)*1e6,label='Z tracking error');ax[1].set_ylabel('Error (um)');ax[2].plot(d.time_s,d.rl_normalized_action,label='Sampled/clipped action');ax[2].plot(d.time_s,d.rl_applied_command_wrench_2,label='Bounded Fz command delta before lag (N)')
    for a in ax:a.grid(alpha=.2);a.legend();a.axvline(.8,color='red',alpha=.4,label='PPO update256')
    ax[-1].set_xlabel('Actual physical time (s)');fig.suptitle('Actual stochastic training, episode2 crosses policy update');fig.tight_layout();fig.savefig(plot/'phase2c2_actual_motion.png',dpi=160);plt.close(fig)
    # Historical comparison verifies old source + all old main-run files by hash.
    old=json.loads((RUN/'historical_file_hashes.json').read_text(encoding="utf-8"));prior=json.loads((ROOT/'results/phase2c1_artifact_manifest.json').read_text(encoding="utf-8"))['files'];expected={k.replace('\\','/'):v for k,v in old.items()};expected.update({k.replace('\\','/'):v['sha256'] for k,v in prior.items()});changes=[k for k,h in expected.items() if not (ROOT/k).exists() or sha(ROOT/k)!=h];write_json(RUN/'history_preservation.json',dict(files_checked=len(expected),changed=changes,passed=not changes))
if __name__=='__main__':main()
