import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT,RUN,SEEDS
COLORS={42:'#2176bc',123:'#e68b23',2026:'#289b58'}
def boundary(ax):
    for n in (4096,8192):ax.axvline(n,color='gray',ls='--',alpha=.4)
    ax.set_xlabel('Actual PPO decisions');ax.grid(alpha=.2)
def main():
    d=pd.read_csv(ROOT/'results/phase2c8_training_samples.csv');ep=pd.read_csv(ROOT/'results/phase2c8_episode_returns.csv');u=pd.read_csv(ROOT/'results/phase2c8_critic_diagnostics.csv');methods=list(d.method.unique())
    out=ROOT/'assets/plots';out.mkdir(exist_ok=True)
    fig,axs=plt.subplots(len(methods),3,figsize=(17,5*len(methods)),squeeze=False)
    for i,method in enumerate(methods):
        for seed in SEEDS:
            g=d[(d.method==method)&(d.seed==seed)];e=ep[(ep.method==method)&(ep.seed==seed)]
            for stage in e.stage.unique():
                a=e[(e.stage==stage)&e.complete];axs[i,0].plot(a.end_decision,a.return_raw,'.-',label=str(seed) if stage=='A' else None,color=COLORS[seed]);axs[i,0].plot(a.end_decision,a.return_raw.rolling(3,min_periods=1).mean(),lw=2,alpha=.5,color=COLORS[seed])
            a=e[~e.complete];axs[i,0].scatter(a.end_decision,a.return_raw,marker='x',color=COLORS[seed]);axs[i,1].plot(g.global_decision,g.reward.rolling(100,min_periods=1).mean(),label=str(seed),color=COLORS[seed])
        for n in ('orientation','omega','xyz','action','slew'):
            g=d[d.method==method];b=g.groupby((g.global_decision-1)//512)['reward_'+n].mean();axs[i,2].plot((b.index+1)*512,-b,label=n)
        for j,title in enumerate(['Complete return; x=budget fragment','Raw reward (100-step display mean)','Actual weighted cost by component (512-step bins)']):axs[i,j].set_title(method+' | '+title);boundary(axs[i,j]);axs[i,j].legend(fontsize=8)
        axs[i,2].set_yscale('symlog',linthresh=.001)
    fig.suptitle('Actual stochastic training; raw V2/V3 rewards have DIFFERENT definitions, do not rank methods by these returns')
    fig.tight_layout();fig.savefig(out/'phase2c8_training_reward.png',dpi=130);plt.close(fig)
    keys=['value_loss','explained_variance','pre_value_mse','value_bias_prediction_minus_target','actor_delta','critic_delta','approx_kl','clip_fraction','actor_std','target_mean','target_std','advantage_std'];fig,axs=plt.subplots(4,3,figsize=(18,15))
    for ax,key in zip(axs.flat,keys):
        for i,method in enumerate(methods):
            for seed in SEEDS:
                g=u[(u.method==method)&(u.seed==seed)];ax.plot(g.decisions,g[key],label=f'{method}/{seed}',color=COLORS[seed],ls='-' if i==0 else '--')
        ax.set_title(key);boundary(ax);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'phase2c8_critic_learning.png',dpi=120);plt.close(fig)
    p=ROOT/'results/phase2c8_checkpoint_evaluation.csv'
    if p.exists():
        ev=pd.read_csv(p);keys=['common_v2_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','z_rmse_mm','xyz_rmse_mm','force_rate_N_s','success'];fig,axs=plt.subplots(3,3,figsize=(17,13))
        for ax,key in zip(axs.flat,keys):
            for i,method in enumerate(methods):
                for seed in SEEDS:
                    g=ev[(ev.method==method)&(ev.seed==seed)].groupby('decisions')[key].mean();ax.plot(g.index,g.values,'.-',label=f'{method}/{seed}',color=COLORS[seed],ls='-' if i==0 else '--')
            classical=ev[ev.method=='classical50'][key]
            if len(classical):ax.axhline(classical.mean(),color='black',ls=':',label='classical50')
            ax.set_title(key);boundary(ax);ax.legend(fontsize=6)
        fig.suptitle('Frozen deterministic DEVELOPMENT tasks, equal task weight | common V2 scoring only for comparable returns')
        fig.tight_layout();fig.savefig(out/'phase2c8_fixed_task_learning.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
