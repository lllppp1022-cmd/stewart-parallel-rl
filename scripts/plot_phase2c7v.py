import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase2c7v_common import ROOT,RUN,SEEDS,NAMES
COLORS={42:'#276ac3',123:'#e08c25',2026:'#27995b'}
def boundary(ax):
    for x in (4096,8192):ax.axvline(x,color='gray',ls='--',alpha=.7)
    ax.set_xlabel('Actual PPO Decisions');ax.grid(alpha=.2)
def main():
    ep=pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv');d=pd.read_csv(ROOT/'results/phase2c7v_reward_components.csv');u=pd.read_csv(RUN/'updates.csv')
    fig,axs=plt.subplots(3,2,figsize=(16,13));a=axs[0,0]
    for seed in SEEDS:
        for stage in ('A','B','C'):
            e=ep[(ep.seed==seed)&(ep.stage==stage)&ep.complete]
            a.scatter(e.end_decision,e.episode_return,color=COLORS[seed],s=14,label=str(seed) if stage=='A' else None)
            a.plot(e.end_decision,e.episode_return.rolling(3,min_periods=1).mean(),color=COLORS[seed],alpha=.6)
        e=ep[(ep.seed==seed)&~ep.complete];a.scatter(e.end_decision,e.episode_return,color=COLORS[seed],marker='x',s=40)
    a.set_title('Complete episode return: dots; 3-episode trend; x=incomplete (not comparable)');a.legend()
    for seed in SEEDS:
        g=d[d.seed==seed];axs[0,1].plot(g.global_decision,g.reward.rolling(100,min_periods=1).mean(),label=str(seed),color=COLORS[seed])
    axs[0,1].set_title('Immediate environment reward: 100-decision display mean');axs[0,1].legend()
    for i,seed in enumerate(SEEDS):
        g=d[d.seed==seed];b=g.groupby(((g.global_decision-1)//512)).agg({**{f'cost_{n}':'mean' for n in NAMES},'global_decision':'max'})
        for name in NAMES:axs[1,0].plot(b.global_decision,b['cost_'+name],label=f'{seed}:{name}',alpha=.7)
    axs[1,0].set_title('Five actual weighted costs / decision (512-decision bins)');axs[1,0].set_yscale('symlog',linthresh=.001);axs[1,0].legend(fontsize=6,ncol=3)
    summary=pd.read_csv(ROOT/'results/phase2c7v_training_summary.csv');bottom=np.zeros(len(summary))
    for n in NAMES:
        v=summary[n+'_fraction'].to_numpy();axs[1,1].bar(np.arange(len(v)),v,bottom=bottom,label=n);bottom+=v
    axs[1,1].set_xticks(np.arange(len(summary)),[f'{r.seed}/{r.stage}' for r in summary.itertuples()],rotation=25)
    axs[1,1].set_title('Cumulative cost fractions by seed/stage; cap=0');axs[1,1].legend(fontsize=8)
    for seed in SEEDS:
        g=u[u.seed==seed];axs[2,0].plot(g.decisions,g['train/value_loss'],label=str(seed),color=COLORS[seed]);axs[2,1].plot(g.decisions,g['train/explained_variance'],label=str(seed),color=COLORS[seed])
    axs[2,0].set_title('Value loss');axs[2,0].set_yscale('symlog',linthresh=1);axs[2,1].set_title('Explained variance')
    for ax in axs.flat:
        if ax is not axs[1,1]:boundary(ax)
    fig.suptitle('HISTORICAL TRAINING ANALYSIS | A:0-4096 B:4096-8192 C:8192-32768\nCurriculum distributions differ; return is not fixed-task performance',fontsize=13)
    fig.tight_layout();fig.savefig(ROOT/'results/phase2c7v_reward_dashboard.png',dpi=130);plt.close(fig)
    metrics=['actor_delta','critic_delta','train/policy_gradient_loss','train/value_loss','train/explained_variance','train/approx_kl','train/clip_fraction','train/std','value_target_mean','value_target_std','advantage_mean','advantage_std']
    fig,axs=plt.subplots(4,3,figsize=(17,15))
    for ax,key in zip(axs.flat,metrics):
        for seed in SEEDS:
            g=u[u.seed==seed];ax.plot(g.decisions,g[key],label=str(seed))
        ax.set_title(key);boundary(ax);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(RUN/'optimizer_dashboard.png',dpi=120);plt.close(fig)
    q=pd.read_csv(ROOT/'results/phase2c7v_rollout_diagnostics.csv');fig,axs=plt.subplots(3,2,figsize=(15,12))
    for i,seed in enumerate(SEEDS):
        g=q[q.seed==seed]
        for j,key in enumerate(('advantages','returns')):
            ax=axs[i,j];ax.fill_between(g.decisions,g[key+'_min'],g[key+'_max'],alpha=.15,label='min/max');ax.fill_between(g.decisions,g[key+'_p05'],g[key+'_p95'],alpha=.35,label='5/95 percentiles');ax.plot(g.decisions,g[key+'_median'],label='median');ax.set_title(f'{seed}: {key} (actual stored rollout)');boundary(ax);ax.legend()
    fig.tight_layout();fig.savefig(RUN/'advantage_target_distributions.png',dpi=120);plt.close(fig)
    show=summary[['seed','stage','decisions','complete_episodes','mean_reward','actor_raw_rms','accepted_rms_Nm','budget_projection_fraction','failures']].copy()
    for key in show:
        if show[key].dtype.kind=='f':show[key]=show[key].map(lambda x:f'{x:.5f}')
    fig,ax=plt.subplots(figsize=(18,4));ax.axis('off');table=ax.table(cellText=show.values,colLabels=show.columns,loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6);fig.tight_layout();fig.savefig(RUN/'training_stage_table.png',dpi=140);plt.close(fig)
    path=ROOT/'results/phase2c7v_checkpoint_evaluation.csv'
    if path.exists():
        ev=pd.read_csv(path);keys=['evaluation_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','z_rmse_mm','xy_peak_mm','force_rate_N_s','success']
        fig,axs=plt.subplots(3,3,figsize=(16,12))
        for ax,key in zip(axs.flat,keys):
            for seed in SEEDS:
                g=ev[ev.seed==seed].sort_values('decisions');ax.plot(g.decisions,g[key].astype(float),'.-',label=str(seed),color=COLORS[seed])
            ax.axhline(float(ev[ev.seed==0][key].iloc[0]),color='black',ls=':',label='classical50');ax.set_title(key);boundary(ax);ax.legend(fontsize=7)
        fig.suptitle('Same Dev_mix task, deterministic frozen checkpoints | dots are measured; connecting lines only guide the eye')
        fig.tight_layout();fig.savefig(ROOT/'results/phase2c7v_learning_curves.png',dpi=130);plt.close(fig)
    path=ROOT/'results/phase2c7v_policy_behavior.csv'
    if path.exists():
        b=pd.read_csv(path);fig,axs=plt.subplots(2,2,figsize=(14,8))
        for i,event in enumerate(('load','unload')):
            for seed in SEEDS:
                g=b[(b.task_id=='F_roll')&(b.seed==seed)&(b.event==event)&b.window.isin(['pre100','post20','post40','post60','post100'])]
                for j,key in enumerate(('requested_along_load_Nm','accepted_along_load_Nm')):axs[i,j].plot(g.window,g[key],'.-',label=str(seed));axs[i,j].set_title(f'F_roll {event}: {key}')
        for ax in axs.flat:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(ROOT/'results/phase2c7v_policy_response.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
