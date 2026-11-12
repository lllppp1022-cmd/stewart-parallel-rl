"""Scientific plots from actual training and frozen development CSVs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from phase2c9b_protocol import ROOT,SEEDS,METHODS


def main():
    samples=pd.read_csv(ROOT/'results/phase2c9b_training_samples.csv')
    episodes=pd.read_csv(ROOT/'results/phase2c9b_episode_returns.csv')
    updates=pd.read_csv(ROOT/'results/phase2c9b_update_events.csv')
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for i,method in enumerate(METHODS):
        for seed in SEEDS:
            d=samples[(samples.method==method)&(samples.seed==seed)]
            e=episodes[(episodes.method==method)&(episodes.seed==seed)&episodes.complete]
            u=updates[(updates.method==method)&(updates.seed==seed)]
            axes[i,0].plot(e.end_decision,e.return_raw,'.-',label=str(seed))
            axes[i,1].plot(u.decisions,u['train/value_loss'],label=str(seed))
            axes[i,2].plot(u.decisions,u['train/explained_variance'],label=str(seed))
        for j,title in enumerate(('Complete episode return: varying tasks','Value loss','Explained variance')):
            axes[i,j].set_title(method+' | '+title);axes[i,j].set_xlabel('Actual PPO decisions');axes[i,j].grid(alpha=.2);axes[i,j].legend()
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c9b_learning_curves.png',dpi=130);plt.close(fig)
    dev=pd.read_csv(ROOT/'results/phase2c9b_checkpoint_evaluation.csv')
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ax,key in zip(axes.flat,('load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','xyz_rmse_mm','common_v2_return')):
        for i,method in enumerate(METHODS):
            for seed in SEEDS:
                d=dev[(dev.method==method)&(dev.seed==seed)].groupby('decisions')[key].mean()
                ax.plot(d.index,d.values,'.-',ls='-' if i==0 else '--',label=f'{method}/{seed}')
        ax.set_title(key);ax.set_xlabel('Actual PPO decisions');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Frozen deterministic DEVELOPMENT tasks; equal task weight')
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c9b_fixed_task_learning.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
