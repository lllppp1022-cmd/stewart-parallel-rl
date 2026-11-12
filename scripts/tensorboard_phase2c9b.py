"""Derived full-course scalars, explicitly sourced from raw CSV and frozen trials."""
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from phase2c9b_protocol import ROOT,RUN,write_json


def main():
    output=RUN/'training/derived'
    if output.exists():raise FileExistsError(output)
    samples=pd.read_csv(ROOT/'results/phase2c9b_training_samples.csv')
    episodes=pd.read_csv(ROOT/'results/phase2c9b_episode_returns.csv')
    updates=pd.read_csv(ROOT/'results/phase2c9b_update_events.csv')
    dev=pd.read_csv(ROOT/'results/phase2c9b_checkpoint_evaluation.csv')
    masses={t['id']:t['mass_kg'] for t in __import__('phase2c9b_protocol').cfg('tasks')['training']}
    for (method,seed),group in samples.groupby(['method','seed']):
        with SummaryWriter(str(output/f'{method}_seed{seed}')) as writer:
            for r in group.itertuples():
                for name in ('reward','reward_orientation','reward_omega','reward_xyz','reward_action','reward_slew','actor_mean_roll','actor_mean_pitch','actor_std_roll','actor_std_pitch','accepted_roll','accepted_pitch'):
                    writer.add_scalar('derived_actual/'+name,getattr(r,name),r.global_decision)
                writer.add_scalar('derived_actual/hidden_mass_log_only',masses[r.task_id],r.global_decision)
            e=episodes[(episodes.method==method)&(episodes.seed==seed)]
            for i,r in enumerate(e.itertuples()):
                writer.add_scalar('episodes/complete_return' if r.complete else 'episodes/budget_fragment_return',r.return_raw,r.end_decision)
                writer.add_scalar('episodes/complete_count',int(e.iloc[:i+1].complete.sum()),r.end_decision)
            u=updates[(updates.method==method)&(updates.seed==seed)]
            for r in u.to_dict('records'):
                for name in ('train/value_loss','train/explained_variance','train/approx_kl','train/clip_fraction','actor_delta','critic_delta','advantage_mean','advantage_std','value_target_mean','value_target_std'):
                    writer.add_scalar('optimization/'+name,r[name],r['decisions'])
            ev=dev[(dev.method==method)&(dev.seed==seed)].groupby('decisions').mean(numeric_only=True)
            for n,r in ev.iterrows():
                for name in ('load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','xyz_rmse_mm','common_v2_return'):
                    writer.add_scalar('fixed_development/'+name,r[name],int(n))
    write_json(RUN/'derived_tensorboard_provenance.json',dict(output=str(output),new_training=False,
        sources=['phase2c9b_training_samples.csv','phase2c9b_episode_returns.csv','phase2c9b_update_events.csv','phase2c9b_checkpoint_evaluation.csv']))


if __name__=='__main__':main()
