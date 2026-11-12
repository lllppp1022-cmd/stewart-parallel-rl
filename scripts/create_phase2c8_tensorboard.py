import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from phase2c8_protocol import ROOT,RUN,SEEDS,write_json,sha
def main():
    root=ROOT/'results/phase2c8_tensorboard';assert not root.exists(),'Refuse duplicate event directory'
    files={n:ROOT/f'results/phase2c8_{n}.csv' for n in ('training_samples','episode_returns','critic_diagnostics','checkpoint_evaluation')};d,e,u,v=[pd.read_csv(files[n]) for n in files]
    for method in d.method.unique():
        for seed in SEEDS:
            with SummaryWriter(str(root/method/f'seed{seed}')) as w:
                for _,r in d[(d.method==method)&(d.seed==seed)].iterrows():
                    for key in ('reward','reward_orientation','reward_omega','reward_xyz','reward_action','reward_slew','projection_fraction'):
                        w.add_scalar('training/'+key,r[key],int(r.global_decision))
                for r in e[(e.method==method)&(e.seed==seed)].itertuples():w.add_scalar('training/complete_episode_return' if r.complete else 'training/budget_fragment_return',r.return_raw,int(r.end_decision))
                for _,r in u[(u.method==method)&(u.seed==seed)].iterrows():
                    for key in ('actor_delta','critic_delta','value_loss','explained_variance','approx_kl','clip_fraction','actor_std','target_mean','target_std','advantage_mean','advantage_std','pre_value_mse','value_bias_prediction_minus_target'):w.add_scalar('optimizer/'+key,r[key],int(r.decisions))
                for n,g in v[(v.method==method)&(v.seed==seed)].groupby('decisions'):
                    for key in ('common_v2_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','xyz_rmse_mm','force_rate_N_s'):w.add_scalar('development_equal_task_mean/'+key,g[key].mean(),int(n))
                w.add_text('provenance','Derived from actual Phase8 logs. V2/V3 raw training rewards are different objectives; compare common physical development metrics. No new training here.')
    write_json(root/'provenance.json',{str(p):sha(p) for p in files.values()})
if __name__=='__main__':main()
