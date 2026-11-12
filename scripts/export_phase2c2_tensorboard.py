"""Export all completed, actually recorded updates including the last SB3 update."""
import _bootstrap
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from research_phase2c2 import RUN

def main():
    for seed in [42,123]:
        f=RUN/f'seed{seed}';target=f/'tensorboard/audited_complete_updates'
        if target.exists():raise FileExistsError(target)
        s=pd.read_csv(f/'actual_training/training_samples.csv');u=pd.read_csv(f/'ppo_update_events.csv')
        with SummaryWriter(str(target)) as w:
            w.add_text('provenance','Actual moving-task PPO CSV export; no resampling, no additional optimisation')
            for r in s.itertuples():
                for k in ['reward','actor_mean','actor_std','raw_action','clipped_action','mapped_Fz_N']:w.add_scalar('actual_sampling/'+k,getattr(r,k),int(r.global_decision))
            for r in u.itertuples():
                for k in ['actor_change_norm','critic_change_norm','policy_loss','value_loss','approx_kl','explained_variance','action_std','advantage_mean','advantage_std','value_target_mean','value_target_std']:w.add_scalar('actual_update/'+k,getattr(r,k),int(r.global_decisions))
if __name__=='__main__':main()
