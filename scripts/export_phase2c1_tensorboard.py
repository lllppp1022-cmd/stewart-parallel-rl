"""Export every completed update, including SB3's undumped last update, from CSV."""
import _bootstrap,argparse
from pathlib import Path
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def main(root):
    for seed in [42,123,2026]:
        base=Path(root)/f'seed{seed}';target=base/'tensorboard/audited_complete_updates'
        if target.exists():raise FileExistsError(target)
        with SummaryWriter(str(target)) as writer:
            writer.add_text('provenance','Offline export of actual training_samples.csv and ppo_update_events.csv; no new sampling or optimization')
            d=pd.read_csv(base/'actual_training/training_samples.csv');u=pd.read_csv(base/'ppo_update_events.csv')
            for r in d.itertuples():
                for field in ['reward','raw_action','clipped_action','mapped_Fz_N','actor_mean','actor_std']:writer.add_scalar('actual_sampling/'+field,getattr(r,field),int(r.global_decision))
            for r in u.itertuples():
                for field in ['actor_change_norm','critic_change_norm','policy_loss','value_loss','approx_kl','entropy_loss','explained_variance','clip_fraction','advantage_mean','advantage_std','value_target_mean','value_target_std','action_std']:writer.add_scalar('actual_update/'+field,getattr(r,field),int(r.global_decisions))
        print('EXPORTED actual metrics',seed,len(d),len(u),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);main(p.parse_args().run_root)
