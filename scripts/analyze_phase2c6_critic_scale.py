"""Quantify how the disturbance curriculum changes critic target scale."""
import numpy as np
import pandas as pd

from phase2c6_protocol import ROOT, RUN, read


def main():
    rows=[]
    for group in ("training","curriculum_training"):
        for root in sorted((RUN/group).glob("seed*")):
            for event in read(root/"updates.json"):
                with np.load(root/f"rollout_{event['update']:03d}.npz") as b:
                    r=b["returns"].astype(float)
                    v=b["values"].astype(float)
                    a=b["advantages"].astype(float)
                    rows.append(dict(group=group,seed=root.name,update=event["update"],decisions=event["decisions"],
                         reward_mean=float(b["rewards"].mean()),reward_min=float(b["rewards"].min()),
                         target_mean=r.mean(),target_std=r.std(),target_min=r.min(),target_max=r.max(),
                         pre_update_value_mse=np.mean((v-r)**2),advantage_min=a.min(),advantage_max=a.max(),
                         value_loss=event["train/value_loss"],explained_variance=event["train/explained_variance"],
                         actor_delta=event["actor_delta"],critic_delta=event["critic_delta"],
                         approx_kl=event["train/approx_kl"]))
    pd.DataFrame(rows).to_csv(ROOT/"results/phase2c6_critic_reward_scale.csv",index=False)


if __name__=="__main__":
    main()
