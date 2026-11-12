import numpy as np
import pandas as pd
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,read

def main():
    cfg=read(ROOT/'configs/phase2c7_ppo.yaml');rows=[]
    for seed in SEEDS:
        for update in range(1,65):
            path=OLD/f'training/seed{seed}'/('pilot' if update<=16 else 'continue')/f'rollout_{update:03d}.npz'
            with np.load(path) as b:
                reward,value,adv,ret,start=[b[k].reshape(-1).astype(float) for k in ('rewards','values','advantages','returns','episode_starts')]
            mask=1-start[1:]
            recomputed=reward[:-1]+cfg['gamma']*value[1:]*mask-value[:-1]+cfg['gamma']*cfg['gae_lambda']*mask*adv[1:]
            rows.append(dict(seed=seed,update=update,decisions=update*512,gae_internal_max_error=float(abs(recomputed-adv[:-1]).max()),
                return_value_plus_advantage_max_error=float(abs(ret-(value+adv)).max()),
                pre_update_value_mse=float(np.mean((ret-value)**2)),pre_update_explained_variance=float(1-np.var(ret-value)/np.var(ret)),
                boundary_note='Last GAE entry retained from actual rollout: final bootstrap observation/value not independently logged; do not invent it',source=str(path)))
    pd.DataFrame(rows).to_csv(RUN/'value_recomputation.csv',index=False)
if __name__=='__main__':main()
