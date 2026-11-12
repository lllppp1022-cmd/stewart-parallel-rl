"""Independently reconstruct both sides of the same-state governor counterfactual."""
import numpy as np
import pandas as pd
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,read

def main():
    rows=[]
    for task in read(ROOT/'configs/phase2c7_eval.yaml')['final']:
        for seed in SEEDS:
            folder=OLD/'final'/task['id']/f'PPO_{seed}/episode_0000';d=pd.read_csv(folder/'physics.csv.gz')
            cols=lambda key:d[[f'{key}_{k}' for k in range(6)]].to_numpy()
            c0=cols('F_C0');command=cols('F_command')
            desired=cols('F_C3')-c0+cols('F_residual_requested')+d[[f'ff_requested_leg_{k}_N' for k in range(6)]].to_numpy()
            leg=d[[f'rl_mapped_leg_{k}_N' for k in range(6)]].to_numpy();base=cols('same_state_base_add') if 'same_state_base_add_0' in d else d[[f'same_state_base_add_{k}_N' for k in range(6)]].to_numpy()
            previous=np.vstack([np.zeros(6),(command-c0)[:-1]])
            lo=np.maximum(np.maximum(-5,previous-.02),-400-c0);hi=np.minimum(np.minimum(5,previous+.02),400-c0)
            expected_base=np.clip(desired,lo,hi);expected_cmd=c0+np.clip(desired+leg,lo,hi)
            ce=float(abs(expected_cmd-command).max());be=float(abs(expected_base-base).max())
            assert ce<1e-8 and be<1e-8,(task['id'],seed,ce,be)
            rows.append(dict(task_id=task['id'],seed=seed,command_max_error_N=ce,counter_base_max_error_N=be,
                executed_request_limited_fraction=float((abs(expected_cmd-(c0+desired+leg)).max(axis=1)>1e-9).mean()),
                counterfactual_no_rl_limited_fraction=float((abs(expected_base-desired).max(axis=1)>1e-9).mean()),
                counterfactual_rate_limited_fraction=float(((desired<previous-.02-1e-9)|(desired>previous+.02+1e-9)).any(axis=1).mean()),
                counterfactual_amplitude_limited_fraction=float((abs(desired)>5+1e-9).any(axis=1).mean()),
                meaning='Accepted is marginal effect versus a no-RL counterfactual sharing the actual previous command, not a scalar authority allocation'))
    pd.DataFrame(rows).to_csv(RUN/'governor_counterfactual_audit.csv',index=False)
if __name__=='__main__':main()
