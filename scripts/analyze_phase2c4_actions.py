"""Offline action timing/sign description, not a causal identification claim."""
import _bootstrap
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT

def main():
    c=pd.read_csv(ROOT/'results/phase2c4_controller_comparison.csv');rows=[]
    tasks={'train_fixed':(5.2,11.),'E1_newtiming':(5.6,10.6),'E2_small_load':(4.8,10.2)}
    for r in c[c.method.str.startswith('PPO')].itertuples():
        d=pd.read_csv(Path(r.folder)/'physics.csv.gz',usecols=['time_s','rl_requested_Fz_N','rl_accepted_Fz_N','p_2','p_ref_2'])
        for event,t in zip(['load','unload'],tasks[r.task_id]):
            m=(d.time_s>=t)&(d.time_s<t+1);early=(d.time_s>=t)&(d.time_s<t+.05);later=(d.time_s>=t+.05)&(d.time_s<t+.1);e=(d.p_2-d.p_ref_2).to_numpy();eligible=m&(abs(e)>1e-7)
            rows.append(dict(method=r.method,task_id=r.task_id,event=event,mean_requested_Fz_N=float(d.rl_requested_Fz_N[m].mean()),mean_accepted_command_Fz_N=float(d.rl_accepted_Fz_N[m].mean()),first50ms_request_N=float(d.rl_requested_Fz_N[early].mean()),next50ms_request_N=float(d.rl_requested_Fz_N[later].mean()),opposite_error_sign_fraction=float(np.mean(d.rl_requested_Fz_N[eligible].to_numpy()*e[eligible]<0)),note='Descriptive same-trajectory action/error signs; paired closed-loop metrics determine benefit, not sign alone.'))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c4_action_window_analysis.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
