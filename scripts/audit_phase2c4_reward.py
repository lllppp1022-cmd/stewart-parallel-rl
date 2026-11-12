"""Independent vector equations, not calls to the online reward function."""
import _bootstrap,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json
from src.rl.stewart_large_motion_env import RUN

def audit(folder):
    d=pd.read_csv(Path(folder)/'physics.csv.gz');e=d[[f'p_{j}' for j in range(3)]].to_numpy()-d[[f'p_ref_{j}' for j in range(3)]].to_numpy();rate=d[[f'ff_total_add_rate_N_s_{j}' for j in range(6)]].to_numpy()
    parts=np.c_[(e[:,2]/1e-5)**2,.1*((e[:,:2]/5e-5)**2).sum(axis=1),.05*(np.deg2rad(d.orientation_error_deg.to_numpy())/.00034906585039886593)**2,.01*(d.rl_requested_Fz_N.to_numpy()/.1)**2,.005*((rate/20)**2).mean(axis=1)];parts*=np.minimum(1,100/np.maximum(parts.sum(axis=1),1e-30))[:,None];logged=d[['reward_cost_'+k for k in ['z','xy','rotation','action','slew']]].to_numpy();err=float(abs(parts-logged).max());assert err<1e-9
    requested=d.rl_requested_Fz_N.to_numpy();accepted=d.rl_accepted_Fz_N.to_numpy();eligible=abs(requested)>1e-8;effective=eligible&(abs(accepted)>1e-7);suppressed=eligible&(abs(accepted-requested)>np.maximum(1e-6,.05*abs(requested)))
    decisions=pd.DataFrame(parts,columns=['z','xy','rotation','action','slew']);decisions['decision']=d.rl_decision;group=decisions.groupby('decision').sum()*(-.001/.05);return dict(folder=str(folder),physical_steps=len(d),maximum_component_difference=err,independent_return=float(-parts.sum()*.001/.05),requested_square_sum=float(np.sum(requested**2)),accepted_square_sum=float(np.sum(accepted**2)),eligible_action_physical_steps=int(eligible.sum()),effective_nonzero_steps=int(effective.sum()),request_difference_over5percent_steps=int(suppressed.sum()),passed=True),group

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',action='store_true');a=p.parse_args();folders=list((RUN/'training').glob('seed*/episode_*/physics.csv.gz')) if a.training else [RUN/f'validation/{s}/episode_0000/physics.csv.gz' for s in ['zero_env','positive','negative','uniform','alternating']];rows=[]
    for file in folders:
        r,g=audit(file.parent);rows.append(r)
        if a.training:
            samples=pd.read_csv(file.parent.parent/'training_samples.csv');ep=int(file.parent.name.split('_')[-1]);s=samples[samples.episode==ep].set_index('decision');ids=s.index.intersection(g.index)
            for k in g:assert np.allclose(g.loc[ids,k],s.loc[ids,'reward_'+k],atol=1e-8)
    write_json(RUN/('reward_training_independent.json' if a.training else 'validation/reward_independent.json'),dict(passed=all(x['passed'] for x in rows),trials=rows))
    if a.training:
        physical=[]
        for seed in [42,123,2026]:
            group=[r for r in rows if f'seed{seed}' in Path(r['folder']).parts];n=sum(r['physical_steps'] for r in group);eligible=sum(r['eligible_action_physical_steps'] for r in group)
            physical.append(dict(seed=seed,physical_steps=n,requested_rms_N=float(np.sqrt(sum(r['requested_square_sum'] for r in group)/n)),accepted_command_rms_N=float(np.sqrt(sum(r['accepted_square_sum'] for r in group)/n)),effective_nonzero_fraction=sum(r['effective_nonzero_steps'] for r in group)/max(eligible,1),request_difference_over5percent_fraction=sum(r['request_difference_over5percent_steps'] for r in group)/max(eligible,1)))
        pd.DataFrame(physical).to_csv(ROOT/'results/phase2c4_physical_action_effectiveness.csv',index=False)
    print('INDEPENDENT REWARD AUDIT',len(rows),'passed')
if __name__=='__main__':main()
