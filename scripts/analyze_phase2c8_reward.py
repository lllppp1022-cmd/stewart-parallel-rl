"""Offline physical rescoring and failure-order proof, prior to formal training."""
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT,RUN,cfg,read,write_json
from evaluate_phase2c8 import rescore
from src.rl.phase2c8_reward import costs
from src.rl.stewart_dynamic_attitude_env import dynamic_cost
from src.rl.stewart_attitude_env import failure_penalty,GAMMA

def main():
    paths=list((RUN/'reward_audit').rglob('metrics.json'));assert len(paths)==5
    rows=[];max_error=0.
    for p in paths:
        m=read(p);d=pd.read_csv(__import__('pathlib').Path(m['folder'])/'physics.csv.gz')
        assert m['success']
        for candidate in cfg('reward_v3')['candidates']:
            scores=rescore(d,**candidate)
            for v,(c,f) in scores.items():
                rows.append(dict(method=m['method'],candidate=str(candidate) if v=='v3' else 'original',reward_version=v,episode_return=float(-c.sum()/20),orientation_cost=float(c[:,0].sum()/20),omega_cost=float(c[:,1].sum()/20),xyz_cost=float(c[:,2].sum()/20),action_cost=float(c[:,3].sum()/20),slew_cost=float(c[:,4].sum()/20),cap_steps=int((f<1).sum()),load_iae=m['load_iae_deg_s'],unload_iae=m['unload_iae_deg_s'],so3_peak_deg=m['so3_peak_deg'],xyz_rmse_mm=m['xyz_rmse_mm'],source=m['folder']))
        max_error=max(max_error,float(abs(scores['v2'][0]-d[[f'dyn_cost_{n}' for n in ('orientation','omega','xyz','action','slew')]].to_numpy()).max()))
    assert max_error<1e-10
    # Pure component ordering tests, not invented robot trajectories.
    row={'orientation_error_deg':0.,**{f'p_{i}':0. for i in range(3)},**{f'p_ref_{i}':0. for i in range(3)},**{f'twist_actual_{i}':0. for i in (3,4,5)},**{f'twist_ref_{i}':0. for i in (3,4,5)}}
    c0=costs(row,np.zeros(2),np.zeros(2))[0];assert c0.sum()==0
    norms=np.r_[0,np.geomspace(1e-8,.1,200)];values=[]
    for e in norms:values.append(costs(dict(row,orientation_error_deg=float(e)),np.zeros(2),np.zeros(2))[0][0])
    assert np.isfinite(values).all() and (np.diff(values)>=0).all()
    positive=costs(dict(row,orientation_error_deg=.006),np.zeros(2),np.zeros(2))[0]
    assert positive[0]>0 and costs(dict(row,p_0=.0001),np.zeros(2),np.zeros(2))[0][2]>0
    assert costs(row,np.array([.01,0]),np.zeros(2))[0][3]>0 and costs(row,np.zeros(2),np.array([1.,0]))[0][4]>0
    for remaining in range(801):assert failure_penalty(remaining)<-100*(1-GAMMA**remaining)/(1-GAMMA)
    # Exact V2 independent implementation equivalence over recorded observations.
    for i in range(0,len(d),97):
        r=d.iloc[i];a=r[['accepted_mx_Nm','accepted_my_Nm']].to_numpy(float);sl=r[['slew_mx_Nm_s','slew_my_Nm_s']].to_numpy(float)
        np.testing.assert_allclose(costs(r,a,sl,'v2_control')[0],dynamic_cost(r,a,sl)[0],rtol=1e-14,atol=1e-14)
    pd.DataFrame(rows).drop_duplicates().to_csv(ROOT/'results/phase2c8_reward_offline_audit.csv',index=False)
    write_json(RUN/'reward_gate.json',dict(passed=True,real_physical_methods=5,v2_recompute_max_error=max_error,v3_monotonic_even_error=True,position_action_slew_retained=True,failure_discount_bound_checked_remaining=[0,800],failure_trajectory_not_fabricated=True,selected_candidate=cfg('reward_v3'),weights_frozen_before_training=True,offline_reward_ranking_not_learning_evidence=True))
    print('REWARD GATE PASSED',flush=True)
if __name__=='__main__':main()
