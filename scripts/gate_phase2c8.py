"""Pre-frozen promotion criteria, never selected from final-test outcomes."""
import argparse
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read,write_json,sha
KEYS=('so3_peak_deg','omega_peak_rad_s','z_rmse_mm','xyz_rmse_mm','xy_peak_mm','force_rate_N_s','force_squared_integral_N2_s')
def pilot():
    rows=[read(p) for p in (RUN/'development').rglob('metrics.json')];d=pd.DataFrame(rows);d=d[d.decisions==8192];details=[];good=0
    for seed in SEEDS:
        a=d[(d.method=='v2_control')&(d.seed==seed)].set_index('task_id');b=d[(d.method=='v3')&(d.seed==seed)].set_index('task_id')
        assert set(a.index)==set(b.index)=={t['id'] for t in cfg('evaluation')['development']}
        safety=bool(a.success.all() and b.success.all());guard=all(np.all(b[k]<=a[k]*1.1+1e-12) for k in KEYS)
        load=1-b.load_iae_deg_s.mean()/a.load_iae_deg_s.mean();unload=1-b.unload_iae_deg_s.mean()/a.unload_iae_deg_s.mean();positive=load>=.01 and unload>=.01 and guard and safety
        good+=int(positive);details.append(dict(seed=seed,load_improvement=float(load),unload_improvement=float(unload),all_task_cost_guard=guard,safe_complete=safety,promising=bool(positive),no_large_iae_regression=bool(load>=-.1 and unload>=-.1)))
    summaries=[read(RUN/f'analysis/{m}/seed{s}/summary.json') for m in ('v2_control','v3') for s in SEEDS]
    matched=all(s['new_decisions']==8192 and s['physical_steps']==163840 and s['failures']==0 and not s['error'] for s in summaries)
    passed=good>=2 and matched and all(r['no_large_iae_regression'] and r['all_task_cost_guard'] and r['safe_complete'] for r in details)
    write_json(RUN/'pilot_gate.json',dict(passed=passed,promising_seeds=good,matched_budget=matched,details=details,criteria=cfg('training')['pilot_gate'],source='development only',no_final_policy_evaluated=True))
    write_json(RUN/'normalization_admission.json',dict(passed=passed,condition='Reward V3 must first pass common physical pilot gate; large Critic loss alone insufficient',status='eligible for value audit' if passed else 'NOT EXECUTED: no common positive reward pilot gate'))
    pd.DataFrame(details).to_csv(ROOT/'results/phase2c8_reward_ablation.csv',index=False);print('PILOT GATE',passed,details,flush=True)
def selection():
    g=read(RUN/'pilot_gate.json');budget=32768 if g['passed'] else 8192
    methods=['v2_control','v3']
    if (RUN/'normalization_pilot_gate.json').exists():
        ng=read(RUN/'normalization_pilot_gate.json')
        if ng['passed']:methods.append('v3_return_norm')
    models={f'{m}/seed{s}':dict(path=str(ROOT/f'models/phase2c8/{m}/seed{s}/checkpoint_{budget:06d}.zip'),sha=sha(ROOT/f'models/phase2c8/{m}/seed{s}/checkpoint_{budget:06d}.zip')) for m in methods for s in SEEDS}
    p=RUN/'final_selection.json';assert not p.exists();write_json(p,dict(frozen=True,budget=budget,models=models,evaluation_sha=sha(ROOT/'configs/phase2c8_evaluation.yaml'),no_best_seed_or_checkpoint_selection=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','selection']);globals()[p.parse_args().mode]()
