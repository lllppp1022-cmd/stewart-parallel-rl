"""Open untouched final tasks only after common budgets/checkpoints are frozen."""
import _bootstrap
import json
import torch,pandas as pd
from research_phase2c2 import RUN,PROTOCOL,episode
from src.model import ROOT,write_json,sha
from src.rl.ppo_trace import TracedPPO,parameter_hash

def main():
    torch.set_num_threads(1);frozen=json.loads((RUN/'final_evaluation_frozen.json').read_text(encoding="utf-8"));assert not (RUN/'final_evaluation_results.json').exists();rows=[]
    models=[]
    for x in frozen['models']:
        assert sha(x['checkpoint'])==x['sha256'];m=TracedPPO.load(x['checkpoint']);assert parameter_hash(m.policy)==x['parameter_hash'];models.append((x['seed'],m))
    write_json(RUN/'final_open_event.json',dict(protocol_sha256=sha(RUN/'protocol_frozen.json'),checkpoints=frozen['models'],frozen_selection=True))
    for task in PROTOCOL['final']:
        for method in ['c4','fz_lead','four_d_lead']:
            r=episode(RUN/'final_evaluation'/task['id']/method,task,baseline=method);r.update(label=method,training_seed=None);rows.append(r);write_json(RUN/'final_evaluation_progress.json',rows)
        for seed,m in models:
            r=episode(RUN/'final_evaluation'/task['id']/f'PPO{seed}',task,model=m,record=task['id']=='F_move1' and seed==42);r.update(label=f'PPO{seed}',training_seed=seed);rows.append(r);write_json(RUN/'final_evaluation_progress.json',rows)
    # Old Home case explicitly regression, never called a fresh final split.
    task=dict(id='old_home_regression',tau_s=.010,Fz_N=-6.,on_step=1700,off_step=3450,amplitude_m=0.,shift_s=0.)
    reg=[episode(RUN/'regression'/'four_d_lead',task)]
    for seed,m in models:reg.append(dict(episode(RUN/'regression'/f'PPO{seed}',task,model=m),training_seed=seed))
    write_json(RUN/'regression_results.json',reg);write_json(RUN/'final_evaluation_results.json',rows)
    pd.DataFrame([{k:v for k,v in r.items() if k!='task'} for r in rows]).to_csv(ROOT/'results/phase2c2_evaluation_comparison.csv',index=False)
    ratios=[]
    for seed,_ in models:
        pr=[r for r in rows if r['label']==f'PPO{seed}'];bs=[r for r in rows if r['label']=='four_d_lead'];ratio=sum(r['z_rmse_mm'] for r in pr)/sum(r['z_rmse_mm'] for r in bs)
        ratios.append(dict(seed=seed,z_rmse_ratio=ratio,safe=all(r['success'] and not r['native_contact_observations'] and not r['warning_count'] for r in pr),energy_ratio=sum(r['force_square_integral_N2_s'] for r in pr)/sum(r['force_square_integral_N2_s'] for r in bs),slew_ratio=max(r['max_force_rate_N_s']/b['max_force_rate_N_s'] for r,b in zip(pr,bs))))
    passed=all(r['safe'] and r['z_rmse_ratio']<1 and r['energy_ratio']<=1.02 and r['slew_ratio']<=1.05 for r in ratios) and sum(r['z_rmse_ratio'] for r in ratios)/len(ratios)<=.90
    write_json(RUN/'expansion_gate.json',dict(passed=passed,ratios=ratios,decision='Freeze a new larger study protocol before expansion' if passed else 'STOP SCALING: bounded pilot did not establish required benefit',thresholds=PROTOCOL['expansion_gate']))
if __name__=='__main__':main()
