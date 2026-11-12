"""Finite candidate screening, never learns weights or changes frozen C4."""
import json
from pathlib import Path
import pandas as pd
from ..model import ROOT,write_json

def select_candidates(root,plan):
    root=Path(root);baseline=json.loads((root/'lag_development/zload/C4_tau0/metrics.json').read_text(encoding='utf-8'));reference=baseline['load_position_iae_mm_s'];chosen=[];audit=[]
    for stage,kind in [('lag_development','actuator_lag'),('delay_development','platform_delay')]:
        candidates=[]
        for p in (root/stage/'zload').glob('C4_*/metrics.json'):
            r=json.loads(p.read_text(encoding='utf-8'));cfg=json.loads((p.parent/'config.json').read_text(encoding='utf-8'));ratio=r['load_position_iae_mm_s']/reference
            nonzero=(cfg['actuator_dynamics']['enabled'] if kind=='actuator_lag' else cfg['measurement'].get('delay_steps',0)>0)
            item=dict(kind=kind,source=str(p.parent),success=r['success'],degradation_ratio=ratio,nonzero_factor=bool(nonzero),tau=cfg['actuator_dynamics'],measurement=cfg['measurement'],eligible=bool(r['success'] and nonzero and ratio>=plan['candidate_degradation_min_ratio']))
            audit.append(item)
            if item['eligible']:candidates.append(item)
        if candidates:chosen.append(max(candidates,key=lambda x:x['degradation_ratio']))
    write_json(root/'candidate_screening.json',dict(primary='load position IAE',matched_direct_reference=reference,all_cases=audit,selected=chosen,no_final_feedback=True))
    pd.DataFrame(audit).to_csv(ROOT/'results/phase2c0_candidate_tasks.csv',index=False)
    return chosen

def evaluate_benefit(baseline,method,gate):
    ratio=method['load_position_iae_mm_s']/baseline['load_position_iae_mm_s']
    xy_ratio=method['xy_peak_mm']/max(baseline['xy_peak_mm'],1e-12)
    return dict(primary_error_reduction=1-ratio,success_preserved=bool(method['success'] and baseline['success']),force_slew_ratio=method['max_force_rate_N_s']/baseline['max_force_rate_N_s'],force_square_integral_ratio=method['force_square_integral_N2_s']/baseline['force_square_integral_N2_s'],xy_peak_ratio=xy_ratio,passed=bool(method['success'] and baseline['success'] and 1-ratio>=gate['primary_error_reduction_min_fraction'] and method['max_force_rate_N_s']<=baseline['max_force_rate_N_s']*gate['actual_force_slew_max_ratio'] and method['force_square_integral_N2_s']<=baseline['force_square_integral_N2_s']*gate['force_square_integral_max_ratio'] and xy_ratio<=gate['non_target_xy_peak_max_ratio']))
