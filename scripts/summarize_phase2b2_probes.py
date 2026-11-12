"""Probe-window metrics: precommitted actions run at .8--1.8s, not load2--4s."""
import _bootstrap
import json
import numpy as np,pandas as pd
from src.model import ROOT,write_json
from src.disturbance.metrics import vec,error_series

def main():
    root=ROOT/'results/runs/phase2b2_20261029_v1';rows=[]
    baseline=pd.read_csv(root/'zero_equivalence/force10/C0/physics.csv.gz')
    for folder in sorted((root/'preflight/quiet').glob('*')):
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        if cfg['action']['kind']!='probe':continue
        d=pd.read_csv(folder/'physics.csv.gz');sel=(d.input_step>=800)&(d.input_step<1800);p,a,_=error_series(d)
        delta=(vec(d,'p',3)-vec(baseline,'p',3))*1000
        # Baseline is identical zero input until2s; comparisons only use <=1.8s.
        rows.append(dict(variant=folder.name,action=json.dumps(cfg['action']['vector']),window_start_s=.8,window_end_s=1.8,peak_position_mm=float(p[sel].max()),peak_angle_deg=float(a[sel].max()),delta_x_peak_mm=float(abs(delta[sel,0]).max()),delta_y_peak_mm=float(abs(delta[sel,1]).max()),delta_z_peak_mm=float(abs(delta[sel,2]).max()),signed_final_delta_x_mm=float(delta[1799,0]),signed_final_delta_y_mm=float(delta[1799,1]),signed_final_delta_z_mm=float(delta[1799,2])))
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/phase2b2_probe_response.csv',index=False)
    d=pd.read_csv(root/'preflight/quiet/C3_known_input_no_external/physics.csv.gz');s=(d.time_s>1)&(d.time_s<1.7)
    write_json(ROOT/'results/phase2b2_known_input_check.json',dict(label='Known commanded +0.5N task Fz, no external input',observer_Fz_range_N=[float(d.diag_equivalent_disturbance_2[s].min()),float(d.diag_equivalent_disturbance_2[s].max())],observer_Fz_mean_N=float(d.diag_equivalent_disturbance_2[s].mean()),corrected_memory_steps=int(d.observer_command_corrected.sum()),not_mistaken_for_minus_half_newton=True))
    print(f.to_string(index=False),flush=True)

if __name__=='__main__':main()
