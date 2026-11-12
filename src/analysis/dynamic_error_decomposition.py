"""Read-only Phase2B3 uneven_dynamic same-time vector decomposition."""
import json
import numpy as np,pandas as pd
from ..model import ROOT,write_json
from ..disturbance.metrics import vec,error_series

def decompose():
    root=ROOT/'results/runs/phase2b3_20261029_continuation_v2/frozen_evaluation/uneven_dynamic';records=[];series={};peaks=[]
    for method in ['C3','C4-PD','C4-Observer','C4_4d20']:
        folder=root/method;d=pd.read_csv(folder/'physics.csv.gz');p,a,rot=error_series(d);e=(vec(d,'p',3)-vec(d,'p_ref',3))*1000;step=d.input_step.to_numpy();series[method]=d
        windows={'preload':step<1600,'load_on_50ms':(step>=1600)&(step<1650),'load_hold':(step>=1650)&(step<3650),'load_all':(step>=1600)&(step<3650),'unload_50ms':(step>=3650)&(step<3700),'unload_recovery':step>=3700}
        for name,mask in windows.items():
            i=np.flatnonzero(mask)[p[mask].argmax()];r=dict(source_folder=str(folder),method=method,window=name,samples=int(mask.sum()),peak_time_s=float(d.time_s.iloc[i]),peak_norm_mm=float(p[i]),so3_at_peak_deg=float(a[i]))
            for j,axis in enumerate('xyz'):r[axis+'_at_norm_peak_mm']=float(e[i,j]);r[axis+'_squared_contribution_percent']=float(100*e[i,j]**2/(p[i]**2)) if p[i] else 0.;r[axis+'_independent_peak_mm']=float(abs(e[mask,j]).max());r[axis+'_rmse_mm']=float(np.sqrt(np.mean(e[mask,j]**2)))
            for j in range(6):r[f'force_at_peak_{j}_N']=float(d[f'force_{j}'].iloc[i]);r[f'slide_at_peak_{j}_m']=float(d[f'q_{j}'].iloc[i]);r[f'slide_velocity_at_peak_{j}_m_s']=float(d[f'dq_{j}'].iloc[i])
            records.append(r)
            if name=='load_all':peaks.append(r)
    f=pd.DataFrame(records);f.to_csv(ROOT/'results/phase2c0_error_decomposition.csv',index=False);write_json(ROOT/'results/phase2c0_historical_peak.json',dict(peaks=peaks,paired_history_limitation='No strict no-load/nominal-gain counterpart for this scaled reference; use explicitly new paired experiments, never subtract unrelated old tasks',no_reintegration=True));return f,series
