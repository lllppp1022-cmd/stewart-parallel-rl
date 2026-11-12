"""Full1ms physical-journal metrics; partial/protected episodes remain incomplete."""
from pathlib import Path
import numpy as np,pandas as pd
from ..model import write_json
from ..disturbance.metrics import vec,error_series

def metrics(folder,reference):
    folder=Path(folder);d=pd.read_csv(folder/'physics.csv.gz');p,angle,rotation=error_series(d);e=vec(d,'p',3)-vec(d,'p_ref',3);force=vec(d,'force',6);q=vec(d,'q',6);qr=vec(d,'q_ref',6);t=d.time_s.to_numpy();complete=len(d)>=round(reference.duration*1000);failed=d.failure.notna().any();tail=t>=reference.duration-.5
    success=bool(complete and not failed and tail.sum()>=500 and p[tail].max()<.75 and angle[tail].max()<.1)
    r=dict(folder=str(folder),success=success,complete=bool(complete),progress=float(min(t[-1]/reference.duration,1)),failure_reason=str(d.failure.dropna().iloc[0]) if failed else '',physical_steps=len(d),duration_s=float(t[-1]),amplitude_m=reference.amplitude,rise_s=reference.rise_s,z_rmse_mm=float(np.sqrt(np.mean(e[:,2]**2))*1000),position_rmse_mm=float(np.sqrt(np.mean(p*p))),peak_z_error_mm=float(max(abs(e[:,2]))*1000),peak_position_error_mm=float(max(p)),so3_rmse_deg=float(np.sqrt(np.mean(angle*angle))),final_home_error_mm=float(np.linalg.norm(vec(d,'p',3)[-1]-reference.geometry.home_position)*1000),max_force_N=float(abs(force).max()),max_force_rate_N_s=float(abs(np.diff(force,axis=0)/.001).max()) if len(d)>1 else 0.,force_energy_N2_s=float((force*force).sum()*.001),native_saturation_steps=int(d.native_force_saturated.sum()),max_weld_translation_m=float(d.weld_translation_m.max()),max_weld_angle_rad=float(d.weld_angle_rad.max()),expanded_mesh_gap_m=float(d.expanded_mesh_gap_m.min()),legacy_proxy_gap_m=float(d.proxy_clearance_m.min()),max_jacobian_condition=float(d.actual_jacobian_condition.max()),native_contact_observations=int(d.native_contacts.sum()),warning_count=int(d.warning_count.max()),actual_z_peak_m=float(d.actual_z_from_home_m.max()),xy_peak_mm=float(np.linalg.norm(e[:,:2],axis=1).max()*1000),observer_z_mean_N=float(d.diag_equivalent_disturbance_2.mean()),observer_z_peak_N=float(abs(d.diag_equivalent_disturbance_2).max()))
    for name,a,b in [('rise',1,1+reference.rise_s),('fall',2+reference.rise_s,2+2*reference.rise_s)]:
        s=(t>=a)&(t<=b);r[name+'_z_rmse_mm']=float(np.sqrt(np.mean(e[s,2]**2))*1000) if s.any() else None
    ref=(d.p_ref_2-reference.geometry.home_position[2]).to_numpy();actual=(d.p_2-reference.geometry.home_position[2]).to_numpy();shifts=range(-min(200,len(d)//4),min(200,len(d)//4)+1);scores=[]
    for k in shifts:scores.append(np.mean((ref[max(0,k):len(ref)+min(0,k)]-actual[max(0,-k):len(actual)-max(0,k)])**2))
    r['best_reference_shift_s']=int(list(shifts)[int(np.argmin(scores))])*.001
    for i in range(6):r[f'leg{i+1}_stroke_rmse_mm']=float(np.sqrt(np.mean((q[:,i]-qr[:,i])**2))*1000);r[f'leg{i+1}_actual_range_mm']=float(np.ptp(q[:,i])*1000);r[f'leg{i+1}_peak_force_N']=float(max(abs(force[:,i])))
    for i,name in enumerate(['roll','pitch','yaw']):r[name+'_peak_drift_deg']=float(max(abs(d[f'rpy_{i}'])))
    write_json(folder/'metrics.json',r);return r
