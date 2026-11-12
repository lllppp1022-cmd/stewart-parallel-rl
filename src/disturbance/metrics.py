"""Independent recomputation from full-rate physical logs; no summary metric inputs."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation

def vec(df,name,n):return df[[f'{name}_{j}' for j in range(n)]].to_numpy(float)
def error_series(df):
    p=np.linalg.norm(vec(df,'p',3)-vec(df,'p_ref',3),axis=1)*1000
    ra=Rotation.from_euler('xyz',vec(df,'rpy',3),degrees=True);rr=Rotation.from_euler('xyz',vec(df,'rpy_ref',3),degrees=True)
    rotation=(rr*ra.inv()).as_rotvec()
    return p,np.rad2deg(np.linalg.norm(rotation,axis=1)),rotation

def recovery(times,good,start,end,hold=.3,dt=.001,complete=True):
    mask=(times>start+1e-10)&(times<=end+1e-10);idx=np.flatnonzero(mask);count=round(hold/dt)
    if len(idx)>=count:
        flags=np.asarray(good)[idx];outside=np.flatnonzero(~flags)
        if not len(outside):return dict(time_s=None,status='NO EXCURSION / WITHIN TOLERANCE' if complete else 'WITHIN TOLERANCE UNTIL CENSOR',right_censored=not complete)
        # Initial in-band latency before the robot responds is not a recovery.
        first=int(outside[0])+1;remaining=flags[first:]
        if len(remaining)>=count:
            runs=np.convolve(remaining.astype(int),np.ones(count,dtype=int),'valid');found=np.flatnonzero(runs==count)
            if len(found):return dict(time_s=float(times[idx[first+found[0]]]-start),status='RECOVERED',right_censored=False)
    return dict(time_s=None,status='NOT RECOVERED' if complete else 'RIGHT CENSORED / EARLY STOP',right_censored=not complete)

def compute(folder,protocol,baseline=None):
    folder=Path(folder);df=pd.read_csv(folder/'physics.csv.gz');cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));pr=json.loads((folder/'disturbance_profile.json').read_text(encoding='utf-8'))
    times=df.time_s.to_numpy();dt=protocol['dt'];on=pr['start_step']*dt;off=pr['stop_step']*dt;duration=cfg['max_sim_seconds'];p,a,rot=error_series(df);good=(p<=protocol['recovery_position_mm'])&(a<=protocol['recovery_angle_deg']);finished=len(df)>=round(duration/dt);failed=df.failure.notna().any();tail=(times>duration-.5-1e-10)
    success=bool(finished and not failed and tail.sum()>=500 and p[tail].max()<.75 and a[tail].max()<.1)
    load=(df.input_step>=pr['start_step'])&(df.input_step<pr['stop_step']);post=(df.input_step>=pr['stop_step'])&(times<=duration+1e-10);pre=(times>max(0,on-.5))&(df.input_step<pr['start_step']);after=df.input_step>=pr['start_step'];forces=vec(df,'force',6);dq=np.diff(vec(df,'q',6),axis=0)/dt
    out=dict(controller=cfg['controller'],trajectory=cfg['trajectory'],task=cfg.get('task',''),stage=cfg.get('stage',''),folder=str(folder),success=success,completed=bool(finished),failure_reason=str(df.failure.dropna().iloc[0]) if failed else ('' if success else 'completion_accuracy'),physical_steps=len(df),simulation_time_s=float(times[-1]),progress=float(min(times[-1]/duration,1)),load_started=bool(df.input_step.max()>=pr['start_step']),unload_reached=bool(df.input_step.max()>=pr['stop_step']),position_rmse_mm=float(np.sqrt(np.mean(p*p))),orientation_rmse_deg=float(np.sqrt(np.mean(a*a))),peak_position_mm=float(p[after].max()) if after.any() else None,peak_orientation_deg=float(a[after].max()) if after.any() else None,peak_z_error_mm=float(np.max(abs(vec(df,'p',3)[after,2]-vec(df,'p_ref',3)[after,2]))*1000) if after.any() else None,peak_roll_error_deg=float(np.max(abs(rot[after,0]))*180/np.pi) if after.any() else None,peak_pitch_error_deg=float(np.max(abs(rot[after,1]))*180/np.pi) if after.any() else None,pre_position_rmse_mm=float(np.sqrt(np.mean(p[pre]**2))) if pre.any() else None,pre_orientation_rmse_deg=float(np.sqrt(np.mean(a[pre]**2))) if pre.any() else None,max_force_N=float(abs(forces).max()),max_force_rate_N_s=float(abs(np.diff(forces,axis=0)/dt).max()) if len(df)>1 else 0.,force_square_integral_N2_s=float(np.sum(forces**2)*dt),command_saturated_steps=int(df.command_clipped.sum()),native_saturated_steps=int(df.native_force_saturated.sum()),max_weld_translation_m=float(df.weld_translation_m.max()),max_weld_angle_rad=float(df.weld_angle_rad.max()),minimum_partial_proxy_m=float(df.proxy_clearance_m.min()),proxy_distance_lower_bound_seen=bool(df.proxy_distance_lower_bound.any()),maximum_jacobian_condition=float(df.jacobian_condition.max()),maximum_mapping_error=float(df.generalized_mapping_error.max()),warning_count=int(df.warning_count.max()),native_contact_observations=int(df.native_contacts.sum()),first_failure_time_s=float(times[np.flatnonzero(df.failure.notna())[0]]) if failed else None,final_position_error_mm=float(p[-1]),final_orientation_error_deg=float(a[-1]),maximum_slide_tracking_error_mm=float(abs(vec(df,'q',6)-vec(df,'q_ref',6)).max()*1000),max_slide_speed_m_s=float(abs(dq).max()) if len(dq) else 0.)
    for prefix,mask,begin,end in [('load',load,on,off),('unload',post,off,duration)]:
        observed=float(mask.sum()*dt);full=bool(times[-1]>=end-1e-10)
        out.update({prefix+'_observed_s':observed,prefix+'_window_complete':full,prefix+'_position_iae_mm_s':float(p[mask].sum()*dt),prefix+'_orientation_iae_deg_s':float(a[mask].sum()*dt),prefix+'_position_mean_mm':float(p[mask].mean()) if mask.any() else None,prefix+'_orientation_mean_deg':float(a[mask].mean()) if mask.any() else None})
        rec=recovery(times,good,begin,end,protocol['recovery_hold_s'],dt,full)
        out[prefix+'_recovery_s']=rec['time_s'];out[prefix+'_recovery_status']=('NOT RECOVERED DURING LOAD' if prefix=='load' and rec['status']=='NOT RECOVERED' else rec['status'])
    for j in range(6):
        out[f'pre_force_leg{j+1}_N']=float(forces[pre,j].mean()) if pre.any() else None;out[f'load_force_leg{j+1}_N']=float(forces[load,j].mean()) if load.any() else None;out[f'pre_slide_leg{j+1}_m']=float(vec(df,'q',6)[pre,j].mean()) if pre.any() else None
    out['pre_linear_velocity_rms_m_s']=float(np.sqrt(np.mean(vec(df,'twist_actual',6)[pre,:3]**2))) if pre.any() else None
    out['pre_angular_velocity_rms_rad_s']=float(np.sqrt(np.mean(vec(df,'twist_actual',6)[pre,3:]**2))) if pre.any() else None
    if baseline is not None:
        b=pd.read_csv(Path(baseline)/'physics.csv.gz');n=min(len(b),len(df));assert np.array_equal(df.input_step[:n],b.input_step[:n]);assert np.array_equal(vec(df,'p_ref',3)[:n],vec(b,'p_ref',3)[:n]);assert np.array_equal(vec(df,'rpy_ref',3)[:n],vec(b,'rpy_ref',3)[:n])
        extra=np.linalg.norm(vec(df,'p',3)[:n]-vec(b,'p',3)[:n],axis=1)*1000;angle=np.rad2deg((Rotation.from_euler('xyz',vec(df,'rpy',3)[:n],degrees=True)*Rotation.from_euler('xyz',vec(b,'rpy',3)[:n],degrees=True).inv()).magnitude());sel=np.asarray(after[:n]);out.update(paired_zero=str(baseline),paired_observed_steps=n,extra_peak_position_mm=float(extra[sel].max()) if sel.any() else None,extra_peak_orientation_deg=float(angle[sel].max()) if sel.any() else None,compensation_peak_vs_paired_zero_N=float(abs(forces[:n]-vec(b,'force',6)[:n]).max()),unload_z_reverse_peak_mm=float(abs(vec(df,'p',3)[post,2]-vec(df,'p_ref',3)[post,2]).max()*1000) if post.any() else None)
    return out
