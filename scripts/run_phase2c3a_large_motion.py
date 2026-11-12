"""Bounded, gated classical-control experiments. No RL imports/training."""
import _bootstrap,json,time,argparse,copy
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.rl.stewart_residual_env import owner_config
from src.phase1.plant import create_plant
from src.workspace.geometry import ExpandedGeometryQuery
from src.workspace.reference import LargeReference
from src.workspace.execution import LargeOwner
from src.disturbance.recording import install_colored_capture
from src.disturbance.metrics import vec,error_series
from analyze_phase2c3a_workspace import RUN

CONFIG=ROOT/'configs/phase2c3a_large_workspace.json'

def metrics(folder,reference):
    d=pd.read_csv(folder/'physics.csv.gz');p,a,rot=error_series(d);e=vec(d,'p',3)-vec(d,'p_ref',3);force=vec(d,'force',6);q=vec(d,'q',6);qr=vec(d,'q_ref',6);duration=reference.duration;complete=len(d)>=round(duration*1000);failed=d.failure.notna().any();tail=d.time_s>=duration-.5;success=bool(complete and not failed and tail.sum()>=500 and p[tail].max()<.75 and a[tail].max()<.1)
    out=dict(folder=str(folder),success=success,complete=complete,progress=min(float(d.time_s.iloc[-1]/duration),1.),failure_reason=str(d.failure.dropna().iloc[0]) if failed else '',physical_steps=len(d),duration_s=float(d.time_s.iloc[-1]),amplitude_m=reference.amplitude,rise_s=reference.rise_s,z_rmse_mm=float(np.sqrt(np.mean(e[:,2]**2))*1000),position_rmse_mm=float(np.sqrt(np.mean(p*p))),peak_position_error_mm=float(max(p)),so3_rmse_deg=float(np.sqrt(np.mean(a*a))),final_home_error_mm=float(np.linalg.norm(vec(d,'p',3)[-1]-reference.geometry.home_position)*1000),max_force_N=float(abs(force).max()),max_force_rate_N_s=float(abs(np.diff(force,axis=0)/.001).max()),native_saturation_steps=int(d.native_force_saturated.sum()),force_energy_N2_s=float((force*force).sum()*.001),max_weld_translation_m=float(d.weld_translation_m.max()),max_weld_angle_rad=float(d.weld_angle_rad.max()),expanded_mesh_gap_m=float(d.expanded_mesh_gap_m.min()),legacy_proxy_gap_m=float(d.proxy_clearance_m.min()),max_jacobian_condition=float(d.actual_jacobian_condition.max()),native_contact_observations=int(d.native_contacts.sum()),warning_count=int(d.warning_count.max()),actual_z_peak_m=float(d.actual_z_from_home_m.max()),minimum_native_limit_margin=float(d.native_limit_margin.min()),native_unlimited_passive_count=30)
    for key,mask in [('rise',(d.time_s>=1)&(d.time_s<=1+reference.rise_s)),('fall',(d.time_s>=2+reference.rise_s)&(d.time_s<=2+2*reference.rise_s))]:out[key+'_z_rmse_mm']=float(np.sqrt(np.mean(e[mask,2]**2))*1000)
    # Phase alignment over a frozen ±0.2s window, not a pure-deadtime identification.
    ref=(d.p_ref_2-reference.geometry.home_position[2]).to_numpy();actual=(d.p_2-reference.geometry.home_position[2]).to_numpy();shifts=range(-200,201);scores=[]
    for k in shifts:
        x=ref[max(0,k):len(ref)+min(0,k)];y=actual[max(0,-k):len(actual)-max(0,k)];scores.append(np.mean((x-y)**2))
    out['best_reference_shift_s']=int(list(shifts)[int(np.argmin(scores))])*.001
    for i in range(6):
        out[f'leg{i+1}_actual_range_mm']=float(np.ptp(q[:,i])*1000);out[f'leg{i+1}_stroke_rmse_mm']=float(np.sqrt(np.mean((q[:,i]-qr[:,i])**2))*1000);out[f'leg{i+1}_peak_force_N']=float(abs(force[:,i]).max())
    for i,n in enumerate(['roll','pitch','yaw']):out[n+'_peak_drift_deg']=float(abs(vec(d,'rpy',3)[:,i]).max())
    out['xy_peak_mm']=float(np.linalg.norm(e[:,:2],axis=1).max()*1000);out['observer_z_mean_N']=float(d.diag_equivalent_disturbance_2.mean());out['observer_z_peak_N']=float(abs(d.diag_equivalent_disturbance_2).max());out['compensation_peak_leg_N']=float(abs(vec(d,'F_add_total',6)).max())
    out['maximum_actual_z_speed_m_s']=float(abs(d.twist_actual_2).max());out['maximum_actual_z_acceleration_m_s2']=float(abs(np.diff(d.twist_actual_2)/.001).max());moving=abs(d.twist_ref_2.to_numpy()[1:])>1e-6;acc=np.diff(d.twist_actual_2)/.001;out['maximum_moving_z_acceleration_m_s2']=float(abs(acc[moving]).max()) if moving.any() else 0.
    out['shared_compensation_budget_peak_fraction']=out['compensation_peak_leg_N']/5;out['observer_clipped_steps']=int(d.diag_observer_clipped.sum());out['compensation_rate_limited_steps']=int(d.diag_compensation_rate_limited.sum());out['compensation_amplitude_limited_steps']=int(d.diag_compensation_amplitude_limited.sum());out['maximum_leg_force_spread_N']=float(np.ptp(force,axis=1).max());write_json(folder/'metrics.json',out);return out

def trial(method,amp,rise,tau=0.,load=0.,record=False):
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));limits=json.loads((ROOT/'results/phase2c3a_workspace_limits.json').read_text(encoding='utf-8'))
    if amp and not limits['candidates'][f'L{round(amp*1000)}']['simulation_preflight_passed']:raise RuntimeError('Candidate did not pass geometry')
    query=ExpandedGeometryQuery();reference=LargeReference(create_plant(),amp,rise,RUN/'references',query);name=f'{method}_L{round(amp*1000)}_T{rise:g}_tau{tau:g}_load{load:g}';folder=RUN/'dynamics'/name
    task=dict(id=name,tau_s=tau,Fz_N=load,on_step=round((1+.7*rise)*1000),off_step=round((2+1.5*rise)*1000));oc=owner_config(folder,task,record);oc.update(stage='phase2c3a',max_sim_seconds=reference.duration,max_wall_seconds=300,controller='C0' if method=='C0' else 'C3')
    if method=='C0':oc['yaw_policy']=dict(kind='zero4')
    if method=='fourDlead':oc['diagnostic']=dict(kind='four_d_lead')
    owner=LargeOwner(oc,reference,query,cfg);owner.config['trajectory']=reference.name;write_json(folder/'config.json',owner.config);write_json(folder/'large_workspace.json',cfg);write_json(folder/'reference_preflight.json',reference.audit)
    # Nominal force-headroom screen, not a bound on actual nonlinear transients.
    local=oc['task_model'];M=np.array(local['mass']);D=np.array(local['damping']);inertia=np.zeros(6);damping=np.zeros(6);inertia[2]=reference.audit['max_acceleration_m_s2'];damping[2]=reference.audit['max_speed_m_s'];j=owner.residual.base.jhome
    conservative_nominal_peak=reference.audit['max_static_force_N']+float(abs(np.linalg.solve(j.T,M@inertia)).max())+float(abs(np.linalg.solve(j.T,D@damping)).max())
    write_json(folder/'force_headroom_preflight.json',dict(nominal_component_peak_estimate_N=conservative_nominal_peak,native_force_limit_N=400,accepted=conservative_nominal_peak<300,scope='Component-wise nominal screen with100N reserve; live ±400N still enforced'))
    if conservative_nominal_peak>=300:
        owner.recorder.close(dict(physics_steps=0,complete=False,failure='nominal_force_preflight'));raise RuntimeError('Nominal force screen failed')
    if record:install_colored_capture(owner.recorder,owner.model)
    start=time.perf_counter();exception=None
    try:
        while owner.data.time<reference.duration-1e-10:
            if time.perf_counter()-start>oc['max_wall_seconds'] and not owner.failure:owner.hold('wall_budget_exceeded',failed=True)
            owner.step()
            if owner.failure and owner.data.time>=owner.protect_end-1e-10:break
    except Exception as exc:exception=repr(exc);owner.event('exception',error=exception);raise
    finally:owner.recorder.close(dict(physics_steps=owner.n,failure=owner.failure,exception=exception,requested_duration_s=reference.duration,complete=owner.n>=round(reference.duration*1000) and not owner.failure,no_live_qpos_writes=True,controller=method,source='LARGE-WORKSPACE DYNAMIC MOTION VALIDATION'))
    r=metrics(folder,reference);r.update(method=method,tau_s=tau,external_Fz_N=load,wall_seconds=time.perf_counter()-start,reference=reference.audit);write_json(folder/'trial_result.json',r);print(name,r['success'],r['failure_reason'],'ZRMSE',r['z_rmse_mm'],flush=True);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','slow','speed','load']);a=p.parse_args()
    if a.stage=='prepare':
        scan=pd.read_csv(ROOT/'results/phase2c3a_workspace_scan.csv');tube=scan[(scan.delta_z_m>=-.002-1e-10)&(scan.delta_z_m<=.052+1e-10)]
        if not tube.simulation_geometric_candidate.all():raise RuntimeError('New ±tracking tube failed preflight')
        cfg=dict(version='phase2c3a_independent',actual_z_bounds_m=[-.002,.052],evidence='2mm sampled full-chain geometry interval [-2,+52]mm, online mesh monitoring; not a certified multidimensional invariant set',old_z_limit_m=.005,reference_amplitudes_m=[.03,.05],slow_rise_s=10.,speed_candidates_rise_s=[6.,4.],physical_dt_s=.001,actual_xy_limit_m=.002,actual_roll_pitch_limit_deg=.5,actual_yaw_limit_deg=.2,weld_translation_limit_m=.0001,weld_angle_limit_rad=.0001,jacobian_condition_limit=50,mesh_clearance_m=.005,monitor_period_s=.02,force_limit_N=400.,total_trial_budget=18,rl_training_allowed=False,passive_physical_hard_stops_known=False)
        if CONFIG.exists():raise FileExistsError(CONFIG)
        write_json(CONFIG,cfg);write_json(RUN/'dynamic_protocol_frozen.json',dict(slow='L30,L50 x C0,C4,fourDlead, 10s one-way, no disturbances',speed='L50 x C0,C4,fourDlead at6s and4s only if ALL slow trials pass; faster tier requires previous tier all pass',load='L50 10s, C4/fourDlead x(5ms,no load),(10ms,no load),(5ms,-5.5N); only if slow trials pass',additional_runs='Home equivalence and recorder on/off tests separately accounted',no_retries=True))
    else:
        if a.stage!='slow':
            slow=json.loads((RUN/'slow_results.json').read_text(encoding='utf-8'))
            if len(slow)!=6 or not all(r['success'] for r in slow):write_json(RUN/f'{a.stage}_not_executed.json',dict(reason='Not all six slow dynamic trials passed'));return
        results=[]
        if a.stage=='slow':
            for amp in [.03,.05]:
                for method in ['C0','C4','fourDlead']:results.append(trial(method,amp,10.,record=amp==.05));write_json(RUN/'slow_results.json',results)
        elif a.stage=='speed':
            for rise in [6.,4.]:
                tier=[]
                for method in ['C0','C4','fourDlead']:tier.append(trial(method,.05,rise,record=method=='fourDlead'));results.append(tier[-1]);write_json(RUN/'speed_results.json',results)
                if not all(r['success'] for r in tier):break
        else:
            for tau,load in [(.005,0.),(.010,0.),(.005,-5.5)]:
                for method in ['C4','fourDlead']:results.append(trial(method,.05,10.,tau,load,record=tau==.005 and load!=0));write_json(RUN/'load_results.json',results)
if __name__=='__main__':main()
