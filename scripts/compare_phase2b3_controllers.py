"""Read-only full-rate metrics, causal-observability audit and physical validation."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from src.model import ROOT,write_json,sha,load_model
from src.control.phase2b1_experiment import trial_metrics
from src.control.phase2b2_experiment import actual_force_map
from src.disturbance.metrics import vec,recovery,error_series

def analyze(root,continuation=None):
    root=Path(root);protocol=json.loads((root/'protocol_before_experiments.json').read_text(encoding='utf-8'));rows=[];validation=[];memory=[];engineering=[];m=load_model();scratch=mujoco.MjData(m)
    folders=sorted(root.glob('*/*/*'))+(sorted(Path(continuation).glob('*/*/*')) if continuation else [])
    for folder in folders:
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary['physics_steps']==0:
            engineering.append(dict(folder=str(folder),reason=summary['failure'],physics_steps=0,excluded_from_control_success_rate=True));continue
        c=json.loads((folder/'config.json').read_text(encoding='utf-8'));r=trial_metrics(folder,protocol);d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');n=len(d);p,ang,rot=error_series(d);yaw=np.rad2deg(-rot[:,2]);t=d.time_s.to_numpy();pr=c['disturbance'];pre=d.input_step<pr['start_step'];post=d.input_step>=pr['stop_step'];load=(~pre)&(~post);active=d.budget_applicable.to_numpy(bool)
        eyaw=vec(d,'rpy',3)[:,2]-vec(d,'rpy_ref',3)[:,2];force=vec(d,'force',6);cmd=vec(d,'f_cmd',6);add=vec(d,'F_add_total',6);req=vec(d,'F_residual_requested',6);app=vec(d,'F_residual_applied',6);a=vec(d,'true_kinematic_map',36).reshape(-1,6,6);gain=json.loads((folder/'plant.json').read_text(encoding='utf-8'))['gain'];s=pre&(t>max(0,pr['start_step']*.001-.5))
        r.update(variant=folder.name,privileged=c['stage']=='oracle_diagnostic',yaw_peak_deg=float(abs(yaw).max()),yaw_IAE_deg_s=float(abs(yaw).sum()*.001),yaw_display_IAE_deg_s=float(abs(eyaw).sum()*.001),preload_yaw_peak_deg=float(abs(yaw[pre]).max()) if pre.any() else None,preload_yaw_bias_deg=float(yaw[s].mean()) if s.any() else None,yaw_rate_peak_rad_s=float(abs(vec(d,'twist_actual',6)[:,5]).max()),max_xy_error_mm=float(np.linalg.norm(vec(d,'p',3)[:,:2]-vec(d,'p_ref',3)[:,:2],axis=1).max()*1000),z_error_peak_mm=float(abs(vec(d,'p',3)[:,2]-vec(d,'p_ref',3)[:,2]).max()*1000),roll_error_peak_deg=float(abs(np.rad2deg(rot[:,0])).max()),pitch_error_peak_deg=float(abs(np.rad2deg(rot[:,1])).max()),shared_add_peak_N=float(abs(add[active]).max()) if active.any() else None,shared_add_rate_N_s=float(abs(np.diff(np.r_[np.zeros((1,6)),add[active]],axis=0)/.001).max()) if active.any() else None,projected_steps=int(d.action_rejection_reason.eq('projected_shared_budget').sum()),force_command_difference_peak_N=float(abs(force-cmd).max()),requested_residual_peak_N=float(abs(req).max()),applied_residual_peak_N=float(abs(app).max()),zero_effect_fraction=float((np.linalg.norm(app[np.linalg.norm(req,axis=1)>1e-10],axis=1)<1e-10).mean()) if np.any(np.linalg.norm(req,axis=1)>1e-10) else None)
        for prefix,mask,begin,end in [('preload',pre,0,pr['start_step']*.001),('load',load,pr['start_step']*.001,pr['stop_step']*.001),('unload',post,pr['stop_step']*.001,6)]:
            r[prefix+'_yaw_peak_deg']=float(abs(yaw[mask]).max()) if mask.any() else None;r[prefix+'_yaw_IAE_deg_s']=float(abs(yaw[mask]).sum()*.001) if mask.any() else None
            rec=recovery(t,abs(yaw)<=.02,begin,end,.3,.001,t[-1]>=end-1e-10);r[prefix+'_yaw_recovery_s']=rec['time_s'];r[prefix+'_yaw_recovery_status']=rec['status']
        observed=float(np.sign(yaw[s].mean())) if s.any() else 0; r['unload_yaw_reverse_deg']=float(max(0,np.max(-observed*yaw[post]))) if post.any() and observed else None
        # Gain diagnostics are offline; these values never enter a causal policy.
        delta=np.einsum('nij,ni->nj',a,force-cmd);np.testing.assert_allclose(delta,vec(d,'gain_wrench_error',6),atol=1e-11)
        known=vec(d,'diag_equivalent_disturbance',6)[:,5];obswindow=(t>.15)&pre&d.failure.isna().to_numpy();delta_mean=float(delta[obswindow,5].mean()) if obswindow.any() else 0.
        rise=np.flatnonzero(np.sign(delta_mean)*known>=.632*abs(delta_mean)) if abs(delta_mean)>.005 else []
        matrix=a[obswindow,:,5]*cmd[obswindow];sv=np.linalg.svd(matrix,compute_uv=False) if len(matrix) else np.zeros(6)
        memory.append(dict(stage=c['stage'],task=c['task'],variant=folder.name,gain_Mz_mean_Nm=delta_mean,observer_Mz_mean_Nm=float(known[obswindow].mean()) if obswindow.any() else None,observer_gain_Mz_MAE_Nm=float(abs(known[obswindow]-delta[obswindow,5]).mean()) if obswindow.any() else None,estimate_first_63percent_s=float(t[rise[0]]) if len(rise) else None,yaw_excitation_effective_rank=int(np.sum(sv>sv[0]*1e-4)) if sv[0] else 0,yaw_excitation_condition=float(sv[0]/sv[-1]) if sv[-1]>0 else None,rank_caveat='Offline local yaw projection, no claim to identify all six gains; actual geometric map used for diagnostic only'))
        # Full physical state and actual limits, independent archive checks.
        with np.load(folder/'integration_states.npz') as z:
            states=z['states'];assert states.shape==(n,458) and np.isfinite(states).all();np.testing.assert_array_equal(states[:,0],t);np.testing.assert_array_equal(states[:,1:44],vec(d,'qpos',43));np.testing.assert_array_equal(states[:,44:86],vec(d,'qvel',42))
            initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'));map_err=0
            for i in np.linspace(0,n-1,3,dtype=int):
                scratch.qpos[:]=initial['qpos'] if not i else states[i-1,1:44];scratch.qvel[:]=initial['qvel'] if not i else states[i-1,44:86];mujoco.mj_forward(m,scratch);map_err=max(map_err,float(abs(actual_force_map(m,scratch)-a[i]).max()))
            assert map_err<2e-8
        np.testing.assert_allclose(force,np.clip(np.array(gain)*cmd,-400,400),atol=1e-10)
        assert abs(force).max()<=400+1e-9 and abs(cmd).max()<=400+1e-9
        if active.any():
            np.testing.assert_allclose(cmd[active]-vec(d,'F_C0',6)[active],add[active],atol=1e-12,rtol=0);assert abs(add[active]).max()<=5+1e-9 and r['shared_add_rate_N_s']<=20+1e-6
            np.testing.assert_allclose(np.einsum('nij,ni->nj',a,force-np.clip(np.array(gain)*(cmd-app),-400,400)),vec(d,'W_physical_residual',6),atol=1e-10)
        np.testing.assert_array_equal(vec(d,'ctrl',12)[:,6:],vec(d,'external_ctrl',6));assert not c.get('measurement') and set(c.get('actual_perturbation',{}))<={'gain_scales'}
        assert all(abs(np.array(gain)-1)<.01) # Known failed1% never re-executed.
        assert d.warning_count.max()==0
        rows.append(r);validation.append(dict(folder=str(folder),steps=n,state_csv_exact=True,map_error=map_err,shared_budget=True,native_limits=True,channels_isolated=True))
        write_json(folder/'metrics_recomputed.json',r)
        if len(rows)%10==0:print('RECOMPUTED',len(rows),flush=True)
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/phase2b3_all_trials.csv',index=False)
    for stage,name in [('frozen_evaluation','controller_comparison'),('safety_preflight','safety_preflight'),('action_preflight','yaw_action_sensitivity'),('development','development'),('oracle_diagnostic','oracle_diagnostics')]:f[f.stage==stage].to_csv(ROOT/f'results/phase2b3_{name}.csv',index=False)
    pd.DataFrame(memory).to_csv(ROOT/'results/phase2b3_history_observability.csv',index=False)
    hist=json.loads((ROOT/'results/phase2b3_frozen_history.json').read_text(encoding='utf-8'))
    for name,entry in hist['files'].items():assert sha(ROOT/name)==entry['sha256'],name
    frozen=json.loads((root/'evaluation_frozen.json').read_text(encoding='utf-8'));assert frozen['controllers_sha256']==sha(root/'controllers_frozen.json')
    assert len(rows)<=protocol['maximum_research_trials']
    result=dict(trials=len(rows),successes=int(f.success.sum()),failures=int((~f.success).sum()),engineering_failures=engineering,physical_steps=int(f.physical_steps.sum()),history_files_unchanged=len(hist['files']),all_state_archives_exact=True,all_shared_budgets_valid=True,all_native_limits_valid=True,all_channel_isolation_valid=True,max_scratch_mapping_error=max(x['map_error'] for x in validation),final_parameters_frozen=True,no_one_percent_rerun=True,no_RL=True,records=validation)
    write_json(ROOT/'results/phase2b3_validation.json',result);plots(root,f);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    return f

def plots(root,f):
    def read(stage,task,label):return pd.read_csv(root/stage/task/label/'physics.csv.gz')
    def save(fig,name):fig.tight_layout();fig.savefig(ROOT/'assets/plots'/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for row in f[f.stage=='action_preflight'].itertuples():
        d=pd.read_csv(Path(row.folder)/'physics.csv.gz');axes[0].plot(d.time_s,d.task4_target_3,label=row.variant);axes[1].plot(d.time_s,d.rpy_2);axes[2].plot(d.time_s,np.linalg.norm(vec(d,'p',3)[:,:2]-vec(d,'p_ref',3)[:,:2],axis=1)*1000)
    for ax,title in zip(axes,['Requested Mz (Nm); eight signed local probes','Actual yaw (deg)','Non-target XY tracking error (mm)']):ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel('Physical time (s)');save(fig,'phase2b3_task4_yaw_probes.png')
    for task in ['reverse_z','uneven_dynamic']:
        subset=f[(f.stage=='frozen_evaluation')&(f.task==task)];fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
        for row in subset.itertuples():
            d=pd.read_csv(Path(row.folder)/'physics.csv.gz');axes[0].plot(d.time_s,d.rpy_2-d.rpy_ref_2,label=row.variant);axes[1].plot(d.time_s,d.position_error_mm);axes[2].plot(d.time_s,d.orientation_error_deg);axes[3].plot(d.time_s,np.max(abs(vec(d,'F_add_total',6)),axis=1))
        for ax,title in zip(axes,['Yaw display error (deg)','Position error (mm)','SO3 error (deg)','Shared additional force peak per time (N), limit5']):ax.set_title(title);ax.grid(alpha=.2)
        axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Physical time (s)');save(fig,f'phase2b3_{task}_comparison.png')
    formal=f[f.stage=='frozen_evaluation'];fig,axes=plt.subplots(1,2,figsize=(13,5))
    for row in formal.itertuples():axes[0].scatter(row.max_force_rate_N_s,row.yaw_IAE_deg_s,label=f'{row.task}/{row.variant}');axes[1].scatter(row.max_xy_error_mm,row.yaw_peak_deg)
    axes[0].set_xlabel('Peak actual total force slew (N/s)');axes[0].set_ylabel('Yaw IAE (deg s)');axes[1].set_xlabel('Non-target XY peak (mm)');axes[1].set_ylabel('Yaw peak (deg)');axes[0].legend(fontsize=6,ncol=3,bbox_to_anchor=(.5,-.2),loc='upper center');save(fig,'phase2b3_cost_coupling.png')
    history=pd.read_csv(ROOT/'results/phase2b3_history_observability.csv');q=history[(history.stage=='frozen_evaluation')&(history.variant.isin(['C3','C4-Observer']))];fig,ax=plt.subplots(figsize=(10,5));ax.scatter(q.gain_Mz_mean_Nm,q.observer_Mz_mean_Nm);lim=max(abs(q.gain_Mz_mean_Nm).max(),.01)*1.1;ax.plot([-lim,lim],[-lim,lim],'k--');ax.set_xlabel('Offline actual gain Mz mean (Nm)');ax.set_ylabel('Causal frozen observer Mz mean (Nm)');ax.set_title('Available causal history identifies aggregate yaw bias, not all six gains');ax.grid(alpha=.2);save(fig,'phase2b3_history_observability.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--continuation-root',type=Path);a=p.parse_args();analyze(a.run_root,a.continuation_root)
