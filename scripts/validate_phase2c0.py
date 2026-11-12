"""Independent raw-archive recomputation and physical/causal audit, no integration."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,sha,write_json,load_model
from src.disturbance.metrics import vec,error_series,recovery
from src.control.phase2b2_experiment import actual_force_map

def analyze(root):
    root=Path(root);plan=json.loads((root/'protocol_before_experiments.json').read_text(encoding='utf-8'));rows=[];proof=[];m=load_model();scratch=mujoco.MjData(m);noise={}
    for folder in sorted(root.glob('*/*/*')):
        if not (folder/'metrics.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));r=json.loads((folder/'metrics.json').read_text(encoding='utf-8'));dyn=json.loads((folder/'actuator_dynamics.json').read_text(encoding='utf-8'));plant=json.loads((folder/'plant.json').read_text(encoding='utf-8'));d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');n=len(d)
        p,a,rot=error_series(d);e=(vec(d,'p',3)-vec(d,'p_ref',3))*1000;t=d.time_s.to_numpy();cmd=vec(d,'F_command',6);drive=vec(d,'F_drive',6);prev=vec(d,'F_drive_previous',6);force=vec(d,'force',6);J=vec(d,'true_kinematic_map',36).reshape(-1,6,6);gain=np.array(plant['gain']);active=d.budget_applicable.to_numpy(bool);add=vec(d,'F_add_total',6);tau=np.broadcast_to(np.asarray(dyn['tau_s'],float),(6,));alpha=np.ones(6);positive=tau>0;alpha[positive]=-np.expm1(-.001/tau[positive]);pr=cfg['disturbance'];on=pr['start_step']*.001;off=pr['stop_step']*.001;load=(d.input_step>=pr['start_step'])&(d.input_step<pr['stop_step']);post=d.input_step>=pr['stop_step']
        expected=cmd.copy() if not dyn['enabled'] or not tau.any() else prev+alpha*(np.clip(cmd,-400,400)-prev)
        np.testing.assert_allclose(drive,expected,atol=1e-12,rtol=0);np.testing.assert_array_equal(prev[1:],drive[:-1]);np.testing.assert_array_equal(prev[0],dyn['initialization']['force_N']);np.testing.assert_array_equal(d.actuator_update_count,np.arange(1,n+1));np.testing.assert_allclose(force,np.clip(gain*drive,-400,400),atol=1e-10)
        np.testing.assert_array_equal(vec(d,'ctrl',12)[:,:6],drive);np.testing.assert_array_equal(vec(d,'ctrl',12)[:,6:],vec(d,'external_ctrl',6));np.testing.assert_array_equal(vec(d,'f_cmd',6),cmd)
        assert max(abs(cmd).max(),abs(drive).max(),abs(force).max())<=400+1e-9
        if active.any():
            np.testing.assert_allclose(cmd[active]-vec(d,'F_C0',6)[active],add[active],atol=1e-12,rtol=0);assert abs(add[active]).max()<=5+1e-9 and abs(np.diff(np.r_[np.zeros((1,6)),add[active]],axis=0)).max()<=.02+1e-9
        memory=d.observer_memory_applicable.to_numpy(bool)
        if memory.any():np.testing.assert_allclose(vec(d,'observer_memory_wrench',6)[memory],vec(d,'known_command_wrench',6)[memory],atol=2e-11,rtol=0)
        for field,shadow in [('lagged_residual_wrench','shadow_base_drive'),('lagged_additional_wrench','shadow_c0_drive')]:np.testing.assert_allclose(np.einsum('nij,ni->nj',J,force-np.clip(gain*vec(d,shadow,6),-400,400)),vec(d,field,6),atol=1e-10)
        read=d.sensor_read_step.to_numpy();capture=d.sensor_capture_step.to_numpy();valid=read>=0;delay=cfg.get('measurement',{}).get('delay_steps',0);assert np.all(capture[valid]<=read[valid]);np.testing.assert_array_equal(capture[valid],np.maximum(0,read[valid]-delay))
        measurement=cfg.get('measurement',{})
        if any(measurement.get(k,0) for k in ['position_std_m','rotation_std_rad','velocity_std']):
            key=json.dumps(measurement,sort_keys=True);z=vec(d,'sensor_noise',12)[valid]
            if key in noise:np.testing.assert_array_equal(z[:min(len(z),len(noise[key]))],noise[key][:min(len(z),len(noise[key]))])
            else:noise[key]=z
        assert not (dyn['enabled'] and measurement), 'No lag/sensor mixtures allowed'
        with np.load(folder/'integration_states.npz') as z:
            states=z['states'];assert states.shape==(n,458) and np.isfinite(states).all();np.testing.assert_array_equal(states[:,0],t);np.testing.assert_array_equal(states[:,1:44],vec(d,'qpos',43));np.testing.assert_array_equal(states[:,44:86],vec(d,'qvel',42));initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'));map_error=0.
            for i in np.linspace(0,n-1,3,dtype=int):
                scratch.qpos[:]=initial['qpos'] if i==0 else states[i-1,1:44];scratch.qvel[:]=initial['qvel'] if i==0 else states[i-1,44:86];mujoco.mj_forward(m,scratch);map_error=max(map_error,float(abs(actual_force_map(m,scratch)-J[i]).max()))
            assert map_error<2e-8
        np.testing.assert_allclose(d.position_error_mm,p,atol=1e-10)
        for prefix,mask in [('load',load),('unload',post)]:
            primary=float(p[mask].sum()*.001);assert abs(primary-r[prefix+'_position_iae_mm_s'])<1e-10
        assert abs(float(np.sqrt(np.mean(p*p)))-r['position_rmse_mm'])<1e-10
        # Aggregate lag indicator from allowed command/response histories; actual
        # drive is an OFFLINE diagnostic, never passed to online causal methods.
        command_rate=np.diff(cmd,axis=0);drive_rate=np.diff(drive,axis=0);corr=[]
        for shift in range(21):
            x=command_rate if shift==0 else command_rate[:-shift];y=drive_rate if shift==0 else drive_rate[shift:];den=np.linalg.norm(x)*np.linalg.norm(y);corr.append(float(np.sum(x*y)/den) if den>1e-12 else 0.)
        excitation=command_rate/.001;sv=np.linalg.svd(excitation-excitation.mean(axis=0),compute_uv=False);energy=float(np.sqrt(np.mean(excitation**2)))
        peak=np.flatnonzero(load)[p[load].argmax()] if load.any() else int(p.argmax())
        r.update(variant=folder.name,folder=str(folder),method=cfg['method'],tau_s=json.dumps(dyn['tau_s']),dynamics_enabled=dyn['enabled'],measurement_json=json.dumps(measurement,sort_keys=True),diagnostic_kind=(cfg.get('diagnostic') or {}).get('kind','none'),privileged=cfg['stage']=='privileged_diagnostic',maximum_command_drive_difference_N=float(abs(cmd-drive).max()),maximum_command_actual_difference_N=float(abs(cmd-force).max()),maximum_drive_N=float(abs(drive).max()),shared_additional_peak_N=float(abs(add[active]).max()) if active.any() else None,shared_additional_rate_N_s=float(abs(np.diff(np.r_[np.zeros((1,6)),add[active]],axis=0)/.001).max()) if active.any() else None,shared_budget_utilization=float(abs(add[active]).max()/5) if active.any() else None,command_slew_N_s=float(abs(np.diff(cmd,axis=0)/.001).max()),drive_slew_N_s=float(abs(np.diff(drive,axis=0)/.001).max()),projected_steps=int(d.action_rejection_reason.eq('projected_shared_budget').sum()),yaw_peak_deg=float(abs(np.rad2deg(rot[:,2])).max()),peak_vector_time_s=float(t[peak]),predeclared_primary_load_IAE_mm_s=float(p[load].sum()*.001),xy_peak_mm=float(np.linalg.norm(e[:,:2],axis=1).max()),offline_command_drive_best_correlation_delay_ms=int(np.argmax(corr)),command_excitation_RMS_N_s=energy,command_excitation_rank=int(np.sum(sv>sv[0]*1e-4)) if energy>1e-4 and sv[0] else 0,nominal_memory_hidden_drive_difference_peak=float(abs(vec(d,'known_command_wrench',6)[memory]-vec(d,'nominal_drive_wrench',6)[memory]).max()) if memory.any() else None)
        for j,axis in enumerate('xyz'):r[f'{axis}_rmse_mm']=float(np.sqrt(np.mean(e[:,j]**2)));r[f'{axis}_peak_mm']=float(abs(e[:,j]).max());r[f'{axis}_at_norm_peak_mm']=float(e[peak,j])
        r.pop('measurement',None);r.pop('tau',None);r.pop('diagnostic',None);rows.append(r);proof.append(dict(folder=str(folder),physical_steps=n,zoh_error_max_N=float(abs(drive-expected).max()),state_csv_exact=True,known_command_memory_checked=bool(memory.any()),map_error=map_error,causal_sensor=True,limits_valid=True));write_json(folder/'metrics_recomputed.json',r)
        if len(rows)%10==0:print('RAW VALIDATED',len(rows),flush=True)
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/phase2c0_all_trials.csv',index=False);f[f.stage.isin(['lag_development','differential_lag','delay_development','noise_development','lag_safety'])].to_csv(ROOT/'results/phase2c0_c4_robustness.csv',index=False);f[f.stage.isin(['authority_diagnostic','frozen_evaluation','privileged_diagnostic'])].to_csv(ROOT/'results/phase2c0_action_feasibility.csv',index=False)
    hist=json.loads((ROOT/'results/phase2c0_frozen_history.json').read_text(encoding='utf-8'))
    for name,item in hist['files'].items():assert sha(ROOT/name)==item['sha256'],name
    assert len(rows)<=plan['maximum_research_trials'];gui=json.loads((ROOT/'results/phase2c0_observer_validation.json').read_text(encoding='utf-8'));assert gui['bitwise_equal'] and gui['physics_steps_each']==2048 and not gui['interrupted']
    result=dict(research_trials=len(rows),successful_trials=int(f.success.sum()),failed_trials=int((~f.success).sum()),physical_steps=int(f.physical_steps.sum()),all_zoh_signals_verified=True,all_three_force_signals_verified=True,all_states_csv_exact=True,all_native_and_shared_limits_verified=True,all_observer_memory_uses_command=True,sensor_pairing_verified_groups=len(noise),history_files_unchanged=len(hist['files']),gui_bitwise_nonintrusive=True,maximum_independent_scratch_error=max(x['map_error'] for x in proof),no_RL_or_Gym=True,proof=proof)
    write_json(ROOT/'results/phase2c0_validation.json',result);plots(root,f);print(json.dumps({k:v for k,v in result.items() if k!='proof'}),flush=True);return f

def plots(root,f):
    def save(fig,name):fig.tight_layout();fig.savefig(ROOT/'assets/plots'/name,dpi=150);plt.close(fig)
    def get(row):return pd.read_csv(Path(row.folder)/'physics.csv.gz')
    q=f[(f.stage=='lag_development')&(f.task=='zload')&(f.method=='C4')];fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for r in q.itertuples():
        d=get(r);axes[0].plot(d.time_s,d.position_error_mm,label=r.variant);axes[1].plot(d.time_s,d.orientation_error_deg);axes[2].plot(d.time_s,np.max(abs(vec(d,'F_command',6)-vec(d,'F_drive',6)),axis=1))
    for ax,title in zip(axes,['C4 true position error (mm), uniform synthetic lag','SO3 error (deg)','Max |bounded command - hidden drive| (N)']):ax.set_title(title);ax.grid(alpha=.2);ax.axvline(1.7,color='k',ls=':');ax.axvline(3.45,color='k',ls=':')
    axes[0].legend();save(fig,'phase2c0_lag_response.png')
    q=f[(f.stage=='delay_development')&(f.method=='C4')];fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for r in q.itertuples():
        d=get(r);axes[0].plot(d.time_s,d.position_error_mm,label=r.variant);axes[1].plot(d.time_s,d.orientation_error_deg)
    axes[0].legend();axes[0].set_ylabel('Position mm');axes[1].set_ylabel('SO3 deg');axes[1].set_xlabel('Physical time (s)');save(fig,'phase2c0_delay_response.png')
    lag=pd.read_csv(ROOT/'results/phase2c0_actuator_lag_validation.csv');fig,ax=plt.subplots(figsize=(10,5))
    for tau,g in lag.groupby('tau_s'):ax.plot(g.time_s,g.drive_N,label=f'tau{tau}s')
    g=lag[lag.tau_s==0];ax.plot(g.time_s,g.command_N,'k:',label='bounded input');ax.set_ylabel('Force N, support85N plus +/-0.1N step');ax.set_xlabel('Time s');ax.legend();ax.grid(alpha=.2);save(fig,'phase2c0_zoh_validation.png')
    for task,g in f[f.stage=='frozen_evaluation'].groupby('task'):
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
        for r in g.itertuples():
            d=get(r);axes[0].plot(d.time_s,d.position_error_mm,label=r.variant);axes[1].plot(d.time_s,np.max(abs(vec(d,'F_add_total',6)),axis=1));axes[2].plot(d.time_s,np.max(abs(vec(d,'F_command',6)-vec(d,'F_drive',6)),axis=1))
        axes[0].legend();axes[0].set_ylabel('Position mm');axes[1].set_ylabel('Shared additional N');axes[2].set_ylabel('|command-drive| N');axes[2].set_xlabel('Physical time s');save(fig,f'phase2c0_{task}_comparison.png')
    q=f[f.stage=='frozen_evaluation'];fig,ax=plt.subplots(figsize=(11,5))
    for r in q.itertuples():ax.scatter(r.max_force_rate_N_s,r.load_position_iae_mm_s,label=f'{r.task}/{r.variant}')
    ax.set_xlabel('Actual force peak slew (N/s)');ax.set_ylabel('Load position IAE (mm s)');ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'phase2c0_candidate_cost.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);analyze(p.parse_args().run_root)
