"""Independent full1ms recomputation, paired effects, plots and safe RL design evidence."""
import _bootstrap,json,gc
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from src.model import ROOT,write_json
from src.workspace.feedforward_study import RUN
from analyze_phase2c3b_tracking_mechanism import phases

def main():
    rows=[];checks=[];plots={};phase_rows=[];authority=[]
    for stage in ['dev','formal','lag','joint_preflight','load','authority']:
        for r in json.loads((RUN/f'{stage}_results.json').read_text(encoding='utf-8')):
            folder=Path(r['folder']);d=pd.read_csv(folder/'physics.csv.gz');ep=d[[f'p_{j}' for j in range(3)]].to_numpy()-d[[f'p_ref_{j}' for j in range(3)]].to_numpy();F=d[[f'force_{j}' for j in range(6)]].to_numpy();command=d[[f'F_command_{j}' for j in range(6)]].to_numpy();c0=d[[f'F_C0_{j}' for j in range(6)]].to_numpy();add=command-c0;valid=d.failure.isna().to_numpy();rate=np.diff(np.vstack([np.zeros((1,6)),add[valid]]),axis=0)/.001;angles=Rotation.from_euler('xyz',d[[f'rpy_{j}' for j in range(3)]].to_numpy(),degrees=True);target=Rotation.from_euler('xyz',d[[f'rpy_ref_{j}' for j in range(3)]].to_numpy(),degrees=True);so3=np.rad2deg(np.linalg.norm((target*angles.inv()).as_rotvec(),axis=1));zr=float(np.sqrt(np.mean(ep[:,2]**2))*1000);position=float(np.sqrt(np.mean(np.sum(ep*ep,axis=1)))*1000);force_peak=float(abs(F).max());force_rate=float(abs(np.diff(F,axis=0)/.001).max());energy=float((F*F).sum()*.001)
            errors=[abs(zr-r['z_rmse_mm']),abs(position-r['position_rmse_mm']),abs(force_peak-r['max_force_N']),abs(force_rate-r['max_force_rate_N_s']),abs(energy-r['force_energy_N2_s'])];checks.append(dict(stage=stage,trial=folder.name,max_numeric_error=max(errors),physical_steps=len(d),clock_max_error=float(abs(d.time_s.to_numpy()-np.arange(1,len(d)+1)*.001).max()),extra_budget_peak_N=float(abs(add[valid]).max()),extra_budget_rate_peak_N_s=float(abs(rate).max())))
            flat={k:v for k,v in r.items() if not isinstance(v,(dict,list))};flat.update(tau_s=r['spec']['tau'],external_Fz_N=r['spec']['load'],probe_pattern=r['spec']['pattern'],force_energy_N2_s=energy,z_rmse_mm=zr,position_rmse_mm=position,so3_rmse_deg=float(np.sqrt(np.mean(so3**2))),additional_force_peak_N=float(abs(add[valid]).max()),additional_force_rate_peak_N_s=float(abs(rate).max()),remaining_shared_force_N=5-float(abs(add[valid]).max()));rows.append(flat)
            group=phases(d.time_s,r['rise_s'])
            for phase in np.unique(group):
                mask=group==phase;phase_rows.append(dict(stage=stage,trial=folder.name,method=r['method'],phase=phase,z_rmse_mm=float(np.sqrt(np.mean(ep[mask,2]**2))*1000),z_mean_mm=float(ep[mask,2].mean()*1000),samples=int(mask.sum())))
            if stage=='authority':authority.append(dict(pattern=r['spec']['pattern'],hz=r['spec']['hz'],requested_peak_N=float(abs(d.ff_probe_wrench_2).max()),effective_peak_leg_N=float(abs(d[[f'ff_probe_effect_N_{j}' for j in range(6)]]).to_numpy().max()),zero_effect_fraction=float(np.mean(np.all(abs(d[[f'ff_probe_effect_N_{j}' for j in range(6)]].to_numpy())<1e-9,axis=1))),z_rmse_mm=zr,success=r['success']))
            if stage in ['formal','lag','load','authority']:plots[(stage,folder.name)]=d.iloc[np.unique(np.r_[np.arange(0,len(d),10),len(d)-1])].copy()
            del d;gc.collect()
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/phase2c3b_controller_comparison.csv',index=False);pd.DataFrame(phase_rows).to_csv(ROOT/'results/phase2c3b_control_phase_errors.csv',index=False);pd.DataFrame(authority).to_csv(ROOT/'results/phase2c3b_action_authority.csv',index=False)
    valid=all(x['max_numeric_error']<1e-8 and x['clock_max_error']<1e-8 and x['extra_budget_peak_N']<=5+1e-8 and x['extra_budget_rate_peak_N_s']<=20+1e-6 for x in checks);write_json(RUN/'independent_audit.json',dict(passed=valid,trials=checks))
    formal=df[df.stage=='formal'];formal.to_csv(ROOT/'results/phase2c3b_speed_sensitivity.csv',index=False);df[df.stage.isin(['lag','load'])].to_csv(ROOT/'results/phase2c3b_lag_disturbance.csv',index=False)
    effects=[];counterfactuals=[];strong=json.loads((RUN/'methods_frozen.json').read_text(encoding='utf-8'))['strong_method']
    for method in ['C4','fourDlead',strong]:
        base=formal[(formal.method==method)&(formal.rise_s==6)].iloc[0];lag=df[(df.stage=='lag')&(df.method==method)&(df.tau_s==.005)].iloc[0];loaded=df[(df.stage=='load')&(df.method==method)].iloc[0]
        b=pd.read_csv(Path(base.folder)/'physics.csv.gz');l=pd.read_csv(Path(lag.folder)/'physics.csv.gz');x=pd.read_csv(Path(loaded.folder)/'physics.csv.gz');assert np.array_equal(b.time_s,l.time_s) and np.array_equal(b.time_s,x.time_s) and np.array_equal(b.p_ref_2,x.p_ref_2)
        delta_lag=(l.p_2-b.p_2).to_numpy()*1000;delta_load=(x.p_2-l.p_2).to_numpy()*1000;effects.append(dict(method=method,reference_only_rmse_mm=base.z_rmse_mm,extra_5ms_rmse_mm=lag.z_rmse_mm-base.z_rmse_mm,extra_load_given_5ms_rmse_mm=loaded.z_rmse_mm-lag.z_rmse_mm,paired_lag_position_difference_peak_mm=float(abs(delta_lag).max()),paired_load_position_difference_peak_mm=float(abs(delta_load).max()),note='Matched factor intervention incl closed-loop feedback; RMSE differences are not additive error components'))
        counterfactuals.append(pd.DataFrame(dict(method=method,time_s=b.time_s,reference_only_z_error_mm=(b.p_2-b.p_ref_2)*1000,delta_lag_position_mm=delta_lag,delta_load_position_mm=delta_load)))
    pd.DataFrame(effects).to_csv(ROOT/'results/phase2c3b_paired_effects.csv',index=False)
    pd.concat(counterfactuals).to_csv(ROOT/'results/phase2c3b_paired_time_series.csv',index=False)
    plt.rcParams.update({'font.size':10});out=ROOT/'assets/plots';colors=dict(C0='#777777',C4='#e09425',fourDlead='#218e94',**{'FF-V':'#5c8cd1','FF-VA':'#24974b','FF-VA-Lead':'#9e5ba7'})
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    for m in colors:
        s=formal[formal.method==m].sort_values('rise_s');ax[0].plot(s.rise_s,s.z_rmse_mm,'o-',label=m,color=colors[m]);ax[1].plot(s.rise_s,s.xy_peak_mm,'o-',label=m,color=colors[m])
    for a,title in zip(ax,['Full-trajectory Z RMSE (mm)','Non-target XY peak (mm)']):a.set_ylabel(title);a.set_xlabel('One-way time (s); T8 independent timing');a.grid(alpha=.2);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'phase2c3b_speed_comparison.png',dpi=160);plt.close(fig)
    for T in [4,8,10]:
        fig,ax=plt.subplots(3,2,figsize=(14,10))
        for (stage,name),d in plots.items():
            if stage!='formal' or f'_T{T}_' not in name:continue
            m=str(d.study_method.iloc[0]);c=colors[m];ax[0,0].plot(d.time_s,d.actual_z_from_home_m*1000,color=c,label=m);ax[0,0].plot(d.time_s,d.reference_z_from_home_m*1000,'k--',alpha=.2);ax[0,1].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,color=c,label=m);ax[1,0].plot(d.time_s,np.hypot(d.p_0-d.p_ref_0,d.p_1-d.p_ref_1)*1000,color=c,label=m);ax[1,1].plot(d.time_s,d.ff_velocity_force_N,color=c,label=m+' velocity');ax[1,1].plot(d.time_s,d.ff_acceleration_force_N,color=c,linestyle=':',label=m+' acceleration');ax[2,0].plot(d.time_s,d.ff_requested_wrench_2,color=c,label=m+' requested');ax[2,0].plot(d.time_s,d.ff_realized_wrench_2,color=c,linestyle='--',label=m+' total-add cmd');ax[2,1].plot(d.time_s,d.ff_budget_peak_fraction,color=c,label=m)
        for a,title in zip(ax.flat,['Actual/reference Z (mm)','Z error (mm)','XY error (mm)','Separate reference FF terms (N)','FF request / total-add nominal command wrench (N)','Shared +/-5N budget peak fraction']):a.set_title(title);a.grid(alpha=.2);a.legend(fontsize=6,ncol=2);a.set_xlabel('Physical time (s)')
        fig.suptitle(f'REAL dynamic execution L50 T{T}');fig.tight_layout();fig.savefig(out/f'phase2c3b_T{T}_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    for (stage,name),d in plots.items():
        if stage not in ['lag','load'] or not name.startswith(strong+'_'):continue
        label=name.replace(strong+'_L50_T6_','');ax[0,0].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,label=label);ax[0,1].plot(d.time_s,d.diag_equivalent_disturbance_2,label=label);ax[1,0].plot(d.time_s,d.ff_budget_peak_fraction,label=label);ax[1,1].plot(d.time_s,d.F_command_0-d.F_drive_0,label=label)
    for a,title in zip(ax.flat,['Matched lag/load Z error (mm)','C4 equivalent Fz estimate (N)','Shared budget usage','Leg1 command minus lagged drive (N)']):a.set_title(title);a.grid(alpha=.2);a.legend(fontsize=7);a.set_xlabel('Physical time (s)')
    fig.tight_layout();fig.savefig(out/'phase2c3b_lag_load.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for (stage,name),d in plots.items():
        if stage!='load' or not name.startswith(strong+'_'):continue
        ax[0].plot(d.time_s,d.ff_requested_wrench_2,label='FF requested Fz');ax[0].plot(d.time_s,d.ff_realized_wrench_2,label='Accepted total-add command, nominal J');ax[0].plot(d.time_s,d.lagged_additional_wrench_2,'--',label='Physical total-add vs lagged C0 shadow, actual J')
        for j in range(6):ax[1].plot(d.time_s,d[f'F_command_{j}']-d[f'F_drive_{j}'],label=f'Leg{j+1}')
    ax[0].set_ylabel('Fz (N)');ax[1].set_ylabel('Command minus drive (N)');ax[1].set_xlabel('Physical time (s)')
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.suptitle('Known requests, accepted command and lagged physical effect are different signals');fig.tight_layout();fig.savefig(out/'phase2c3b_command_drive_wrench.png',dpi=160);plt.close(fig)
    for (stage,name),d in plots.items():
        if stage=='formal' and name.startswith(strong+'_L50_T4_'):
            fig,ax=plt.subplots(3,2,figsize=(14,9))
            for j,a in enumerate(ax.flat):
                a.plot(d.time_s,d[f'q_{j}']*1000,label='Actual relative slide stroke');a.plot(d.time_s,d[f'q_ref_{j}']*1000,'--',label='Reference');twin=a.twinx();twin.plot(d.time_s,d[f'force_{j}'],alpha=.4,color='gray');a.set_title(f'Leg{j+1}');a.set_ylabel('Relative slide stroke (mm)');twin.set_ylabel('Actual force (N)');a.legend(fontsize=7);a.grid(alpha=.2)
            fig.tight_layout();fig.savefig(out/'phase2c3b_six_legs.png',dpi=160);plt.close(fig)
            fig,ax=plt.subplots(2,2,figsize=(13,8));ax[0,0].plot(d.time_s,d.rpy_0,label='Roll');ax[0,0].plot(d.time_s,d.rpy_1,label='Pitch');ax[0,0].plot(d.time_s,d.rpy_2,label='Yaw');ax[0,1].plot(d.time_s,d.weld_translation_m*1e6,label='Weld um');ax[0,1].plot(d.time_s,d.weld_angle_rad*1e6,label='Weld urad');ax[0,1].axhline(100,color='red',linestyle='--');ax[1,0].plot(d.time_s,d.actual_jacobian_condition,label='Jacobian condition');ax[1,1].plot(d.time_s,d.expanded_mesh_gap_m*1000,label='Expanded hull clearance mm');ax[1,1].plot(d.time_s,d.proxy_clearance_m*1000,label='Legacy proxy mm');ax[1,1].axhline(5,color='red',linestyle='--')
            for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
            fig.tight_layout();fig.savefig(out/'phase2c3b_safety_curves.png',dpi=160);plt.close(fig)
    print('AUDIT',valid,'TRIALS',len(df));print(formal[['method','rise_s','z_rmse_mm','xy_peak_mm','additional_force_peak_N']].to_string(index=False));assert valid
if __name__=='__main__':main()
