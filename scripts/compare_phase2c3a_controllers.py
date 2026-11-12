"""Recompute full-rate metrics, export sampled combined CSV/plots; no new simulation."""
import _bootstrap,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json,sha
from src.disturbance.metrics import vec
from src.workspace.reference import LargeReference
from src.workspace.geometry import ExpandedGeometryQuery
from src.phase1.plant import create_plant
from analyze_phase2c3a_workspace import RUN
from run_phase2c3a_large_motion import metrics

def main():
    plot=ROOT/'assets/plots';plot.mkdir(exist_ok=True);rows=[];samples=[];checks=[];data={}
    for folder in sorted((RUN/'dynamics').iterdir()):
        if not (folder/'trial_result.json').exists():continue
        saved=json.loads((folder/'trial_result.json').read_text(encoding='utf-8'));reference=LargeReference(create_plant(),saved['amplitude_m'],saved['rise_s'],RUN/'references',ExpandedGeometryQuery());r=metrics(folder,reference);r.update(method=saved['method'],tau_s=saved['tau_s'],external_Fz_N=saved['external_Fz_N']);rows.append(r);d=pd.read_csv(folder/'physics.csv.gz');data[folder.name]=d
        checks.append(dict(trial=folder.name,maximum_saved_numeric_difference=max(abs(float(r[k])-float(saved[k])) for k in r if isinstance(r[k],(float,int)) and k in saved),physical_step_time_error_s=float(np.max(abs(d.time_s-np.arange(1,len(d)+1)*.001)))))
        indices=np.unique(np.r_[np.arange(0,len(d),20),len(d)-1]);selected=d.iloc[indices].copy();names=['time_s','failure','actual_z_from_home_m','reference_z_from_home_m','reference_z_acceleration_m_s2','expanded_mesh_gap_m','proxy_clearance_m','actual_jacobian_condition','weld_translation_m','weld_angle_rad','diag_equivalent_disturbance_2','native_force_saturated']
        for prefix,n in [('p',3),('p_ref',3),('rpy',3),('rpy_ref',3),('q',6),('q_ref',6),('dq',6),('F_command',6),('force',6),('F_add_total',6)]:names += [f'{prefix}_{i}' for i in range(n)]
        selected=selected[names];selected['trial']=folder.name;selected['aggregate_sample_period_s']=.020;samples.append(selected)
    out=pd.DataFrame(rows);out.to_csv(ROOT/'results/phase2c3a_controller_comparison.csv',index=False);pd.concat(samples).to_csv(ROOT/'results/phase2c3a_dynamic_tracking.csv',index=False);write_json(RUN/'independent_recomputation.json',dict(passed=all(x['maximum_saved_numeric_difference']<1e-8 and x['physical_step_time_error_s']<1e-8 for x in checks),trials=checks))
    w=pd.read_csv(ROOT/'results/phase2c3a_workspace_scan.csv');fig,ax=plt.subplots(2,2,figsize=(13,9));ax[0,0].plot(w.delta_z_m*1000,w.simulation_geometric_candidate.astype(int));ax[0,0].set_ylabel('Modeled candidate (not safety certificate)')
    for j in range(1,7):ax[0,1].plot(w.delta_z_m*1000,w[f'slide_{j}_m']*1000,label=f'Leg{j}')
    ax[0,1].set_ylabel('Relative slide stroke (mm)');ax[0,1].legend(ncol=3);ax[1,0].plot(w.delta_z_m*1000,w.jacobian_condition);ax[1,0].set_ylabel('Normalised Jacobian condition')
    for k in ['slide_margin_m','expanded_mesh_gap_m','legacy_proxy_gap_m']:ax[1,1].plot(w.delta_z_m*1000,w[k]*1000,label=k)
    if 'platform_gap_query_2m_m' in w:ax[1,1].plot(w.delta_z_m*1000,w.platform_gap_query_2m_m*1000,label='platform gap (2m query)')
    ax[1,1].set_ylabel('Margins / query distances (mm)');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.set_xlabel('World Z from Home (mm)');a.grid(alpha=.2)
    fig.suptitle('OFFLINE full-chain workspace scan; passive hard stops unknown');fig.tight_layout();fig.savefig(plot/'phase2c3a_workspace.png',dpi=160);plt.close(fig)
    colors=dict(C0='#777777',C4='#d48520',fourDlead='#188b83')
    for amp in [.03,.05]:
        fig,ax=plt.subplots(3,2,figsize=(14,10));subset=out[(out.amplitude_m==amp)&(out.rise_s==10)&(out.tau_s==0)&(out.external_Fz_N==0)]
        for r in subset.itertuples():
            d=data[Path(r.folder).name];e=(vec(d,'p',3)-vec(d,'p_ref',3))*1000;c=colors[r.method]
            ax[0,0].plot(d.time_s,d.actual_z_from_home_m*1000,color=c,label=r.method);ax[0,0].plot(d.time_s,d.reference_z_from_home_m*1000,'k--',alpha=.3);ax[0,1].plot(d.time_s,e[:,2],color=c,label=r.method)
            ax[1,0].plot(d.time_s,np.linalg.norm(e[:,:2],axis=1),color=c,label=r.method)
            for i,n in enumerate(['R','P','Y']):ax[1,1].plot(d.time_s,d[f'rpy_{i}'],color=c,linestyle=['-','--',':'][i],label=r.method+' '+n)
            ax[2,0].plot(d.time_s,d.diag_equivalent_disturbance_2,color=c,label=r.method);ax[2,1].plot(d.time_s,d.actual_jacobian_condition,color=c,label=r.method)
        for a,title in zip(ax.flat,['Actual / reference Z (mm)','Z tracking error (mm)','Non-target XY error (mm)','RPY drift (deg)','Causal equivalent Fz estimate (N)','Actual Jacobian condition']):a.set_title(title);a.grid(alpha=.2);a.legend(fontsize=7,ncol=2);a.set_xlabel('Physical time (s)')
        fig.suptitle(f'L{round(amp*1000)} REAL DYNAMIC EXECUTION, slow rise10s');fig.tight_layout();fig.savefig(plot/f'phase2c3a_L{round(amp*1000)}_tracking.png',dpi=160);plt.close(fig)
    selected=out[(out.amplitude_m==.05)&(out.rise_s==10)&(out.method=='fourDlead')&(out.tau_s==0)]
    if len(selected):
        d=data[Path(selected.iloc[0].folder).name];fig,ax=plt.subplots(3,2,figsize=(14,10))
        for i,a in enumerate(ax.flat):
            a.plot(d.time_s,d[f'q_{i}']*1000,label='Actual relative slide stroke');a.plot(d.time_s,d[f'q_ref_{i}']*1000,'--',label='Reference');twin=a.twinx();twin.plot(d.time_s,d[f'force_{i}'],color='gray',alpha=.6,label='Actual force');a.set_title(f'Leg{i+1}');a.set_ylabel('Relative slide stroke (mm)');twin.set_ylabel('Actual force (N)');a.legend(fontsize=7);a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(plot/'phase2c3a_six_leg_stroke_force.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(13,8))
        for i in range(6):ax[0,0].plot(d.time_s,d[f'force_{i}'],label=f'Leg{i+1}');ax[0,1].plot(d.time_s.iloc[1:],np.diff(d[f'force_{i}'])/.001,label=f'Leg{i+1}')
        ax[1,0].plot(d.time_s,d.weld_translation_m*1e6,label='translation um');ax[1,0].plot(d.time_s,d.weld_angle_rad*1e6,label='rotation urad');ax[1,0].axhline(100,color='red',linestyle='--',label='Original limits')
        ax[1,1].plot(d.time_s,d.expanded_mesh_gap_m*1000,label='Expanded mesh gap');ax[1,1].plot(d.time_s,d.proxy_clearance_m*1000,label='Legacy proxy');ax[1,1].axhline(5,color='red',linestyle='--')
        for a,title in zip(ax.flat,['Actual six-leg forces (N)','Actual force rates (N/s)','Frozen weld thresholds','Measured query clearances (mm)']):a.set_title(title);a.grid(alpha=.2);a.legend(fontsize=7)
        fig.tight_layout();fig.savefig(plot/'phase2c3a_force_constraints.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for method in ['C0','C4','fourDlead']:
        s=out[(out.amplitude_m==.05)&(out.tau_s==0)&(out.external_Fz_N==0)&(out.method==method)].sort_values('rise_s');ax[0].plot(1.875*.05/s.rise_s*1000,s.z_rmse_mm,'o-',label=method);ax[1].plot(1.875*.05/s.rise_s*1000,s.xy_peak_mm,'o-',label=method)
    for a,title in zip(ax,['Z RMSE (mm)','XY peak error (mm)']):a.set_ylabel(title);a.set_xlabel('Peak reference Z speed (mm/s)');a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(plot/'phase2c3a_speed_comparison.png',dpi=160);plt.close(fig)
    if len(selected):
        d=data[Path(selected.iloc[0].folder).name];fig,ax=plt.subplots(figsize=(12,4));ax.plot(d.time_s,d.actual_z_from_home_m*1000,label='Actual Z');ax.plot(d.time_s,d.reference_z_from_home_m*1000,'--',label='Reference');ax.axhline(5,color='gray',linestyle=':',label='Old +5mm (historical only)');ax.axhline(52,color='red',linestyle='--',label='Independent +52mm protection');ax.axhline(-2,color='red',linestyle='--',label='Independent -2mm protection');ax.set_ylabel('World Z relative to Home (mm)');ax.set_xlabel('Physical time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plot/'phase2c3a_workspace_monitor.png',dpi=160);plt.close(fig)
    lm=pd.read_csv(ROOT/'results/phase2c3a_local_model_audit.csv');fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].plot(lm.delta_z_m*1000,lm.static_z_model_error_N);ax[0].set_ylabel('Static local-model Z bias error (N)');ax[1].plot(lm.delta_z_m*1000,lm.nominal_reduced_Mzz_kg);ax[1].set_ylabel('Constrained nominal Mzz (kg)')
    for a in ax:a.set_xlabel('Z from Home (mm)');a.grid(alpha=.2)
    fig.suptitle('OFFLINE nominal model audit; not observed external force');fig.tight_layout();fig.savefig(plot/'phase2c3a_local_model.png',dpi=160);plt.close(fig)
    load=out[(out.tau_s>0)&(out.external_Fz_N!=0)]
    if len(load):
        fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
        for r in load.itertuples():
            d=data[Path(r.folder).name];ax[0].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,label=r.method);ax[1].plot(d.time_s,d.diag_equivalent_disturbance_2,label=r.method)
        ax[0].set_ylabel('Z error (mm)');ax[1].set_ylabel('Estimated equivalent Fz (N)');ax[1].set_xlabel('Actual physical time (s)')
        for a in ax:a.grid(alpha=.2);a.legend()
        fig.tight_layout();fig.savefig(plot/'phase2c3a_load_response.png',dpi=160);plt.close(fig)
    print(out[['method','amplitude_m','rise_s','tau_s','external_Fz_N','success','z_rmse_mm','xy_peak_mm']].to_string(index=False))
if __name__=='__main__':main()
