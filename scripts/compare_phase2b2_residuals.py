"""Independent true-state metrics and equal-budget comparisons, never tunes policies."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json
from src.control.phase2b1_experiment import trial_metrics
from src.disturbance.metrics import vec
from src.visualization.enhanced.style import LEG_COLORS

def metrics(folder,p):
    r=trial_metrics(folder,p);d=pd.read_csv(Path(folder)/'physics.csv.gz');c=json.loads((Path(folder)/'config.json').read_text(encoding='utf-8'));active=d.budget_applicable.to_numpy(bool);add=vec(d,'F_add_total',6);requested=vec(d,'F_residual_requested',6);applied=vec(d,'F_residual_applied',6);nonzero=np.linalg.norm(requested,axis=1)>1e-10
    r.update(variant=Path(folder).name,action_kind=c['action']['kind'],space=c['action']['space'],frequency=c['action']['frequency'],privileged=c['stage']=='oracle_diagnostic',shared_add_peak_N=float(abs(add[active]).max()) if active.any() else None,shared_add_rate_peak_N_s=float(abs(np.diff(add[active],axis=0)/.001).max()) if active.sum()>1 else None,requested_peak_N=float(abs(requested).max()),applied_residual_peak_N=float(abs(applied).max()),projected_fraction=float(d.action_rejection_reason.eq('projected_shared_budget').mean()),nonzero_request_steps=int(nonzero.sum()),zero_effect_fraction=float((np.linalg.norm(applied[nonzero],axis=1)<1e-10).mean()) if nonzero.any() else None,partial_observation=bool(c.get('measurement')),perturbation=json.dumps(c.get('actual_perturbation',{})),measurement=json.dumps(c.get('measurement',{})),observer_pre_bias_Fz_N=float(d.diag_equivalent_disturbance_2.iloc[1500:1900].mean()) if len(d)>1900 else None,observer_pre_bias_Mz_Nm=float(d.diag_equivalent_disturbance_5.iloc[1500:1900].mean()) if len(d)>1900 else None,actual_command_force_peak_difference_N=float(abs(vec(d,'force',6)-vec(d,'f_cmd',6)).max()))
    decisions=[]
    idx=np.flatnonzero(d.action_decision.to_numpy(bool))
    for z,i in enumerate(idx):
        end=idx[z+1] if z+1<len(idx) else len(d);raw=requested[i];norm=float(np.linalg.norm(raw));dot=applied[i:end]@raw;first=np.flatnonzero(dot>=.1*norm*norm) if norm>1e-10 else []
        decisions.append(dict(folder=str(folder),decision_index=z,input_step=int(d.input_step.iloc[i]),control_time_s=float(d.controller_measurement_time_s.iloc[i]),requested_force_norm_N=norm,applied_last_norm_N=float(np.linalg.norm(applied[end-1])),response_delay_s=float(first[0]*.001) if len(first) else None,status='ZERO_REQUEST' if norm<=1e-10 else 'NO_EFFECT_THIS_DECISION' if not len(first) else 'APPLIED',shared_peak_N=float(abs(add[i:end]).max()),cumulative_add_impulse_norm_N_s=float(np.linalg.norm(add[:end].sum(axis=0)*.001))))
    return r,decisions

def analyze(root):
    root=Path(root);p=json.loads((root/'protocol_before_experiments.json').read_text(encoding='utf-8'));rows=[];decisions=[]
    for stage in ('zero_equivalence','preflight','test_safety_precheck','development','frozen_evaluation','oracle_diagnostic','robustness'):
        for folder in sorted((root/stage).glob('*/*')):
            if (folder/'summary.json').exists():
                r,ds=metrics(folder,p);write_json(folder/'metrics_recomputed.json',r);rows.append(r);decisions+=ds
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/phase2b2_all_trials.csv',index=False);f[f.stage.isin(['preflight','development'])].to_csv(ROOT/'results/phase2b2_action_sensitivity.csv',index=False);f[f.stage=='frozen_evaluation'].to_csv(ROOT/'results/phase2b2_c3_comparison.csv',index=False);f[f.stage=='robustness'].to_csv(ROOT/'results/phase2b2_robustness.csv',index=False);f[f.privileged].to_csv(ROOT/'results/phase2b2_oracle_diagnostics.csv',index=False);pd.DataFrame(decisions).to_csv(ROOT/'results/phase2b2_decision_effectiveness.csv',index=False);plots(root,f)
    print(f[['stage','task','variant','success','failure_reason','load_peak_position_mm','unload_reverse_z_mm','max_force_rate_N_s']].to_string(index=False),flush=True);return f

def read(root,stage,task,variant):return pd.read_csv(Path(root)/stage/task/variant/'physics.csv.gz')
def save(fig,name):fig.tight_layout();fig.savefig(ROOT/'assets/plots'/name,dpi=150);plt.close(fig)
def events(ax,on=2,off=4):ax.axvline(on,color='black',ls=':',lw=.8);ax.axvline(off,color='black',ls=':',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Physical time (s)')

def plots(root,f):
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for j in range(3):
        d=read(root,'preflight','quiet',f'task{j}_sign1');axes[0].plot(d.time_s,d[f'W_requested_{[2,3,4][j]}'],label=['Fz N','Mx Nm','My Nm'][j]);axes[1].plot(d.time_s,d[f'W_physical_residual_{[2,3,4][j]}'],label=['Fz N','Mx Nm','My Nm'][j]);axes[2].plot(d.time_s,d.position_error_mm,label=['Fz','Mx','My'][j])
    for ax,title in zip(axes,['Requested task components (mixed stated units)','Realized physical components (same units)','True 3D position error (mm)']):ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    save(fig,'phase2b2_task_residual.png')
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for j,ax in enumerate(axes.flat):
        for sign,style in [(1,'-'),(-1,'--')]:
            d=read(root,'preflight','quiet',f'leg{j+1}_sign{sign}');ax.plot(d.time_s,d[f'F_residual_applied_{j}'],color=LEG_COLORS[j],ls=style,label=f'Leg{j+1} sign {sign} (N)')
        ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Physical time (s)')
    save(fig,'phase2b2_leg_residual.png')
    task='test_force8_time';fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for method in ('C3','C3_task','C3_leg'):
        d=read(root,'frozen_evaluation',task,method);axes[0].plot(d.time_s,d.position_error_mm,label=method);axes[1].plot(d.time_s,d.orientation_error_deg,label=method);axes[2].plot(d.time_s,abs(vec(d,'F_add_total',6)).max(axis=1),label=method)
    for ax,title in zip(axes,['True position error (mm)','SO3 attitude error (deg)','Shared total extra-force budget used (N), limit=5']):ax.set_title(title);events(ax,1.65,3.55);ax.legend()
    save(fig,'phase2b2_load_recovery.png')
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for method in ('C3','C3_task','C3_leg'):
        d=read(root,'frozen_evaluation',task,method);sel=d.time_s>=3.45;axes[0].plot(d.time_s[sel],(d.p_2-d.p_ref_2)[sel]*1000,label=method);axes[1].plot(d.time_s[sel],np.linalg.norm(vec(d,'F_residual_applied',6),axis=1)[sel],label=method);axes[2].plot(d.time_s[sel],np.max(abs(vec(d,'actual_force_rate_N_s',6)),axis=1)[sel],label=method)
    for ax,title in zip(axes,['Signed Z error / reverse overshoot (mm)','Applied residual relative to frozen C3 (N norm)','Actual total force slew (N/s)']):ax.set_title(title);events(ax,1.65,3.55);ax.legend()
    save(fig,'phase2b2_unload_overshoot.png')
    final=f[f.stage=='frozen_evaluation'];fig,ax=plt.subplots(figsize=(11,6))
    for row in final.itertuples():ax.scatter(row.max_force_rate_N_s,row.load_peak_position_mm,label=f'{row.task}/{row.variant}',marker='o' if row.success else 'x')
    ax.set_xlabel('Peak total actual force rate (N/s)');ax.set_ylabel('Load position peak (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2,bbox_to_anchor=(1,1));save(fig,'phase2b2_control_cost.png')
    r=f[f.stage=='robustness'];fig,axes=plt.subplots(2,1,figsize=(14,8));x=np.arange(len(r));labels=r.variant.to_list();axes[0].bar(x,r.peak_position_mm,color=['#2563eb' if c=='C3' else '#999999' for c in r.controller]);axes[1].bar(x,r.peak_orientation_deg,color=['#dc2626' if not ok else '#16a34a' for ok in r.success]);
    for ax in axes:ax.set_xticks(x,labels,rotation=70,ha='right',fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Single-factor pilot; failed short trajectories NOT RMSE-ranked');axes[0].set_ylabel('Observed peak position (mm)');axes[1].set_ylabel('Observed peak SO3 (deg)')
    for i,ok in enumerate(r.success):
        if not ok:axes[1].text(i,.01,'FAIL / EARLY STOP',rotation=90,ha='center',transform=axes[1].get_xaxis_transform(),fontsize=7)
    save(fig,'phase2b2_robustness.png')
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j in (0,1,5):
        d=read(root,'preflight','quiet',f'leg{j+1}_sign1');axes[0].plot(d.time_s,(d.p_0-d.p_ref_0)*1000,color=LEG_COLORS[j],label=f'Leg{j+1}');axes[1].plot(d.time_s,(d.p_1-d.p_ref_1)*1000,color=LEG_COLORS[j]);axes[2].plot(d.time_s,d.rpy_2-d.rpy_ref_2,color=LEG_COLORS[j])
    for ax,title in zip(axes,['Non-target X error (mm)','Non-target Y error (mm)','Non-target yaw error (deg)']):ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel('Physical time (s)')
    axes[0].legend();save(fig,'phase2b2_non_target_coupling.png')
    eq=pd.read_csv(ROOT/'results/phase2b2_zero_residual_equivalence.csv');fig,ax=plt.subplots(figsize=(8,4));ax.bar(eq.base,eq.maximum_state_difference);ax.set_ylabel('Maximum mjSTATE_INTEGRATION difference');ax.set_title('Bitwise zero residual equivalence; ctrl/force/pose/termination also match');save(fig,'phase2b2_zero_equivalence.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);a=p.parse_args();analyze(a.run_root)
