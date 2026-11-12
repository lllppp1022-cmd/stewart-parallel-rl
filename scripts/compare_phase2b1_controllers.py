"""Independent post-freeze comparison, phase-separated metrics, causal plots. No tuning."""
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

METHODS=['C0','C1','C2','C1-L','C3'];COLORS={'C0':'#2563eb','C1':'#f59e0b','C2':'#6b7280','C1-L':'#16a34a','C3':'#dc2626'}
def read(root,task,method):return pd.read_csv(Path(root)/'final'/task/method/'physics.csv.gz')
def marks(ax):ax.axvline(2,color='black',ls=':',lw=.8);ax.axvline(4,color='black',ls=':',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Physical time (s)')
def save(fig,name):fig.tight_layout();fig.savefig(ROOT/'assets/plots'/name,dpi=150);plt.close(fig)

def analyze(root):
    root=Path(root);p=json.loads((root/'protocol_before_development.json').read_text(encoding='utf-8'));rows=[]
    for stage in ('test_safety_precheck','development','equivalence','final'):
        for folder in sorted((root/stage).glob('*/*')):
            if not (folder/'summary.json').exists():continue
            row=trial_metrics(folder,p);cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));row['variant']=folder.name;row['final_protocol_sha256']=cfg.get('final_protocol_sha256');write_json(folder/'metrics_recomputed.json',row);rows.append(row)
    allrows=pd.DataFrame(rows);allrows.to_csv(ROOT/'results/phase2b1_all_trials.csv',index=False);final=allrows[allrows.stage=='final'].copy();final.to_csv(ROOT/'results/phase2b1_controller_comparison.csv',index=False);final.to_csv(ROOT/'results/phase2b1_recovery_metrics.csv',index=False);allrows[allrows.stage=='development'].to_csv(ROOT/'results/phase2b1_development_comparison.csv',index=False)
    if len(final):plots(root,final)
    online=[]
    for row in final[final.controller=='C3'].itertuples():
        df=pd.read_csv(Path(row.folder)/'physics.csv.gz');est=vec(df,'diag_equivalent_disturbance',6);truth=vec(df,'external_world',6);cfg=json.loads((Path(row.folder)/'config.json').read_text(encoding='utf-8'));on=cfg['disturbance']['start_step'];off=cfg['disturbance']['stop_step'];rmse=np.sqrt(np.mean((est-truth)**2,axis=0));steady=np.mean(est[off-400:off-50]-truth[off-400:off-50],axis=0)
        online.append(dict(task=row.task,split=row.split,force_rmse_N=float(rmse[2]),roll_rmse_Nm=float(rmse[3]),pitch_rmse_Nm=float(rmse[4]),force_steady_error_N=float(steady[2]),roll_steady_error_Nm=float(steady[3]),pitch_steady_error_Nm=float(steady[4]),estimates_are_local_equivalent_wrench=True,truth_offline_only=True))
    pd.DataFrame(online).to_csv(ROOT/'results/phase2b1_online_observer_validation.csv',index=False)
    print(final[['task','controller','success','load_peak_position_mm','load_recovery_status','unload_peak_position_mm','unload_recovery_status']].to_string(index=False),flush=True);return allrows

def plots(root,table):
    methods=[m for m in METHODS if m in set(table.controller)];fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for method in methods:
        d=read(root,'benchmark_force10',method);axes[0].plot(d.input_step*.001,d.external_world_2,color=COLORS[method],label=method);axes[1].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,color=COLORS[method],label=method)
    axes[0].set_ylabel('World Fz (N)');axes[1].set_ylabel('Z actual - reference (mm)');axes[1].axhspan(-.25,.25,color='green',alpha=.1,label='0.25mm band (SO3 gate also required)');axes[1].axhline(0,color='black',ls='--',label='reference');[marks(ax) for ax in axes];axes[1].legend(ncol=3,fontsize=8);save(fig,'phase2b1_home_force_response.png')
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for method in methods:
        d=read(root,'benchmark_rollp05',method);axes[0].plot(d.input_step*.001,d.external_world_3,color=COLORS[method]);axes[1].plot(d.time_s,d.rpy_0-d.rpy_ref_0,color=COLORS[method],label=method);axes[2].plot(d.time_s,np.linalg.norm(vec(d,'p',3)[:,:2]-vec(d,'p_ref',3)[:,:2],axis=1)*1000,color=COLORS[method],label=method)
    axes[0].set_ylabel('World Mx (N m)');axes[1].set_ylabel('Roll error (deg)');axes[1].axhspan(-.02,.02,alpha=.1,color='green');axes[2].set_ylabel('Non-target XY error (mm)');[marks(ax) for ax in axes];axes[1].legend(ncol=5);save(fig,'phase2b1_roll_torque_response.png')
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for method,style in [('C1','-'),('C1-L','--')]:
        d=read(root,'benchmark_force10',method)
        for j,color in enumerate(LEG_COLORS):axes[0].plot(d.time_s,d[f'integral_{j}']*3000,color=color,ls=style,label=f'Leg{j+1} {method}')
        axes[2].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,label=method,color=COLORS[method])
    d=read(root,'benchmark_force10','C1-L')
    for j,color in enumerate(LEG_COLORS):axes[1].plot(d.time_s,d[f'diag_leak_force_delta_N_{j}'],color=color,label=f'Leg{j+1}')
    axes[0].set_ylabel('Ki * integral (N)');axes[1].set_ylabel('Leak contribution / step (N)');axes[2].set_ylabel('Z error (mm)');axes[0].legend(fontsize=7,ncol=4);axes[2].legend();[marks(ax) for ax in axes];save(fig,'phase2b1_integral_leakage.png')
    if 'C3' in methods:
        d=read(root,'benchmark_force10','C3');e=read(root,'benchmark_rollp05','C3');fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for ax,df,j,title in [(axes[0],d,2,'Fz (N)'),(axes[1],e,3,'Mx (N m)'),(axes[2],e,4,'My / non-target (N m)')]:
            ax.plot(df.input_step*.001,df[f'external_world_{j}'],'k--',label='Applied external truth (offline only)');ax.plot(df.input_step*.001,df[f'diag_equivalent_disturbance_{j}'],color='#dc2626',label='Local equivalent estimate');ax.set_ylabel(title);marks(ax);ax.legend(fontsize=8)
        axes[0].set_title('Equivalent model disturbance, NOT a true-force sensor');save(fig,'phase2b1_disturbance_estimation.png')
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        for j,color in enumerate(LEG_COLORS):
            axes[0].plot(d.time_s,d[f'force_{j}'],color=color,label=f'Leg{j+1} actual');axes[0].plot(d.time_s,d[f'diag_base_force_N_{j}'],color=color,ls='--',lw=.8);axes[1].plot(d.time_s,d[f'diag_compensation_N_{j}'],color=color,label=f'Leg{j+1}')
        axes[0].set_ylabel('Actual / C0 base force (N)');axes[1].set_ylabel('Applied extra force (N)');axes[0].set_title('Dashed = frozen C0 evaluated at C3 feedback; native limits +/-400 N');axes[0].legend(ncol=3,fontsize=8);axes[1].legend(ncol=3,fontsize=8);[marks(ax) for ax in axes];save(fig,'phase2b1_leg_force_compensation.png')
    f=table[table.task=='benchmark_force10'];fig,axes=plt.subplots(2,3,figsize=(13,8));x=np.arange(len(f))
    for ax,col,title in zip(axes.flat,['load_peak_position_mm','load_position_iae_mm_s','load_recovery_s','unload_peak_position_mm','unload_position_iae_mm_s','unload_recovery_s'],['Load peak position (mm)','Load IAE (mm s)','Load recovery (s)','Unload peak position (mm)','Unload IAE (mm s)','Unload recovery (s)']):
        values=f[col].to_numpy(float);ax.bar(x,values,color=[COLORS[m] for m in f.controller]);ax.set_xticks(x,f.controller);ax.set_title(title);ax.grid(axis='y',alpha=.25)
        for i,v in enumerate(values):
            if not np.isfinite(v):
                status=str(f.iloc[i]['load_recovery_status' if col.startswith('load') else 'unload_recovery_status']);ax.text(i,.03,'N/A: no excursion' if 'NO EXCURSION' in status else 'NOT RECOVERED',rotation=90,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
    save(fig,'phase2b1_load_unload_metrics.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();analyze(a.run_root)
