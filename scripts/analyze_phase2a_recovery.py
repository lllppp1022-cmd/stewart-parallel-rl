"""Recompute Phase2A response metrics independently from raw 1kHz CSV."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json
from src.disturbance.metrics import compute,vec,error_series
from src.visualization.enhanced.style import LEG_COLORS

def analyze(root,output=None):
    root=Path(root);output=Path(output or ROOT/'results');p=json.loads((root/'pilot_protocol_frozen.json').read_text(encoding='utf-8'));rows=[]
    for stage in ('pilot','formal'):
        for folder in sorted((root/stage).glob('*/*')):
            if not (folder/'summary.json').exists():continue
            cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));base=root/'formal'/('zero_'+cfg['trajectory'])/cfg['controller'];base=base if stage=='formal' and base.exists() else None
            result=compute(folder,p,base);write_json(folder/'metrics_recomputed.json',result);rows.append(result)
    pointer=root/'legacy_recheck_pointer.json'
    if pointer.exists():
        extra=Path(json.loads(pointer.read_text(encoding='utf-8'))['root'])
        for folder in sorted((extra/'legacy_recheck').glob('*/*')):
            if not (folder/'summary.json').exists():continue
            cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));result=compute(folder,p,root/'formal/zero_z_roll'/cfg['controller']);write_json(folder/'metrics_recomputed.json',result);rows.append(result)
    df=pd.DataFrame(rows);df['protocol_role']=np.where(df.stage=='pilot','pilot',np.where(df.task=='legacy_local_z_roll','legacy_offset_diagnostic','formal'))
    df.to_csv(output/'phase2a_disturbance_summary.csv',index=False);formal=df[df.protocol_role=='formal'].copy();formal.to_csv(output/'phase2a_controller_comparison.csv',index=False);formal.to_csv(output/'phase2a_recovery_metrics.csv',index=False)
    if len(formal):plots(root,formal,p)
    return df

def data(root,task,method):return pd.read_csv(Path(root)/'formal'/task/method/'physics.csv.gz')
def events(ax):
    ax.axvline(2,color='black',ls=':',lw=.9);ax.axvline(4,color='black',ls=':',lw=.9);ax.grid(alpha=.25);ax.set_xlabel('Physical time (s)')
def save(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'assets/plots'/name,dpi=150);plt.close(fig)
def plots(root,table,p):
    (ROOT/'assets/plots').mkdir(exist_ok=True,parents=True);methods=['C0','C1','C2'];colors=['#2563eb','#16a34a','#dc2626']
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for c,method in zip(colors,methods):
        d=data(root,'A_home_force',method);b=data(root,'zero_home',method);axes[0].plot(d.time_s,d.external_world_2,label=method,color=c);axes[1].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,label=method+' actual-ref',color=c);axes[1].plot(b.time_s,(b.p_2-b.p_ref_2)*1000,ls='--',lw=.7,color=c,label=method+' paired zero')
    axes[0].set_ylabel('Actual world Fz (N)');axes[1].set_ylabel('Z - Z_ref (mm)');axes[1].axhline(0,color='black',label='Z_ref');axes[0].set_title('Step load: same home/reference and native controller');[events(ax) for ax in axes];axes[1].legend(ncol=2,fontsize=8);save(fig,'phase2a_step_force_response.png')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for c,method in zip(colors,methods):
        d=data(root,'B_home_roll_positive',method);axes[0].plot(d.time_s,d.external_world_3,color=c,label=method);axes[1].plot(d.time_s,d.rpy_0-d.rpy_ref_0,color=c,label=method);axes[2].plot(d.time_s,d.rpy_1-d.rpy_ref_1,color=c,label=method)
    for ax in axes[1:]:ax.axhspan(-p['recovery_angle_deg'],p['recovery_angle_deg'],alpha=.12,color='green');ax.legend();ax.axhline(0,color='black',ls='--')
    axes[0].set_ylabel('World torque X (N m)');axes[1].set_ylabel('Roll - ref (deg)');axes[2].set_ylabel('Pitch - ref (deg)');axes[0].set_title('Euler components for display; recovery uses SO(3) magnitude');[events(ax) for ax in axes];save(fig,'phase2a_step_torque_response.png')
    d=data(root,'A_home_force','C1');b=data(root,'zero_home','C1');home=json.loads((Path(root)/'formal/A_home_force/C1/home_baseline.json').read_text(encoding='utf-8'))
    for typ,name,label in [('force','phase2a_leg_forces.png','Native actuator force (N)'),('q','phase2a_leg_strokes.png','Slide delta from fixed Home (mm), NOT full rod length')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for j,c in enumerate(LEG_COLORS):
            a=d[f'{typ}_{j}'].to_numpy();bv=b[f'{typ}_{j}'].to_numpy()
            if typ=='q':a=(a-home['slide_m'][j])*1000;bv=(bv-home['slide_m'][j])*1000
            ax.plot(d.time_s,a,color=c,label=f'Leg{j+1}');ax.plot(b.time_s,bv,color=c,ls='--',lw=.65)
        ax.set_ylabel(label);ax.set_title('C1 / world Fz=-10 N; dashed=paired zero');events(ax);ax.legend(ncol=3);save(fig,name)
    subset=table[table.task.isin(['A_home_force','B_home_roll_positive','C_z_roll_composite'])];fig,axes=plt.subplots(2,2,figsize=(13,8));labels=[f'{r.task}\n{r.controller}' for r in subset.itertuples()];x=np.arange(len(subset))
    for ax,column,title in zip(axes.flat,['peak_position_mm','peak_orientation_deg','load_recovery_s','load_position_iae_mm_s'],['Peak position (mm)','Peak SO(3) attitude (deg)','Load recovery (s); NR=not recovered','Load position IAE (mm s)']):
        values=subset[column].to_numpy(float);ax.bar(x,values,color=[colors[methods.index(c)] for c in subset.controller]);ax.set_xticks(x,labels,rotation=60,ha='right',fontsize=7);ax.set_title(title);ax.grid(axis='y',alpha=.25)
        for i,(v,ok) in enumerate(zip(values,subset.success)):
            if not np.isfinite(v):ax.text(i,.02,'N/A' if 'NO EXCURSION' in subset.iloc[i].load_recovery_status else 'NR',rotation=90,transform=ax.get_xaxis_transform(),ha='center')
            if not ok:ax.text(i,.85,'FAIL',rotation=90,transform=ax.get_xaxis_transform(),ha='center',color='red')
    save(fig,'phase2a_recovery_comparison.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for c,method in zip(colors,methods):
        d=data(root,'C_z_roll_composite',method)
        for ax,key,scale in [(axes[0,0],'weld_translation_m',1e6),(axes[0,1],'weld_angle_rad',1e6),(axes[1,0],'proxy_clearance_m',1000)]:ax.plot(d.time_s,d[key]*scale,label=method,color=c)
        axes[1,1].plot(d.time_s,np.max(abs(vec(d,'force',6)),axis=1),label=method,color=c)
    for ax,title,gate in zip(axes.flat,['Weld translation (um)','Weld angle (urad)','PARTIAL PROXY clearance (mm)','Maximum native leg force (N)'],[100,100,5,400]):ax.set_title(title);ax.axhline(gate,color='black',ls='--');events(ax);ax.legend()
    save(fig,'phase2a_constraints.png')
    # Dynamic-task reference/actual/control overlay, no changed reference trajectories.
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for c,method in zip(colors,methods):
        d=data(root,'C_z_roll_composite',method);axes[0].plot(d.time_s,(d.p_2-d.p_ref_2)*1000,color=c,label=method);axes[1].plot(d.time_s,d.rpy_0,color=c,label=method);axes[2].plot(d.time_s,d.external_world_2,color=c)
    axes[1].plot(d.time_s,d.rpy_ref_0,'k--',label='reference');axes[0].set_ylabel('Z error (mm)');axes[1].set_ylabel('Roll (deg)');axes[2].set_ylabel('World Fz (N)');[events(ax) for ax in axes];axes[0].legend();axes[1].legend();save(fig,'phase2a_dynamic_response.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();df=analyze(a.run_root);print(df[['stage','task','controller','success','peak_position_mm','load_recovery_status']].to_string(index=False))
