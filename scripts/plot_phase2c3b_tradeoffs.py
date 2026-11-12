"""Additional transparent plots; all decision metrics remain full-rate audited."""
import _bootstrap
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT
from src.workspace.feedforward_study import RUN

def main():
    out=ROOT/'assets/plots';d=pd.read_csv(ROOT/'results/phase2c3b_controller_comparison.csv');fig,ax=plt.subplots(2,2,figsize=(13,8))
    for method,c in [('C4','#e09425'),('FF-VA','#24974b')]:
        s=pd.read_csv(RUN/f'formal/{method}_L50_T4_tau0_load0_none/physics.csv.gz').iloc[::10]
        ax[0,0].plot(s.time_s,(s.p_2-s.p_ref_2)*1000,label=method,color=c);ax[0,1].plot(s.time_s,(s.p_0-s.p_ref_0)*1000,label=method+' X',color=c);ax[0,1].plot(s.time_s,(s.p_1-s.p_ref_1)*1000,'--',label=method+' Y',color=c);ax[1,0].plot(s.time_s,s.rpy_0,label=method+' Roll',color=c);ax[1,0].plot(s.time_s,s.rpy_1,'--',label=method+' Pitch',color=c);ax[1,1].plot(s.time_s,s.rpy_2,label=method+' Yaw',color=c)
    for a,title in zip(ax.flat,['Z error (mm)','Non-target X/Y errors (mm)','Roll/Pitch (deg)','Yaw (deg)']):a.set_title(title);a.set_xlabel('Physical time (s)');a.grid(alpha=.2);a.legend(fontsize=8)
    fig.suptitle('T4: Z improves while non-target tracking worsens; no all-axis superiority');fig.tight_layout();fig.savefig(out/'phase2c3b_coupling_tradeoff.png',dpi=160);plt.close(fig)
    s=pd.read_csv(RUN/'load/FF-VA_L50_T6_tau0.005_load-5.5_none/physics.csv.gz').iloc[::10];fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for j in range(6):
        ax[0].plot(s.time_s,s[f'ff_desired_total_add_N_{j}'],':',alpha=.5);ax[0].plot(s.time_s,s[f'ff_actual_total_add_N_{j}'],label=f'Leg{j+1} accepted');ax[1].plot(s.time_s,s[f'ff_total_add_rate_N_s_{j}'],label=f'Leg{j+1}');ax[2].plot(s.time_s,s[f'force_{j}'],label=f'Leg{j+1}')
    for y in [-5,5]:ax[0].axhline(y,color='red',linestyle='--')
    for y in [-20,20]:ax[1].axhline(y,color='red',linestyle='--')
    for a,title in zip(ax,['Requested dotted / accepted total-add force (N)','1ms total-add slew (N/s), displayed at10ms','Actual full actuator force (N), distinct from additional budget']):a.set_ylabel(title);a.legend(ncol=3,fontsize=8);a.grid(alpha=.2)
    ax[2].set_xlabel('Physical time (s)');fig.tight_layout();fig.savefig(out/'phase2c3b_budget_and_force_rate.png',dpi=160);plt.close(fig)
    selected=d[((d.stage=='formal')&(d.rise_s==6))|d.stage.isin(['lag','load'])];fig,ax=plt.subplots(1,3,figsize=(15,5));methods=['C4','fourDlead','FF-VA'];x=np.arange(4);labels=['tau0','tau5ms','tau10ms','tau5ms+load']
    for j,m in enumerate(methods):
        q=selected[selected.method==m];order=[q[(q.stage=='formal')],q[(q.stage=='lag')&(q.tau_s==.005)],q[(q.stage=='lag')&(q.tau_s==.01)],q[q.stage=='load']];z=[r.iloc[0].z_rmse_mm for r in order];peak=[r.iloc[0].peak_z_error_mm for r in order];margin=[r.iloc[0].remaining_shared_force_N for r in order]
        for a,values in zip(ax,[z,peak,margin]):a.bar(x+(j-1)*.24,values,width=.24,label=m)
    for a,title in zip(ax,['Full Z RMSE (mm)','Peak Z error (mm)','Remaining per-leg force headroom (N)']):a.set_xticks(x,labels,rotation=20);a.set_title(title);a.legend(fontsize=8);a.grid(axis='y',alpha=.2)
    fig.suptitle('Future task candidates: unknown lag alone is a small residual; load transient is measurable');fig.tight_layout();fig.savefig(out/'phase2c3b_rl_candidates.png',dpi=160);plt.close(fig)
    p=pd.read_csv(ROOT/'results/phase2c3b_phase_action_effects.csv');fig,ax=plt.subplots(figsize=(12,5));phases=['rise_accel','rise_decel','top_hold','fall_accel','fall_decel'];x=np.arange(5)
    for j,pattern in enumerate(['positive','negative','sine20','sine50']):a=p[p.pattern==pattern].set_index('phase').loc[phases];ax.bar(x+(j-1.5)*.2,a.ratio,width=.2,label=pattern)
    ax.axhline(1,color='black',linestyle='--',label='Zero fixed probe');ax.set_xticks(x,phases);ax.set_ylabel('Phase Z RMSE / zero-probe Z RMSE');ax.set_title('Fixed probes improve some windows and worsen others; all worsen full-trajectory RMSE');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'phase2c3b_action_phase_effects.png',dpi=160);plt.close(fig)
    print('Four supplemental figures generated from archived actual data')

if __name__=='__main__':main()
