import _bootstrap
from src.analysis.gain_mismatch_audit import audit
from src.model import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    d=audit();f,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    for name,g in d.groupby('case'):
        axes[0].plot(g.time_s,g.gain_error_Mz_Nm,label=name);axes[1].plot(g.time_s,g.observer_Mz_Nm,label=name);axes[2].plot(g.time_s,g.yaw_rate_rad_s,label=name);axes[3].plot(g.time_s,g.yaw_deg,label=name)
    for ax,title in zip(axes,['Actual gain-error world Mz at upper_center (Nm)','Frozen C3 internal equivalent Mz estimate (Nm)','Actual world angular velocity Z (rad/s)','Yaw angle (deg); historical protection at .384s']):ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.axvline(.384,color='red',ls=':')
    axes[-1].axhline(-.2,color='red',ls='--');axes[-1].set_xlabel('Physical time (s)');f.tight_layout();f.savefig(ROOT/'assets/plots/phase2b3_yaw_failure_chain.png',dpi=150);plt.close(f)
    f,axes=plt.subplots(3,1,figsize=(12,9),sharex=True);g=d[d.case=='alternating_0.005']
    for j in range(6):axes[0].plot(g.time_s,g[f'force_difference_{j}'],label=f'Leg{j+1}');axes[1].plot(g.time_s,g[f'wrench_error_{j}'],label=['Fx N','Fy N','Fz N','Mx Nm','My Nm','Mz Nm'][j])
    axes[2].plot(g.time_s,g.local_required_Mz_Nm,label='Required equivalent local Mz');axes[2].plot(g.time_s,g.gain_error_Mz_Nm,label='Actual known gain difference');axes[2].plot(g.time_s,g.unexplained_local_Mz_Nm,label='Remaining model residual')
    for ax,title in zip(axes,['Actual minus commanded leg force (N), gain mismatch .5%','Actual minus nominal-command platform wrench (stated mixed units)','Local momentum balance / unmodelled remainder (Nm)']):ax.set_title(title);ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel('Physical time (s)');f.tight_layout();f.savefig(ROOT/'assets/plots/phase2b3_gain_force_wrench.png',dpi=150);plt.close(f);print('Historical audit complete; no mj_step')
if __name__=='__main__':main()
