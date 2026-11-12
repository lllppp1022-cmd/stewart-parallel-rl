import _bootstrap
from src.analysis.dynamic_error_decomposition import decompose
from src.disturbance.metrics import vec,error_series
from src.model import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__=='__main__':
    f,series=decompose();fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    for method,d in series.items():
        e=(vec(d,'p',3)-vec(d,'p_ref',3))*1000
        for j in range(3):axes[j].plot(d.time_s,e[:,j],label=method)
        axes[3].plot(d.time_s,np.linalg.norm(e,axis=1),label=method)
    for ax,title in zip(axes,['X error (mm)','Y error (mm)','Z error (mm)','Position norm (mm), not independent peak combination']):ax.set_title(title);ax.axvline(1.6,color='k',ls=':');ax.axvline(3.65,color='k',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[3].axhline(.25,color='red',ls='--');axes[3].set_xlabel('Physical time (s)');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c0_xyz_historical.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for method,d in series.items():
        _,_,rot=error_series(d)
        for j in range(3):axes[j].plot(d.time_s,-np.rad2deg(rot[:,j]),label=method)
    for ax,title in zip(axes,['World SO3 X error (deg)','World SO3 Y error (deg)','World SO3 Z error (deg)']):ax.set_title(title);ax.axvline(1.6,color='k',ls=':');ax.axvline(3.65,color='k',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c0_so3_historical.png',dpi=150);plt.close(fig);print(f[f.window=='load_all'][['method','peak_time_s','peak_norm_mm','x_at_norm_peak_mm','y_at_norm_peak_mm','z_at_norm_peak_mm']].to_string(index=False))
