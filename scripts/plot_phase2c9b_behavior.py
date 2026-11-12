"""1 ms event behavior figures for every seed and representative mass task."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase2c9b_protocol import ROOT,RUN,METHODS,read


def main():
    output=RUN/'behavior_plots';output.mkdir(exist_ok=True)
    for path in (RUN/'final/Final_m21.5_6').rglob('metrics.json'):
        m=read(path)
        if m['method'] not in METHODS:continue
        d=pd.read_csv(Path(m['folder'])/'physics.csv.gz');t=d.input_step.to_numpy()*.001;task=m['task']
        series=[('External Mx/My (Nm)',d[['external_world_3','external_world_4']]),
                ('Roll/Pitch reference / actual (deg)',d[['rpy_ref_0','rpy_0','rpy_ref_1','rpy_1']]),
                ('SO3 error (deg)',d[['orientation_error_deg']]),
                ('Angular velocity (rad/s)',d[[f'twist_actual_{j}' for j in (3,4,5)]]),
                ('Requested / marginal accepted torque (Nm)',d[['requested_mx_Nm','requested_my_Nm','accepted_mx_Nm','accepted_my_Nm']]),
                ('Six actual forces (N)',d[[f'force_{j}' for j in range(6)]])]
        for zoom in ('full','load','unload'):
            if zoom=='full':mask=np.arange(len(d))[::10]
            else:
                edge=task['on' if zoom=='load' else 'off'];mask=np.flatnonzero((t>=edge-.1)&(t<edge+.1))
            fig,axes=plt.subplots(3,2,figsize=(15,10),sharex=True)
            for ax,(title,values) in zip(axes.flat,series):
                ax.plot(t[mask],values.to_numpy()[mask]);ax.set_title(title);ax.grid(alpha=.2)
                for edge in (task['on'],task['off']):
                    if t[mask[0]]<=edge<=t[mask[-1]]:ax.axvline(edge,color='red',ls='--')
                if zoom!='full':
                    for at in np.arange(np.ceil(t[mask[0]]/.02)*.02,t[mask[-1]],.02):ax.axvline(at,color='gray',ls=':',alpha=.3)
            fig.suptitle(f"Frozen evaluation | {m['method']} | seed{m['seed']} | {task['id']} | {zoom}")
            fig.tight_layout();fig.savefig(output/f"{m['method']}_{m['seed']}_{zoom}.png",dpi=110);plt.close(fig)
    # Show all unseen mass values and every training seed, not only the
    # representative trajectory chosen for the long-form video.
    final=pd.read_csv(ROOT/'results/phase2c9b_generalization.csv')
    final['reference_case']=final.task_id.str.rsplit('_',n=1).str[-1]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for row,case in enumerate(sorted(final.reference_case.unique())):
        for ax,key in zip(axes[row],('load_iae_deg_s','unload_iae_deg_s','xyz_rmse_mm')):
            part=final[final.reference_case==case]
            for method,g in part.groupby('method'):
                mean=g.groupby('mass_kg')[key].mean().sort_index()
                line=ax.plot(mean.index,mean.values,'o-',label=method)[0]
                if method in METHODS:
                    for _,seed_group in g.groupby('seed'):
                        seed_group=seed_group.sort_values('mass_kg')
                        ax.plot(seed_group.mass_kg,seed_group[key],color=line.get_color(),alpha=.3,lw=.7)
            ax.set_title(f'Reference {case} | {key}');ax.set_xlabel('Actual platform mass (kg)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Frozen final evaluation | bold: mean; faint: all three PPO training seeds')
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c9b_generalization_curves.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
