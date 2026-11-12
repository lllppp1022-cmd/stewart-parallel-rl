"""Actual command/force comparison, independent of static geometry plots."""
import _bootstrap
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT
from analyze_phase2c3a_workspace import RUN

def main():
    f=RUN/'dynamics/fourDlead_L50_T10_tau0.01_load0/physics.csv.gz';d=pd.read_csv(f);fig,ax=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for j,a in enumerate(ax.flat):
        a.plot(d.time_s,d[f'F_command_{j}'],label='Bounded command before lag',linestyle='--',color='#cf7d23');a.plot(d.time_s,d[f'force_{j}'],label='Actual native force',color='#2174a6',alpha=.8);a.set_title(f'Leg{j+1}');a.set_ylabel('Force (N)');a.grid(alpha=.2);a.legend(fontsize=8)
    for a in ax[-1]:a.set_xlabel('Physical time (s)')
    fig.suptitle('REAL L50 execution, 10ms synthetic lag: command is not actual force');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c3a_command_vs_force.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
