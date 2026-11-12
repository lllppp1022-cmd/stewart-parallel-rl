"""Paired task plots from final CSV, including the observed Home tradeoff."""
import _bootstrap,json
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_phase2c2 import RUN
from src.model import ROOT

def main():
    d=pd.read_csv(ROOT/'results/phase2c2_evaluation_comparison.csv');reg=json.loads((RUN/'regression_results.json').read_text(encoding='utf-8'));labels=['c4','fz_lead','four_d_lead','PPO42','PPO123'];fig,ax=plt.subplots(1,2,figsize=(13,5));x=np.arange(5)
    for off,task,color in [(-.17,'F_move1','#367bbb'),(.17,'F_move2','#d88722')]:
        vals=[float(d[(d.label==k)&(d.task_id==task)].z_rmse_mm.iloc[0])*1000 for k in labels];ax[0].bar(x+off,vals,width=.34,label=task,color=color)
    ax[0].set_xticks(x,labels,rotation=20);ax[0].set_title('Untouched final moving tasks (6 s, all completed)');ax[0].set_ylabel('Full-trajectory Z RMSE (um)');ax[0].legend()
    names=['fourDlead','PPO42','PPO123'];v=[r['z_rmse_mm']*1000 for r in reg];ax[1].bar(names,v,color=['#707b85','#249e83','#b75d95']);ax[1].set_ylabel('Full-episode Z RMSE (um)');ax[1].set_title('Old Home regression: small precision cost')
    for a in ax:a.set_ylim(bottom=0);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    ax[0].set_ylim(0,d.z_rmse_mm.max()*1000*1.23)
    fig.suptitle('Same constraints and matched task conditions; two training seeds, no significance claim');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c2_controller_comparison.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
