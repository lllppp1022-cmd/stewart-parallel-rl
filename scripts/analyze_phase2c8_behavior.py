import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path
from phase2c8_protocol import ROOT,RUN,read,write_json

def main():
    rows=[];events=[];out=RUN/'behavior_plots';out.mkdir(exist_ok=True)
    for p in (RUN/'final').rglob('metrics.json'):
        m=read(p)
        if m['method'] in ('baseline','classical50'):continue
        d=pd.read_csv(Path(m['folder'])/'physics.csv.gz');task=m['task'];t=d.input_step.to_numpy()*.001
        req=d[['requested_mx_Nm','requested_my_Nm']].to_numpy();accepted=d[['accepted_mx_Nm','accepted_my_Nm']].to_numpy()
        actual=Rotation.from_euler('xyz',d[[f'rpy_{j}' for j in range(3)]],degrees=True);ref=Rotation.from_euler('xyz',d[[f'rpy_ref_{j}' for j in range(3)]],degrees=True);err=(actual*ref.inv()).as_rotvec();omega=d[[f'twist_actual_{j}' for j in (3,4,5)]].to_numpy()
        c0=d[[f'F_C0_{j}' for j in range(6)]].to_numpy();command=d[[f'F_command_{j}' for j in range(6)]].to_numpy();base=d[[f'same_state_base_add_{j}_N' for j in range(6)]].to_numpy();leg=d[[f'rl_mapped_leg_{j}_N' for j in range(6)]].to_numpy()
        desired=d[[f'F_C3_{j}' for j in range(6)]].to_numpy()-c0+d[[f'F_residual_requested_{j}' for j in range(6)]].to_numpy()+d[[f'ff_requested_leg_{j}_N' for j in range(6)]].to_numpy()
        previous=np.vstack([np.zeros(6),(command-c0)[:-1]]);lo=np.maximum(np.maximum(-5,previous-.02),-400-c0);hi=np.minimum(np.minimum(5,previous+.02),400-c0)
        command_error=float(abs(c0+np.clip(desired+leg,lo,hi)-command).max());base_error=float(abs(np.clip(desired,lo,hi)-base).max());assert command_error<1e-9 and base_error<1e-9
        row=dict(method=m['method'],seed=m['seed'],task_id=task['id'],requested_rms_Nm=float(np.sqrt((req**2).mean())),marginal_accepted_rms_Nm=float(np.sqrt((accepted**2).mean())),request_accept_rms_difference=float(np.sqrt(((req-accepted)**2).mean())),executed_projection_fraction=float(d.projected.mean()),counterfactual_rate_limited_fraction=float(((desired<previous-.02-1e-9)|(desired>previous+.02+1e-9)).any(axis=1).mean()),command_recompute_error_N=command_error,counterfactual_recompute_error_N=base_error,restoring_sign_fraction=float((np.sum(accepted*err[:,:2],axis=1)<0).mean()),source=m['folder']);rows.append(row)
        direction=np.array(task['torque'])/np.linalg.norm(task['torque'])
        for name,key in [('load','on'),('unload','off')]:
            edge=task[key];prior=req[np.flatnonzero(t<edge-1e-9)[-1]];changed=np.flatnonzero((t>=edge-1e-9)&(t<edge+.5)&(np.linalg.norm(req-prior,axis=1)>1e-6));latency=float((t[changed[0]]-edge)*1000) if len(changed) else None
            windows=[('pre100',edge-.1,edge)]+[(f'post{n}',edge,edge+n/1000) for n in (20,40,60,100)]+[('sustained' if name=='load' else 'recovery',edge+.1,task['off'] if name=='load' else min(edge+1,t[-1]))]
            for label,lo_t,hi_t in windows:
                mask=(t>=lo_t-1e-9)&(t<hi_t-1e-9)
                events.append(dict(method=m['method'],seed=m['seed'],task_id=task['id'],event=name,window=label,requested_along_load_Nm=float((req[mask]@direction).mean()),accepted_along_load_Nm=float((accepted[mask]@direction).mean()),first_recorded_change_ms=latency,not_proven_causal_latency=True,so3_mean_deg=float(d.orientation_error_deg[mask].mean())))
        series=[('External Mx/My (Nm)',d[['external_world_3','external_world_4']].to_numpy()),('Roll reference/actual (deg)',d[['rpy_ref_0','rpy_0']].to_numpy()),('Pitch reference/actual (deg)',d[['rpy_ref_1','rpy_1']].to_numpy()),('SO3 error (deg)',d[['orientation_error_deg']].to_numpy()),('Angular velocity xyz (rad/s)',omega),('Requested Mx/My (Nm)',req),('Marginal accepted Mx/My (Nm)',accepted),('Six actuator forces (N)',d[[f'force_{j}' for j in range(6)]].to_numpy()),('Negative physical cost / orientation cost',np.column_stack([-d[[f'dyn_cost_{n}' for n in ('orientation','omega','xyz','action','slew')]].sum(axis=1),d.dyn_cost_orientation])),('Shared additional forces (N)',command-c0)]
        for zoom in ('full','load','unload'):
            limits=(0,t[-1]) if zoom=='full' else (task['on' if zoom=='load' else 'off']-.1,task['on' if zoom=='load' else 'off']+.1);idx=np.flatnonzero((t>=limits[0])&(t<=limits[1]))[::10 if zoom=='full' else 1]
            fig,axs=plt.subplots(5,2,figsize=(15,14),sharex=True)
            for ax,(title,values) in zip(axs.flat,series):
                ax.plot(t[idx],values[idx]);ax.set_title(title,fontsize=9);ax.set_xlim(*limits);ax.grid(alpha=.2)
                for event in (task['on'],task['off']):ax.axvline(event,color='red',ls='--')
                if zoom!='full':
                    for at in np.arange(np.ceil(limits[0]/.02)*.02,limits[1],.02):ax.axvline(at,color='gray',ls=':',alpha=.3)
            fig.suptitle(f'{task["id"]} | {m["method"]}/{m["seed"]} | {zoom} | FROZEN DETERMINISTIC EVALUATION');fig.tight_layout();fig.savefig(out/f'{task["id"]}_{m["method"]}_{m["seed"]}_{zoom}.png',dpi=110);plt.close(fig)
        print('Behavior',task['id'],m['method'],m['seed'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c8_action_audit.csv',index=False);pd.DataFrame(events).to_csv(ROOT/'results/phase2c8_action_events.csv',index=False)
if __name__=='__main__':main()
