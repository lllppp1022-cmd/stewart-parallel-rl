"""Archived final-policy behavior and off-manifold local input sensitivities."""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from phase2c7v_common import ROOT,OLD,RUN,SEEDS,read,write_json,checkpoint,sha
from src.rl.ppo_trace import parameter_hash

GROUPS=[('SO3 error',0,3,'rad',[.001]*3,'measurement/reference'),('angular velocity',3,6,'rad/s',[.02]*3,'measurement'),
 ('reference angular velocity',6,9,'rad/s',[.02]*3,'known reference'),('XYZ error',9,12,'m',[.0002,.0002,.00005],'measurement/reference'),
 ('linear velocity',12,15,'m/s',[.01]*3,'measurement'),('relative slides',15,21,'m',[.06]*6,'measurement'),
 ('slide velocity',21,27,'m/s',[.03]*6,'measurement'),('known command',27,33,'N',[5]*6,'known command minus home force'),
 ('causal observer',33,39,'N,Nm',[20,20,20,.8,.8,.8],'causal estimate'),('base compensation',39,45,'N',[5]*6,'same-state known controller'),
 ('previous request',45,47,'Nm',[.05]*2,'previous command'),('previous accepted',47,49,'Nm',[.05]*2,'known governor'),
 ('reference rotation',49,52,'rad',[.01]*3,'known reference'),('reference Z/v/a',52,55,'m,m/s,m/s2',[.05,.03,.03],'known reference'),
 ('reference angular acceleration',55,58,'rad/s2',[.02]*3,'known reference')]

def data(folder):
    d=pd.read_csv(folder/'physics.csv.gz')
    a=Rotation.from_euler('xyz',d[[f'rpy_{i}' for i in range(3)]],degrees=True)
    b=Rotation.from_euler('xyz',d[[f'rpy_ref_{i}' for i in range(3)]],degrees=True)
    err=(a*b.inv()).as_rotvec()
    for i in range(3):d[f'error_rad_{i}']=err[:,i]
    return d

def behavior():
    rows=[];action=[];out=RUN/'behavior_plots';out.mkdir(exist_ok=True)
    for task in read(ROOT/'configs/phase2c7_eval.yaml')['final']:
        classical=data(OLD/'final'/task['id']/'classical50/episode_0000')
        for seed in SEEDS:
            folder=OLD/'final'/task['id']/f'PPO_{seed}'/'episode_0000';d=data(folder)
            t=d.input_step.to_numpy()*.001
            req=d[['requested_mx_Nm','requested_my_Nm']].to_numpy();accepted=d[['accepted_mx_Nm','accepted_my_Nm']].to_numpy()
            err=d[['error_rad_0','error_rad_1']].to_numpy();omega=d[['twist_actual_3','twist_actual_4']].to_numpy()-d[['twist_ref_3','twist_ref_4']].to_numpy()
            norm=np.linalg.norm(req,axis=1);direction=np.array(task['torque'])/np.linalg.norm(task['torque'])
            # Actual force-map projection is distinct from the controller's nominal Jacobian.
            delta=d[[f'F_command_{i}' for i in range(6)]].to_numpy()-d[[f'F_C0_{i}' for i in range(6)]].to_numpy()-d[[f'same_state_base_add_{i}_N' for i in range(6)]].to_numpy()
            j=d[[f'true_kinematic_map_{i}' for i in range(36)]].to_numpy().reshape(-1,6,6)
            mapped=np.einsum('nji,nj->ni',j,delta)[:,3:5]
            action.append(dict(seed=seed,task_id=task['id'],requested_rms_Nm=float(np.sqrt((req**2).mean())),accepted_rms_Nm=float(np.sqrt((accepted**2).mean())),
                requested_accepted_rms_ratio=float(np.sqrt((accepted**2).mean())/max(np.sqrt((req**2).mean()),1e-15)),
                request_accept_error_rms_Nm=float(np.sqrt(((accepted-req)**2).mean())),budget_projected_fraction=float(d.projected.mean()),
                actual_vs_nominal_mapping_max_Nm=float(abs(mapped-accepted).max()),restoring_sign_fraction=float((np.sum(accepted*err,1)<0).mean()),
                damping_sign_fraction=float((np.sum(accepted*omega,1)<0).mean()),source=str(folder)))
            for event,key in [('load','on'),('unload','off')]:
                edge=task[key];pre=(t>=edge-.1-1e-9)&(t<edge-1e-9)
                last=req[np.flatnonzero(t<edge-1e-9)[-1]]
                changed=np.flatnonzero((t>=edge-1e-9)&(t<edge+.5)&(np.linalg.norm(req-last,axis=1)>1e-6))
                first=float((t[changed[0]]-edge)*1000) if len(changed) else None
                windows=[('pre100',edge-.1,edge)]+[(f'post{ms}',edge,edge+ms/1000) for ms in (20,40,60,100)]
                windows += [('sustained',task['on']+.1,task['off'])] if event=='load' else [('recovery',edge+.1,min(edge+1,t[-1]))]
                for name,lo,hi in windows:
                    m=(t>=lo-1e-9)&(t<hi-1e-9)
                    if not m.any():continue
                    row=dict(seed=seed,task_id=task['id'],event=event,window=name,start_s=lo,end_s=hi,
                        first_recorded_request_change_ms=first,not_a_causal_latency_measure=True,
                        post_event_observation_opportunity_ms=float((np.ceil((edge+.001-1e-10)/.02)*.02-edge)*1000),
                        requested_along_load_Nm=float((req[m]@direction).mean()),accepted_along_load_Nm=float((accepted[m]@direction).mean()),
                        so3_mean_deg=float(d.orientation_error_deg[m].mean()),error_along_load_deg=float((err[m]@direction).mean()*180/np.pi),
                        budget_projection_fraction=float(d.projected[m].mean()))
                    for k,axis in enumerate(('mx','my')):row['requested_'+axis]=float(req[m,k].mean());row['accepted_'+axis]=float(accepted[m,k].mean())
                    rows.append(row)
            for zoom in ('full','load','unload'):
                fig,axs=plt.subplots(5,2,figsize=(15,15),sharex=True)
                limits=(0,t[-1]) if zoom=='full' else (task['on' if zoom=='load' else 'off']-.1,task['on' if zoom=='load' else 'off']+.1)
                m=(t>=limits[0])&(t<=limits[1]);stride=10 if zoom=='full' else 1;idx=np.flatnonzero(m)[::stride];x=t[idx]
                plots=[('External Mx/My (Nm)',d[['external_world_3','external_world_4']].to_numpy()),
                    ('Roll reference/actual (deg)',d[['rpy_ref_0','rpy_0']].to_numpy()),('Pitch reference/actual (deg)',d[['rpy_ref_1','rpy_1']].to_numpy()),
                    ('SO3 error (deg)',d[['orientation_error_deg']].to_numpy()),('Angular velocity xyz (rad/s)',d[[f'twist_actual_{i}' for i in (3,4,5)]].to_numpy()),
                    ('PPO requested Mx/My (Nm)',req),('PPO accepted Mx/My (Nm)',accepted),('Six actual forces (N)',d[[f'force_{i}' for i in range(6)]].to_numpy()),
                    ('Classical requested Mx/My (Nm), independent matched trajectory',classical[['requested_mx_Nm','requested_my_Nm']].to_numpy()),
                    ('Negative physical cost / orientation cost (20ms-normalized rate)',np.column_stack([-d[[f'dyn_cost_{n}' for n in ('orientation','omega','xyz','action','slew')]].sum(axis=1),d.dyn_cost_orientation]))]
                for ax,(label,values) in zip(axs.flat,plots):
                    for c in range(values.shape[1]):ax.plot(x,values[idx,c],label=str(c+1),lw=.9)
                    ax.set_title(label,fontsize=9);ax.grid(alpha=.2);ax.set_xlim(*limits)
                    for edge in (task['on'],task['off']):ax.axvline(edge,color='red',ls='--',alpha=.5)
                    if zoom!='full':
                        for at in np.arange(np.ceil(limits[0]/.02)*.02,limits[1]+1e-6,.02):ax.axvline(at,color='gray',ls=':',alpha=.3)
                fig.suptitle(f'{task["id"]} | seed {seed} | {zoom} | archived deterministic evaluation; no new training')
                fig.tight_layout();fig.savefig(out/f'{task["id"]}_seed{seed}_{zoom}.png',dpi=120);plt.close(fig)
            print('Behavior',task['id'],seed,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c7v_policy_behavior.csv',index=False)
    pd.DataFrame(action).to_csv(ROOT/'results/phase2c7v_action_audit.csv',index=False)

def sensitivity():
    torch.set_num_threads(1);rows=[];details=[]
    for seed in SEEDS:
        model=PPO.load(checkpoint(seed,32768),device='cpu');model.policy.set_training_mode(False);before=parameter_hash(model.policy)
        with np.load(OLD/f'training/seed{seed}/continue/rollout_064.npz') as b:obs=b['observations'].reshape(-1,58)[::8].copy()
        eps=.01;derivatives=[]
        for col in range(58):
            plus=obs.copy();minus=obs.copy();plus[:,col]+=eps;minus[:,col]-=eps
            assert abs(plus).max()<10 and abs(minus).max()<10
            with torch.no_grad():
                get=lambda x:model.policy.get_distribution(torch.as_tensor(x)).distribution.mean.numpy()
                delta=(np.clip(get(plus),-1,1)-np.clip(get(minus),-1,1))*.05/(2*eps)
            derivatives.append(delta)
        jac=np.stack(derivatives,axis=2)
        for name,lo,hi,unit,scale,source in GROUPS:
            for axis in (0,1):rows.append(dict(seed=seed,group=name,axis=('Mx','My')[axis],mean_l2_sensitivity_Nm_per_normalized_unit=float(np.linalg.norm(jac[:,axis,lo:hi],axis=1).mean()),label='LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC',samples=len(obs)))
            for col in range(lo,hi):
                for axis in (0,1):details.append(dict(seed=seed,feature=col,group=name,axis=axis,signed_mean=float(jac[:,axis,col].mean()),mean_absolute=float(abs(jac[:,axis,col]).mean()),physical_scale=scale[col-lo],unit=unit))
        assert parameter_hash(model.policy)==before
    pd.DataFrame(rows).to_csv(RUN/'sensitivity_groups.csv',index=False);pd.DataFrame(details).to_csv(RUN/'sensitivity_features.csv',index=False)
    write_json(RUN/'observation_groups.json',[dict(name=n,start=a,end_exclusive=b,dimension=b-a,unit=u,scale=s,source=source) for n,a,b,u,s,source in GROUPS])

if __name__=='__main__':behavior();sensitivity()
