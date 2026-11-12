import _bootstrap,argparse,json,gc
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from src.model import ROOT,write_json,sha
from src.rl.stewart_large_motion_env import RUN,StewartLargeMotionResidualEnv,large_owner,FIXED
from src.workspace.feedforward_metrics import metrics
from src.disturbance.recording import install_colored_capture

def tasks():return [FIXED]+json.loads((ROOT/'configs/phase2c4_eval.yaml').read_text(encoding='utf-8'))['final_tasks']
def matrix():return [(t,m) for t in tasks() for m in ['C4','fourDlead','FF-VA','FF-VA-zero','initial_stochastic','PPO42','PPO123','PPO2026']]

def one(task,method):
    root=RUN/'evaluation'/task['id']/method;record=task['id']=='E1_newtiming' and method in ['C4','fourDlead','FF-VA','PPO42']
    if method in ['C4','fourDlead','FF-VA']:
        o,r=large_owner(root,task,record,method)
        if record:install_colored_capture(o.recorder,o.model)
        for _ in range(round(r.duration*1000)):
            o.step()
            if o.failure and o.data.time>=o.protect_end-1e-10:break
        o.recorder.close(dict(complete=o.n>=round(r.duration*1000) and not o.failure,source='Baseline Comparison',method=method));folder=root;modelsha=''
    else:
        e=StewartLargeMotionResidualEnv(root,tasks=[task],record_episodes=[0] if record else [],mode='Checkpoint Evaluation');obs,_=e.reset(seed=6401);r=e.reference;model=None;modelsha=''
        if method!='FF-VA-zero':
            seed=42 if method=='initial_stochastic' else int(method[3:]);path=ROOT/f'models/phase2c4/seed{seed}'/('initial.zip' if method=='initial_stochastic' else 'final.zip')
            model=PPO.load(path,device='cpu');modelsha=sha(path);torch.manual_seed(6401)
        for _ in range(round(r.duration/.05)):
            raw=0.;mean=0.;std=0.
            if model:
                with torch.no_grad():dist=model.policy.get_distribution(torch.as_tensor(obs.reshape(1,-1),device='cpu'));mean=float(dist.distribution.mean[0,0]);std=float(dist.distribution.stddev[0,0]);raw=float(dist.sample()[0,0]) if method=='initial_stochastic' else mean
            e.policy_context=dict(raw_action=raw,actor_mean=mean,actor_std=std,policy_version=16 if method.startswith('PPO') else 0);obs,_,term,trunc,_=e.step([raw])
            if term or trunc:break
        folder=Path(e.episode_summaries[-1]['folder']);e.close();del e,model
    result=metrics(folder,r);d=__import__('pandas').read_csv(Path(folder)/'physics.csv.gz');add=d[[f'F_command_{i}' for i in range(6)]].to_numpy()-d[[f'F_C0_{i}' for i in range(6)]].to_numpy();rate=np.diff(np.vstack([np.zeros((1,6)),add]),axis=0)/.001
    result.update(method=method,task_id=task['id'],task=task,source='Stochastic untrained Actor evaluation' if method=='initial_stochastic' else 'Deterministic Checkpoint Evaluation' if method.startswith('PPO') else 'Baseline Comparison',model_sha256=modelsha,additional_force_peak_N=float(abs(add).max()),additional_slew_peak_N_s=float(abs(rate).max()),budget_projection_fraction=float(np.any(abs(d[[f'ff_desired_total_add_N_{i}' for i in range(6)]].to_numpy()-add)>1e-9,axis=1).mean()))
    for k,t in [('load',task['on_s']),('unload',task['off_s'])]:s=(d.time_s>=t)&(d.time_s<t+1);result[k+'_IAE_mm_s']=float(abs(d.loc[s,'p_2']-d.loc[s,'p_ref_2']).sum()*.001*1000)
    write_json(root/'evaluation_result.json',result);print('EVALUATION',task['id'],method,result['success'],result['z_rmse_mm'],result['load_IAE_mm_s'],flush=True);del d;gc.collect();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-id',type=int,choices=[0,1,2]);p.add_argument('--collect',action='store_true');a=p.parse_args()
    for seed in [42,123,2026]:s=json.loads((RUN/f'training/seed{seed}/training_summary.json').read_text(encoding='utf-8'));assert s['decisions']==4096 and not s['error'] and not s['stop_reason']
    torch.set_num_threads(1)
    if a.collect:
        rows=[]
        for i in range(3):rows+=json.loads((RUN/f'evaluation_worker{i}.json').read_text(encoding='utf-8'))
        assert len(rows)==24;write_json(RUN/'evaluation_all.json',rows);__import__('pandas').DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(ROOT/'results/phase2c4_controller_comparison.csv',index=False);return
    if a.worker_id is None:p.error('worker-id required')
    rows=[]
    for t,m in matrix()[a.worker_id::3]:rows.append(one(t,m));write_json(RUN/f'evaluation_worker{a.worker_id}.json',rows)
if __name__=='__main__':main()
