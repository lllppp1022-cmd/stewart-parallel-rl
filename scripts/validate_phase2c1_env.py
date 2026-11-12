"""API, exact zero-residual gate, reward reconstruction and finite combo prechecks."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from stable_baselines3.common.env_checker import check_env
from src.model import ROOT,write_json
from src.rl.stewart_residual_env import StewartResidualEnv,env_config
from src.rl.rewards import physical_cost,failure_penalty
from src.control.phase2b1_experiment import trial_metrics

def audit_folder(folder,cfg):
    folder=Path(folder);d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');previous=np.r_[np.zeros((1,6)),d[[f'rl_applied_command_{j}' for j in range(6)]].to_numpy()[:-1]];costs={k:[] for k in ['z','xy','angle','amplitude','slew','request']}
    for i,row in enumerate(d.itertuples()):
        e=np.array([getattr(row,f'p_{j}')-getattr(row,f'p_ref_{j}') for j in range(3)]);a=np.array([getattr(row,f'rl_applied_command_{j}') for j in range(6)]);parts,factor=physical_cost(e,np.deg2rad(row.orientation_error_deg),a,previous[i],row.rl_mapped_target4_0,cfg['action_scale_N'],cfg['reward'])
        for key,value in parts.items():assert abs(value-getattr(row,'reward_cost_'+key))<1e-9;costs[key].append(value)
    return d,costs

def run_episode(output,task,pattern,seed=7401):
    cfg=env_config();env=StewartResidualEnv(output,tasks=[task],mode='Random Safety Precheck' if pattern!='zero' else 'Baseline Comparison');obs,_=env.reset(seed=seed,options=dict(task_id=task['id']));rng=np.random.default_rng(seed);done=False
    while not done:
        a=0 if pattern=='zero' else rng.uniform(-1,1) if pattern=='seeded_uniform' else (1 if env.decision%2==0 else -1);obs,reward,te,tr,info=env.step(np.array([a],np.float32));assert obs.shape==(255,) and obs.dtype==np.float32 and np.isfinite(obs).all() and np.isfinite(reward);done=te or tr
    env.close();folder=Path(output)/'episode_0000';r=trial_metrics(folder,dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3));r.update(pattern=pattern,task_id=task['id']);write_json(Path(output)/'physical_metrics.json',r);d,costs=audit_folder(folder,cfg);samples=pd.read_csv(Path(output)/'training_samples.csv');checks=[]
    for row in samples.itertuples():
        sl=slice(int(row.physical_start_step),int(row.physical_end_step));recomputed=-sum(sum(np.asarray(values)[sl]) for values in costs.values())*.001/.05+row.failure_penalty;assert abs(recomputed-row.reward)<1e-8;checks.append(dict(folder=str(folder),decision=row.decision,logged_reward=row.reward,recomputed_reward=recomputed,error=abs(recomputed-row.reward),physical_steps=row.action_physical_steps+row.hold_physical_steps))
    pd.DataFrame(checks).to_csv(Path(output)/'reward_audit.csv',index=False);print(task['id'],pattern,'PASS' if r['success'] else 'FAIL',flush=True);return r

def api(root):
    cfg=env_config();out=root/'api';env=StewartResidualEnv(out,mode='API Validation');check_env(env,warn=True);env.close()
    # Full historical zero action under the same10ms, -6N and1.7/3.45s profile.
    r=run_episode(root/'zero_equivalence',cfg['regression_task'],'zero');new=root/'zero_equivalence/episode_0000';old=ROOT/'results/runs/phase2c0_20261029_v1/lag_development/zload/C4_tau0.01'
    with np.load(old/'integration_states.npz') as a,np.load(new/'integration_states.npz') as b:assert np.array_equal(a['states'],b['states']);delta=float(abs(a['states']-b['states']).max())
    a=pd.read_csv(old/'physics.csv.gz',float_precision='round_trip');b=pd.read_csv(new/'physics.csv.gz',float_precision='round_trip');cols=[c for c in a.columns if c.startswith(('qpos_','qvel_','ctrl_','force_','F_command_','F_drive_','diag_','observer_memory_wrench_'))];assert np.array_equal(a[cols].to_numpy(),b[cols].to_numpy())
    assert json.loads((old/'summary.json').read_text())['failure']==json.loads((new/'summary.json').read_text())['failure']
    write_json(root/'api_gate.json',dict(check_env_passed=True,zero_complete_state_bitwise_equal=True,zero_signal_observer_equal=True,maximum_difference=delta,zero_decisions=120,zero_physics_steps=6000));print('API AND ZERO GATE PASS',flush=True)

def worker(root,ids):
    cfg=env_config();all_tasks=cfg['training_tasks']+cfg['validation_tasks']+cfg['final_tasks'];rows=[]
    for task in all_tasks:
        if task['id'] not in ids:continue
        for pattern in ['zero']+cfg['random_safety_patterns']:rows.append(run_episode(root/'precheck'/task['id']/pattern,task,pattern,seed=cfg['random_safety_seed']))
    write_json(root/f'worker_{"_".join(ids)}.json',dict(results=rows,passed=all(r['success'] for r in rows)));assert all(r['success'] for r in rows)

def finish(root):
    cfg=env_config();tasks=cfg['training_tasks']+cfg['validation_tasks']+cfg['final_tasks'];rows=[];audits=[]
    assert json.loads((root/'api_gate.json').read_text())['zero_complete_state_bitwise_equal']
    for task in tasks:
        for pattern in ['zero']+cfg['random_safety_patterns']:
            folder=root/'precheck'/task['id']/pattern;r=json.loads((folder/'physical_metrics.json').read_text());assert r['success'];rows.append(r);audits.append(pd.read_csv(folder/'reward_audit.csv'))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c1_safety_prechecks.csv',index=False);pd.concat(audits,ignore_index=True).to_csv(ROOT/'results/phase2c1_reward_audit.csv',index=False);write_json(ROOT/'results/phase2c1_env_validation.json',dict(api=json.loads((root/'api_gate.json').read_text()),task_combinations=len(tasks),patterns=3,precheck_episodes=len(rows),all_success=True,action_scale_N=cfg['action_scale_N'],physical_steps=sum(r['physical_steps'] for r in rows)+6000,environment_gate_passed=True,remaining_required='reset/termination/observer equivalence tests before training',no_training_yet=True));print('SAFETY/REWARD GATES PASS',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--part',choices=['api','worker','finish'],required=True);p.add_argument('--task-ids',nargs='*',default=[]);a=p.parse_args();a.run_root.mkdir(parents=True,exist_ok=True)
    if a.part=='api':api(a.run_root)
    elif a.part=='worker':worker(a.run_root,a.task_ids)
    else:finish(a.run_root)
