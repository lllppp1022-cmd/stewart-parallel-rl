"""Paired deterministic/stochastic evaluation, with RNG isolation from training."""
import json,random,contextlib
from pathlib import Path
import numpy as np,torch
from ..model import write_json
from ..control.phase2b1_experiment import trial_metrics
from .stewart_residual_env import StewartResidualEnv,env_config

@contextlib.contextmanager
def preserve_rng():
    pt=torch.get_rng_state();npstate=np.random.get_state();py=random.getstate()
    try:yield
    finally:torch.set_rng_state(pt);np.random.set_state(npstate);random.setstate(py)

def evaluate_model(model,output,tasks,seed=4101,deterministic=True,record=False):
    rows=[]
    with preserve_rng():
        torch.manual_seed(seed)
        for task in tasks:
            env=StewartResidualEnv(Path(output)/task['id'],tasks=[task],record_episodes=[0] if record else [],mode='Checkpoint Evaluation');obs,_=env.reset(seed=seed,options=dict(task_id=task['id']));done=False
            while not done:
                with torch.no_grad():
                    tensor=torch.as_tensor(obs.reshape(1,-1),device=model.device);dist=model.policy.get_distribution(tensor);mean=float(dist.distribution.mean[0,0]);std=float(dist.distribution.stddev[0,0]);action=dist.get_actions(deterministic=deterministic).cpu().numpy()[0]
                env.policy_context=dict(raw_action=float(action[0]),actor_mean=mean,actor_std=std,policy_version=getattr(model,'_phase_update_index',-1));obs,reward,te,tr,info=env.step(action);done=te or tr
            env.close();folder=Path(output)/task['id']/'episode_0000';r=trial_metrics(folder,dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3));r.update(task_id=task['id'],deterministic=deterministic,evaluation_seed=seed);write_json(Path(output)/task['id']/'metrics.json',r);rows.append(r)
    return rows
