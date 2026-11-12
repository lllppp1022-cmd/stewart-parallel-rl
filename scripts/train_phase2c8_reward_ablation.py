"""Bounded matched real PPO. Resume only at explicitly reset curriculum boundaries."""
import argparse
import copy
import pickle
import random
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read,write_json,sha
from src.rl.phase2c8_env import Phase2C8Env
from src.rl.ppo_trace import parameter_hash
from train_phase2c7_ppo import DynamicPPO
from train_phase2c5_ppo import Capture

class Phase8PPO(DynamicPPO):
    def train(self):
        b=self.rollout_buffer
        extra=dict(pre_value_mse=float(np.mean((b.returns-b.values)**2)),critic_prediction_mean=float(b.values.mean()),critic_prediction_min=float(b.values.min()),critic_prediction_max=float(b.values.max()),task_composition={k:sum(r['task_id']==k for r in self._trace_env.records[-b.buffer_size:]) for k in set(r['task_id'] for r in self._trace_env.records[-b.buffer_size:])})
        super().train();self._trace_updates[-1].update(extra)
        path=self._trace_env.root/f'rollout_{len(self._trace_updates):03d}.npz'
        with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
        arrays.update(last_values=self._trace_last_values,raw_environment_rewards=np.array([r['reward'] for r in self._trace_env.records[-b.buffer_size:]]))
        np.savez_compressed(path,**arrays);write_json(self._trace_env.root/'updates.json',self._trace_updates)

class Phase8Capture(Capture):
    def _on_rollout_end(self):self.model._trace_last_values=self.locals['values'].detach().cpu().numpy().copy()

def rng_snapshot(env):return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),tasks=env.sampler.snapshot(),global_decisions=env.global_decisions,physical_steps=env.total_physical_steps)
def restore_global(state):torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
def make_vector(env,method,norm_path=None):
    if method!='v3_return_norm':return env
    base=DummyVecEnv([lambda:env]);c=cfg('return_norm')
    if norm_path:return VecNormalize.load(norm_path,base)
    return VecNormalize(base,norm_obs=False,norm_reward=True,gamma=c['gamma'],clip_reward=c['clip_reward'],epsilon=c['epsilon'])
def create_model(env,method,seed,root):
    c=cfg('training')['ppo'];kwargs={k:c[k] for k in ('n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','max_grad_norm','target_kl')}
    model=Phase8PPO('MlpPolicy',make_vector(env,method),seed=seed,device='cpu',tensorboard_log=str(root/'tensorboard'),policy_kwargs=dict(net_arch=dict(pi=c['net'],vf=c['net']),log_std_init=c['log_std_init']),**kwargs)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    model._trace_updates=[];return model
def attach(model,env,models,updates=None):
    model._trace_env=env;model._trace_models=models
    if updates is not None:model._trace_updates=copy.deepcopy(updates)
    def forward(obs,deterministic=False):
        result=model.policy.__class__.forward(model.policy,obs,deterministic=deterministic)
        dist=model.policy.get_distribution(obs).distribution
        env.context.update(raw=result[0][0].detach().cpu().numpy().copy(),mean=dist.mean[0].detach().cpu().numpy().copy(),std=dist.stddev[0].detach().cpu().numpy().copy(),version=len(model._trace_updates))
        return result
    model.policy.forward=forward
def train_stage(method,seed,stage,budget,previous=None,model=None,custom_root=None):
    root=(custom_root or RUN/'training'/method/f'seed{seed}')/stage
    models=(custom_root/'models' if custom_root else ROOT/f'models/phase2c8/{method}/seed{seed}');models.mkdir(parents=True,exist_ok=True)
    tasks=cfg('training')['curriculum'][stage[0]]
    env=Phase2C8Env(root,tasks,seed=seed,version=method,sampler_state=previous['rng']['tasks'] if previous else None,record_episodes=[0,6,12,24] if seed==42 and custom_root is None else [])
    if previous:
        env.global_decisions=previous['rng']['global_decisions'];env.total_physical_steps=previous['rng']['physical_steps']
        vec=make_vector(env,method,previous.get('norm_path'))
        if model is None:model=Phase8PPO.load(previous['model_path'],env=vec,device='cpu',tensorboard_log=str(root/'tensorboard'))
        else:model.set_env(vec,force_reset=True);model.tensorboard_log=str(root/'tensorboard')
        # Clear pending SB3 vec reset seeds; task RNG is independently restored regardless.
        model.env._reset_seeds()
        attach(model,env,models,previous['updates']);restore_global(previous['rng'])
    else:
        model=create_model(env,method,seed,root);attach(model,env,models);model.save(models/'checkpoint_000000.zip')
    initial=parameter_hash(model.policy);start=time.perf_counter();cb=Phase8Capture(env);error=None;target=int(model.num_timesteps)+budget
    try:
        model.learn(budget,callback=cb,reset_num_timesteps=previous is None)
        if cb.stop:raise RuntimeError(cb.stop)
        assert model.num_timesteps==target and env.global_decisions==target
        output=models/f'checkpoint_{target:06d}.zip';model.save(output)
        state=rng_snapshot(env);updates=copy.deepcopy(model._trace_updates)
        norm_path=None
        if method=='v3_return_norm':
            norm_path=models/f'normalization_{target:06d}.pkl';model.env.save(norm_path)
        saved=dict(rng=state,updates=updates,model_path=str(output),norm_path=str(norm_path) if norm_path else None)
        with (models/f'boundary_{target:06d}.pkl').open('wb') as f:pickle.dump(saved,f)
        loaded=PPO.load(output,device='cpu');assert parameter_hash(loaded.policy)==parameter_hash(model.policy)
        np.testing.assert_array_equal(loaded.predict(env.obs,deterministic=True)[0],model.predict(env.obs,deterministic=True)[0]);restore_global(state)
        return model,saved
    except Exception as exc:
        error=repr(exc);model.save(models/f'interrupted_{stage}_{model.num_timesteps}.zip');raise
    finally:
        env.close();write_json(root/'summary.json',dict(method=method,seed=seed,stage=stage,decisions=model.num_timesteps,new_decisions=len(env.records),physical_steps=env.total_physical_steps,new_physical_steps=sum(r['physical_steps'] for r in env.records),updates=len(model._trace_updates),complete_episodes=sum(e['complete'] for e in env.episodes),fragments=sum(not e['complete'] for e in env.episodes),failures=sum(bool(e['failure']) for e in env.episodes),initial_hash=initial,final_hash=parameter_hash(model.policy),error=error,wall_s=time.perf_counter()-start,task_rng=env.sampler.snapshot(),control_constraints_unchanged=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['v2_control','v3','v3_return_norm'],required=True);p.add_argument('--seed',choices=SEEDS,type=int,required=True);p.add_argument('--phase',choices=['pilot','continue'],default='pilot');a=p.parse_args();torch.set_num_threads(1)
    assert read(RUN/'training_gate.json')['passed']
    if a.method=='v3_return_norm':assert read(RUN/'normalization_admission.json')['passed']
    if a.phase=='pilot':
        model,state=train_stage(a.method,a.seed,'A',4096)
        train_stage(a.method,a.seed,'B',4096,state,model)
    else:
        assert read(RUN/('normalization_pilot_gate.json' if a.method=='v3_return_norm' else 'pilot_gate.json'))['passed']
        with (ROOT/f'models/phase2c8/{a.method}/seed{a.seed}/boundary_008192.pkl').open('rb') as f:state=pickle.load(f)
        # Split C only for checkpoint files, not physical reset: one continuous learn.
        model,state=train_stage(a.method,a.seed,'C',24576,state)
        models=ROOT/f'models/phase2c8/{a.method}/seed{a.seed}'
        for n in (16384,24576):
            import shutil
            shutil.copyfile(models/f'update_{n//512:03d}.zip',models/f'checkpoint_{n:06d}.zip')
if __name__=='__main__':main()
