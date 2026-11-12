"""Matched single-frame/history PPO with bounded, auditable boundary continuation."""
import argparse
import copy
import pickle
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,write_json
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation
from src.rl.ppo_trace import parameter_hash
from train_phase2c8_reward_ablation import Phase8PPO,Phase8Capture,attach,rng_snapshot,restore_global


class TrainingCapture(Phase8Capture):
    def _on_step(self):
        task_id=self.env.records[-1]['task_id']
        mass=next(t['mass_kg'] for t in self.env.tasks if t['id']==task_id)
        self.logger.record_mean('mass/actual_kg_log_only',mass)
        return super()._on_step()


def train(method,seed,budget,previous=None,custom_root=None):
    start_decisions=previous['rng']['global_decisions'] if previous else 0
    root=(custom_root or RUN/'training'/method/f'seed{seed}')/f'from_{start_decisions:06d}'
    models=custom_root/'models' if custom_root else ROOT/f'models/phase2c9b/{method}/seed{seed}'
    models.mkdir(parents=True,exist_ok=True)
    base=StewartAdaptiveEnv(root,cfg('tasks')['training'],seed=seed,version='v2_control',
        sampler_state=previous['rng']['tasks'] if previous else None,
        record_episodes=[0,7,20] if seed==42 and custom_root is None else [],
        source='ACTUAL STOCHASTIC PHASE2C9B TRAINING')
    env=HistoryObservation(base,5) if method=='history_5frame' else base
    config=cfg('ppo')
    kwargs={key:config[key] for key in ('n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','max_grad_norm','target_kl')}
    if previous:
        base.global_decisions=start_decisions;base.total_physical_steps=previous['rng']['physical_steps']
        model=Phase8PPO.load(previous['model_path'],env=env,device='cpu',tensorboard_log=str(root/'tensorboard'))
        model.env._reset_seeds();attach(model,base,models,previous['updates']);restore_global(previous['rng'])
    else:
        model=Phase8PPO('MlpPolicy',env,seed=seed,device='cpu',tensorboard_log=str(root/'tensorboard'),
            policy_kwargs=dict(net_arch=dict(pi=config['net'],vf=config['net']),log_std_init=config['log_std_init']),**kwargs)
        with torch.no_grad():
            model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model._trace_updates=[];attach(model,base,models)
        model.save(models/'checkpoint_000000.zip')
    initial_hash=parameter_hash(model.policy);started=time.perf_counter();error=None
    write_json(root/'network.json',dict(input_dim=model.observation_space.shape[0],parameters=sum(p.numel() for p in model.policy.parameters()),initial_hash=initial_hash))
    try:
        callback=TrainingCapture(base)
        model.learn(budget,callback=callback,reset_num_timesteps=previous is None)
        if callback.stop:raise RuntimeError(callback.stop)
        assert model.num_timesteps==start_decisions+budget
        output=models/f'checkpoint_{model.num_timesteps:06d}.zip';model.save(output)
        for point in (8192,16384,24576):
            if start_decisions<point<model.num_timesteps:
                shutil.copyfile(models/f'update_{point//512:03d}.zip',models/f'checkpoint_{point:06d}.zip')
        saved=dict(rng=rng_snapshot(base),updates=copy.deepcopy(model._trace_updates),model_path=str(output))
        with (models/f'boundary_{model.num_timesteps:06d}.pkl').open('wb') as stream:pickle.dump(saved,stream)
        restored=PPO.load(output,device='cpu')
        assert parameter_hash(restored.policy)==parameter_hash(model.policy)
        return saved
    except Exception as exc:
        error=repr(exc);model.save(models/f'interrupted_{model.num_timesteps:06d}.zip');raise
    finally:
        env.close()
        write_json(root/'summary.json',dict(method=method,seed=seed,start_decisions=start_decisions,
            new_decisions=len(base.records),physical_steps=sum(r['physical_steps'] for r in base.records),
            complete_episodes=sum(e['complete'] for e in base.episodes),fragments=sum(not e['complete'] for e in base.episodes),
            failures=sum(bool(e['failure']) for e in base.episodes),error=error,wall_s=time.perf_counter()-started,
            initial_hash=initial_hash,final_hash=parameter_hash(model.policy),total_updates=len(model._trace_updates)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--continue-training',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    assert read(RUN/'training_gate.json')['passed']
    previous=None;budget=8192
    if args.continue_training:
        assert read(RUN/'pilot_gate.json')['passed'];budget=24576
        with (ROOT/f'models/phase2c9b/{args.method}/seed{args.seed}/boundary_008192.pkl').open('rb') as stream:previous=pickle.load(stream)
    train(args.method,args.seed,budget,previous)


if __name__=='__main__':main()
