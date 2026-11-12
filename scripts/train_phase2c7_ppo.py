"""Three bounded, gated real-dynamics PPO runs. No automatic budget expansion."""
import argparse
import pickle
import random
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from phase2c7_protocol import ROOT,RUN,config,read,write_json,SEEDS
from train_phase2c5_ppo import AttitudePPO,Capture
from src.rl.ppo_trace import parameter_hash
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv
from src.model import sha


class DynamicPPO(AttitudePPO):
    def train(self):
        b=self.rollout_buffer
        extra=dict(value_target_mean=float(b.returns.mean()),value_target_std=float(b.returns.std()),
                   value_target_min=float(b.returns.min()),value_target_max=float(b.returns.max()),
                   advantage_min=float(b.advantages.min()),advantage_max=float(b.advantages.max()))
        super().train()
        self._trace_updates[-1].update(extra)
        write_json(self._trace_env.root/'updates.json',self._trace_updates)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--phase',choices=['pilot','continue'],default='pilot')
    args=p.parse_args()
    assert read(RUN/'training_gate.json')['passed']
    if args.phase=='continue':assert read(RUN/'pilot_gate.json')['passed']
    admitted=read(RUN/'admission.json')['admitted']
    stages={k:[t for t in admitted if t['stage']==k] for k in ('A','B','C')}
    assert all(stages.values())
    torch.set_num_threads(1)
    seed=args.seed
    root=RUN/'training'/f'seed{seed}'/args.phase
    models=ROOT/'models/phase2c7'/f'seed{seed}'
    cfg=config('ppo')
    kwargs={k:cfg[k] for k in ('n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','max_grad_norm','target_kl')}
    env=StewartDynamicAttitudeEnv(root,stages['A' if args.phase=='pilot' else 'C'],
                                record_episodes=[0,6,12,24] if seed==42 else [])
    if args.phase=='pilot':
        models.mkdir(parents=True,exist_ok=False)
        model=DynamicPPO('MlpPolicy',env,seed=seed,device='cpu',tensorboard_log=str(root/'tensorboard'),
                         policy_kwargs=dict(net_arch=dict(pi=cfg['net'],vf=cfg['net']),log_std_init=cfg['log_std_init']),**kwargs)
        with torch.no_grad():
            model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model._trace_updates=[]
        model.save(models/'initial.zip')
    else:
        model=DynamicPPO.load(models/'pilot.zip',env=env,device='cpu',tensorboard_log=str(root/'tensorboard'))
        prior=read(RUN/'training'/f'seed{seed}'/'pilot'/'summary.json')
        model._trace_updates=read(RUN/'training'/f'seed{seed}'/'pilot'/'updates.json')
        with (models/'pilot_rng.pkl').open('rb') as stream:state=pickle.load(stream)
        torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
        env.np_random.bit_generator.state=state['environment']
        env.total_physical_steps=prior['physical_steps'];env.global_decisions=prior['decisions']
    model._trace_env,model._trace_models=env,models
    initial=parameter_hash(model.policy)
    callback=Capture(env)
    forward=model.policy.forward
    def traced(obs,deterministic=False):
        action,value,log_prob=forward(obs,deterministic=deterministic)
        distribution=model.policy.get_distribution(obs).distribution
        env.context.update(raw=action[0].detach().cpu().numpy().copy(),version=len(model._trace_updates),
                           mean=distribution.mean[0].detach().cpu().numpy().copy(),
                           std=distribution.stddev[0].detach().cpu().numpy().copy())
        return action,value,log_prob
    model.policy.forward=traced
    start=time.perf_counter();error=None
    output=models/('pilot.zip' if args.phase=='pilot' else 'final.zip')
    try:
        if args.phase=='pilot':
            model.learn(4096,callback=callback)
            if callback.stop:raise RuntimeError(callback.stop)
            env.finish('curriculum_stage_boundary')
            env.tasks=stages['B'];model._last_obs=None
            model.learn(4096,callback=callback,reset_num_timesteps=False)
        else:model.learn(24576,callback=callback,reset_num_timesteps=False)
        if callback.stop:raise RuntimeError(callback.stop)
        assert model.num_timesteps==(8192 if args.phase=='pilot' else 32768)
        model.save(output)
        rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                 environment=env.np_random.bit_generator.state)
        loaded=PPO.load(output,device='cpu')
        assert parameter_hash(loaded.policy)==parameter_hash(model.policy)
        np.testing.assert_array_equal(loaded.predict(env.obs,deterministic=True)[0],model.predict(env.obs,deterministic=True)[0])
        if args.phase=='pilot':
            with (models/'pilot_rng.pkl').open('wb') as stream:pickle.dump(rng,stream)
    except Exception as exc:
        error=repr(exc);model.save(models/(args.phase+'_interrupted.zip'))
        raise
    finally:
        env.close()
        write_json(root/'summary.json',dict(seed=seed,phase=args.phase,decisions=model.num_timesteps,
           physical_steps=env.total_physical_steps,updates=len(model._trace_updates),
           adam_steps=sum(x['adam_steps'] for x in model._trace_updates),stop=callback.stop,error=error,
           wall_s=time.perf_counter()-start,complete_episodes=sum(x['complete'] for x in env.episodes),
           failures=sum(bool(x['failure']) for x in env.episodes),initial_hash=initial,final_hash=parameter_hash(model.policy),
           model_sha=sha(output) if output.exists() else None,stages={k:[t['id'] for t in v] for k,v in stages.items()},
           physical_steps_include_previous_phase=args.phase=='continue',new_decisions=len(env.records)))


if __name__=='__main__':main()
