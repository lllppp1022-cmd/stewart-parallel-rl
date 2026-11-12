"""One bounded seed; seed42 smoke is included, not free extra interaction."""
import _bootstrap,argparse,json,time,random
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from src.model import ROOT,write_json,sha
from src.rl.stewart_large_motion_env import RUN,StewartLargeMotionResidualEnv
from src.rl.ppo_trace import TracedPPO,SamplingCallback,parameter_hash

class LargeCallback(SamplingCallback):
    def _on_step(self):
        ok=super()._on_step();r=self.real_env.records[-1]
        for key in ['episode_reward','Z_error_rmse','XY_error_rmse','SO3_error','raw_action','accepted_residual_mean_N','action_clip_fraction','budget_projection_fraction','actor_std']:
            self.logger.record('large_motion/'+key,r[key])
        for key in [x for x in r if x.startswith('reward_')]:self.logger.record('reward_components/'+key,r[key])
        self.logger.record('large_motion/episode_length',r['decision']);self.logger.record('large_motion/physical_steps',self.real_env.total_physical_steps);self.logger.record('large_motion/safety_termination_count',sum(bool(x['failure']) for x in self.real_env.episode_summaries))
        if self.real_env.episode_summaries:
            for k in ['load_window_IAE_mm_s','unload_window_IAE_mm_s']:self.logger.record('offline_episode/'+k,self.real_env.episode_summaries[-1][k])
        if time.perf_counter()-self.started_all>7200:self.stop_reason='bounded_wall_budget';return False
        return ok

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,123,2026],required=True);p.add_argument('--output-root',type=Path,default=RUN/'training');p.add_argument('--model-root',type=Path,default=ROOT/'models/phase2c4');a=p.parse_args();gate=json.loads((ROOT/'results/phase2c4_env_validation.json').read_text(encoding='utf-8'));assert gate['passed'];smoke_root=a.output_root.parent
    if a.seed!=42:assert json.loads((smoke_root/'smoke_validation.json').read_text(encoding='utf-8'))['passed']
    cfg=json.loads((ROOT/'configs/phase2c4_ppo.yaml').read_text(encoding='utf-8'));folder=a.output_root/f'seed{a.seed}';env=StewartLargeMotionResidualEnv(folder,record_episodes=cfg['record_seed42_episodes'] if a.seed==42 else []);models=a.model_root/f'seed{a.seed}';models.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.perf_counter()
    kwargs={k:cfg[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','vf_coef','max_grad_norm','target_kl']}
    m=TracedPPO('MlpPolicy',env,seed=a.seed,device='cpu',verbose=0,tensorboard_log=str(folder/'tensorboard'),policy_kwargs=dict(net_arch=cfg['net_arch'],activation_fn=torch.nn.Tanh,log_std_init=cfg['log_std_init']),**kwargs)
    with torch.no_grad():m.policy.action_net.weight.zero_();m.policy.action_net.bias.zero_()
    m.save(models/'initial.zip');initial_hash=parameter_hash(m.policy);m.attach(env,folder,models/'checkpoints');callback=LargeCallback(env,dict(protected_failures_in_last3_at_least=1));callback.started_all=start;write_json(folder/'training_config.json',dict(seed=a.seed,config=cfg,initial_parameter_hash=initial_hash,env_sha256=sha(ROOT/'configs/phase2c4_env.yaml')))
    error=None
    try:
        if a.seed==42:
            m.learn(total_timesteps=256,callback=callback,tb_log_name='large_motion',reset_num_timesteps=True)
            state=(torch.get_rng_state(),np.random.get_state(),random.getstate());m.save(models/'smoke256.zip');loaded=PPO.load(models/'smoke256.zip',device='cpu');np.testing.assert_array_equal(m.predict(env.obs,deterministic=True)[0],loaded.predict(env.obs,deterministic=True)[0]);assert parameter_hash(loaded.policy)==parameter_hash(m.policy);torch.set_rng_state(state[0]);np.random.set_state(state[1]);random.setstate(state[2]);del loaded
            passed=bool(m.num_timesteps==256 and not callback.stop_reason and len(m._phase_updates)==1 and m._phase_updates[0]['actor_change_norm']>0 and m._phase_updates[0]['critic_change_norm']>0)
            write_json(smoke_root/'smoke_validation.json',dict(passed=passed,decisions=m.num_timesteps,physical_steps=env.total_physical_steps,model=str(models/'smoke256.zip'),reload_parameter_equal=True,reload_rng_restored=True,update=m._phase_updates[0]));assert passed
            m.learn(total_timesteps=3840,callback=callback,tb_log_name='large_motion',reset_num_timesteps=False)
        else:m.learn(total_timesteps=4096,callback=callback,tb_log_name='large_motion',reset_num_timesteps=True)
    except Exception as exc:error=repr(exc);raise
    finally:
        m.save(models/'final.zip');m.finish_trace();env.close();summary=dict(seed=a.seed,decisions=int(m.num_timesteps),physical_steps=env.total_physical_steps,updates=len(m._phase_updates),optimizer_steps=sum(r['optimizer_steps'] for r in m._phase_updates),actor_updated=all(r['actor_change_norm']>0 for r in m._phase_updates),critic_updated=all(r['critic_change_norm']>0 for r in m._phase_updates),model=str(models/'final.zip'),model_sha256=sha(models/'final.zip'),parameter_hash=parameter_hash(m.policy),wall_s=time.perf_counter()-start,stop_reason=callback.stop_reason,error=error,complete_episodes=sum(x['complete'] for x in env.episode_summaries),safety_terminations=sum(bool(x['failure']) for x in env.episode_summaries));write_json(folder/'training_summary.json',summary);print('TRAINING SUMMARY',summary,flush=True)

if __name__=='__main__':main()
