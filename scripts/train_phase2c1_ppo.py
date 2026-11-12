"""Cumulative256/2048/4096 decision Pilot, gated; no unbounded retries."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from src.model import ROOT,write_json,sha
from src.rl.stewart_residual_env import StewartResidualEnv,env_config
from src.rl.ppo_trace import TracedPPO,SamplingCallback,parameter_hash
from src.rl.evaluation import evaluate_model,preserve_rng

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();root=a.run_root;hyper=json.loads((ROOT/'configs/phase2c1_ppo.yaml').read_text(encoding='utf-8'));cfg=env_config();assert a.seed in hyper['training_seeds'];gate=json.loads((ROOT/'results/phase2c1_training_start_gate.json').read_text(encoding='utf-8'));assert gate['passed']
    if a.seed!=42:assert json.loads((root/'seed42/stage2_gate.json').read_text(encoding='utf-8'))['passed']
    folder=root/f'seed{a.seed}';folder.mkdir(parents=True,exist_ok=False);models=ROOT/'models/phase2c1'/root.name/f'seed{a.seed}';models.mkdir(parents=True,exist_ok=False);torch.set_num_threads(hyper['torch_threads']);start=time.perf_counter();env=StewartResidualEnv(folder/'actual_training',config=cfg,record_episodes=hyper['record_training_episodes'] if a.seed==42 else [],mode='Actual Training')
    kwargs={k:hyper[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','vf_coef','max_grad_norm','target_kl']};model=TracedPPO('MlpPolicy',env,seed=a.seed,device='cpu',policy_kwargs=dict(net_arch=hyper['net_arch'],activation_fn=torch.nn.Tanh,log_std_init=hyper['log_std_init']),tensorboard_log=str(folder/'tensorboard'),verbose=0,**kwargs)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    model.attach(env,folder,models);initial=models/'initial.zip';model.save(initial);initial_hash=parameter_hash(model.policy);write_json(folder/'configuration_frozen.json',dict(seed=a.seed,env=cfg,ppo=hyper,actual_model_class='SB3 PPO MlpPolicy, separate32x32 Actor/Critic',observation_dimension=env.observation_space.shape,action_dimension=env.action_space.shape,initial_sha256=sha(initial),initial_parameter_hash=initial_hash));callback=SamplingCallback(env,hyper['stop_rules']);stages=[];failure=None
    try:
        for target in hyper['cumulative_stage_decisions']:
            delta=target-model.num_timesteps;assert delta>0 and delta%hyper['n_steps']==0
            model.learn(total_timesteps=delta,reset_num_timesteps=False,callback=callback,tb_log_name=f'seed{a.seed}',progress_bar=False)
            if callback.stop_reason:failure=callback.stop_reason;break
            assert model.num_timesteps==target and env.global_decisions==target
            checkpoint=models/f'stage_{target:04d}.zip';model.save(checkpoint)
            with preserve_rng():
                reloaded=TracedPPO.load(checkpoint,device='cpu');assert parameter_hash(reloaded.policy)==parameter_hash(model.policy);x=model._last_obs.copy();aa=model.predict(x,deterministic=True)[0];bb=reloaded.predict(x,deterministic=True)[0];assert np.array_equal(aa,bb)
            result=dict(target=target,actual_decisions=model.num_timesteps,physical_steps=env.total_physical_steps,updates=len(model._phase_updates),checkpoint=str(checkpoint),sha256=sha(checkpoint),reload_exact=True,passed=True)
            if target>=2048:
                metrics=evaluate_model(model,folder/f'validation_{target}',cfg['validation_tasks'],seed=3000+a.seed,deterministic=True)
                ratios=[]
                for row in metrics:
                    baseline=json.loads((root/'env_validation/precheck'/row['task_id']/'zero/physical_metrics.json').read_text(encoding='utf-8'));ratio=row['load_position_iae_mm_s']/baseline['load_position_iae_mm_s'];slew=row['max_force_rate_N_s']/baseline['max_force_rate_N_s'];passed=bool(row['success'] and ratio<=hyper['stop_rules']['stage_validation_load_IAE_max_ratio_to_C4'] and slew<=hyper['stop_rules']['stage_validation_force_slew_max_ratio_to_C4']);ratios.append(dict(task=row['task_id'],IAE_ratio_to_C4=ratio,force_slew_ratio=slew,passed=passed))
                result.update(validation=ratios,passed=all(x['passed'] for x in ratios))
            stages.append(result);write_json(folder/f'stage{1 if target==256 else 2 if target==2048 else 3}_gate.json',result);print('STAGE',target,'PASSED',result['passed'],flush=True)
            if not result['passed']:failure='significant_control_degradation';break
        final=models/'final.zip';model.save(final);write_json(folder/'training_result.json',dict(seed=a.seed,actual_decisions=model.num_timesteps,env_decisions=env.global_decisions,actual_physical_steps=env.total_physical_steps,updates=len(model._phase_updates),optimizer_steps_total=model._phase_updates[-1]['optimizer_steps_total'] if model._phase_updates else 0,stages=stages,stop_reason=failure,initial_parameter_hash=initial_hash,final_parameter_hash=parameter_hash(model.policy),final_checkpoint=str(final),final_sha256=sha(final),wall_seconds=time.perf_counter()-start,protected_episodes=sum(bool(x['failure']) for x in env.episode_summaries),budget_per_seed=4096))
    except Exception as exc:
        model.save(models/'exception_last.zip');write_json(folder/'training_exception.json',dict(error=repr(exc),algorithm_decisions=model.num_timesteps,env_decisions=env.global_decisions,physical_steps=env.total_physical_steps));raise
    finally:model.finish_trace();env.close()
    print('SEED FINISHED',a.seed,model.num_timesteps,'stop',failure,flush=True)
if __name__=='__main__':main()
