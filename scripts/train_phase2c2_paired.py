"""Two live training states retained across joint gate; global RNG restored per seed."""
import _bootstrap
import json,time,random
import numpy as np,torch
from research_phase2c2 import RUN,PROTOCOL,cfg_for,episode
from src.model import ROOT,write_json,sha
from src.rl.phase2c2_env import MovingResidualEnv
from src.rl.ppo_trace import TracedPPO,SamplingCallback,parameter_hash
from src.rl.evaluation import preserve_rng

def rng_get():return (torch.get_rng_state(),np.random.get_state(),random.getstate())
def rng_set(s):torch.set_rng_state(s[0]);np.random.set_state(s[1]);random.setstate(s[2])

def main():
    assert json.loads((RUN/'training_gate.json').read_text(encoding="utf-8"))['passed'];assert json.loads((RUN/'observer_equivalence_v2.json').read_text(encoding="utf-8"))['passed']
    torch.set_num_threads(1);jobs=[]
    try:
        for seed in [42,123]:
            folder=RUN/f'seed{seed}';folder.mkdir(exist_ok=False);models=ROOT/'models/phase2c2'/RUN.name/f'seed{seed}';models.mkdir(parents=True,exist_ok=False)
            cfg=cfg_for(PROTOCOL['train']);hyper=json.loads((ROOT/'configs/phase2c1_ppo.yaml').read_text(encoding="utf-8"));hyper.update(log_std_init=-1.5,training_seeds=[42,123],cumulative_stage_decisions=[256,2048,4096],record_training_episodes=[2,8,16,32])
            env=MovingResidualEnv(folder/'actual_training',config=cfg,record_episodes=hyper['record_training_episodes'] if seed==42 else [],mode='Actual Training');kwargs={k:hyper[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','vf_coef','max_grad_norm','target_kl']}
            model=TracedPPO('MlpPolicy',env,seed=seed,device='cpu',policy_kwargs=dict(net_arch=hyper['net_arch'],activation_fn=torch.nn.Tanh,log_std_init=hyper['log_std_init']),tensorboard_log=str(folder/'tensorboard'),verbose=0,**kwargs)
            with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
            model.attach(env,folder,models);model.save(models/'initial.zip');write_json(folder/'config_frozen.json',dict(seed=seed,env=cfg,ppo=hyper,initial_parameter_hash=parameter_hash(model.policy),std_reason='Initial std0.2231 reduces noise relative to old pilot; authority still separately tested at extremes',zero_equivalence='Original four_d_lead exact; separately tested2200steps',budget=4096,joint_gate=PROTOCOL['continuation_gate']))
            jobs.append(dict(seed=seed,folder=folder,models=models,model=model,env=env,callback=SamplingCallback(env,hyper['stop_rules']),rng=rng_get(),started=time.perf_counter(),stages=[]))
        for target in [256,2048,4096]:
            checks=[]
            for j in jobs:
                rng_set(j['rng']);m=j['model'];e=j['env'];m.learn(total_timesteps=target-m.num_timesteps,reset_num_timesteps=False,callback=j['callback'],tb_log_name=f'seed{j["seed"]}');j['rng']=rng_get();ck=j['models']/f'stage_{target}.zip';m.save(ck)
                with preserve_rng():
                    load=TracedPPO.load(ck);assert parameter_hash(load.policy)==parameter_hash(m.policy)
                    result=dict(target=target,actual_decisions=m.num_timesteps,physical_steps=e.total_physical_steps,reload_exact=True,stop=j['callback'].stop_reason,passed=not bool(j['callback'].stop_reason))
                    if target>=2048:
                        r=episode(j['folder']/f'validation_{target}',PROTOCOL['validation'],model=m);b=json.loads((RUN/'baselines/V_move/four_d_lead/metrics.json').read_text(encoding="utf-8"));ratio=r['z_rmse_mm']/b['z_rmse_mm'];xy=r['xy_peak_mm']/b['xy_peak_mm']
                        result.update(validation=r,z_ratio=ratio,xy_ratio=xy,passed=bool(result['passed'] and r['success'] and not r['native_contact_observations'] and ratio<=1.10 and xy<=1.10))
                    checks.append(result);j['stages'].append(result);write_json(j['folder']/f'stage_{target}.json',result)
            passed=all(x['passed'] for x in checks);write_json(RUN/f'joint_{target}_gate.json',dict(passed=passed,checks=checks,rule=PROTOCOL['continuation_gate']));print('JOINT GATE',target,passed,flush=True)
            if not passed:break
        # Lock all final checkpoints before any final task execution.
        frozen=[]
        for j in jobs:
            m=j['model'];e=j['env'];m.save(j['models']/'final.zip');frozen.append(dict(seed=j['seed'],checkpoint=str(j['models']/'final.zip'),sha256=sha(j['models']/'final.zip'),parameter_hash=parameter_hash(m.policy),decisions=m.num_timesteps))
            write_json(j['folder']/'training_result.json',dict(seed=j['seed'],decisions=m.num_timesteps,physical_steps=e.total_physical_steps,updates=len(m._phase_updates),optimizer_steps=m._phase_updates[-1]['optimizer_steps_total'],stages=j['stages'],wall_seconds=time.perf_counter()-j['started'],protected_episodes=sum(bool(x['failure']) for x in e.episode_summaries),final_parameter_hash=parameter_hash(m.policy)))
        write_json(RUN/'final_evaluation_frozen.json',dict(models=frozen,protocol_sha256=sha(RUN/'protocol_frozen.json'),selection='Last common budget; no best-test selection',opened=False))
    except Exception as exc:
        write_json(RUN/'training_exception.json',dict(error=repr(exc),jobs=[dict(seed=j['seed'],decisions=j['model'].num_timesteps,physical_steps=j['env'].total_physical_steps) for j in jobs]));raise
    finally:
        for j in jobs:j['model'].finish_trace();j['env'].close()
if __name__=='__main__':main()
