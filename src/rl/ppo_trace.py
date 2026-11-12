"""SB3 PPO instrumentation: records existing samples, never resamples a policy."""
import csv,time,types,hashlib
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ..model import write_json,sha

def parameters(policy,kind):
    result=[]
    for name,p in policy.named_parameters():
        actor=name.startswith(('action_net','log_std','mlp_extractor.policy_net'))
        if actor==(kind=='actor'):result.append(p.detach().cpu().flatten())
    return torch.cat(result)

def parameter_hash(policy):
    h=hashlib.sha256()
    for name,p in policy.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def optimizer_steps(policy):return max([int(float(x['step'])) for x in policy.optimizer.state.values() if 'step' in x] or [0])

class TracedPPO(PPO):
    def _excluded_save_params(self):return super()._excluded_save_params()+[k for k in self.__dict__ if k.startswith('_phase_')]
    def attach(self,env,folder,checkpoint_folder):
        self._phase_env=env;self._phase_folder=Path(folder);self._phase_checkpoints=Path(checkpoint_folder);self._phase_checkpoints.mkdir(parents=True,exist_ok=True);self._phase_update_index=0;self._phase_collecting=False;self._phase_updates=[];self._phase_last_mean=0.;self._phase_stream=(self._phase_folder/'ppo_update_events.csv').open('w',newline='',encoding='utf-8');self._phase_writer=None
        def mean_hook(module,args,output):
            if self._phase_collecting:self._phase_last_mean=float(output.detach().cpu()[0,0])
        self._phase_hook=self.policy.action_net.register_forward_hook(mean_hook);original=self.policy.forward
        def traced_forward(policy_self,obs,deterministic=False):
            result=original(obs,deterministic=deterministic)
            if self._phase_collecting:
                self._phase_env.policy_context=dict(raw_action=float(result[0].detach().cpu()[0,0]),actor_mean=self._phase_last_mean,actor_std=float(torch.exp(self.policy.log_std.detach()).cpu()[0]),critic_prediction=float(result[1].detach().cpu()[0,0]),policy_version=self._phase_update_index)
            return result
        self.policy.forward=types.MethodType(traced_forward,self.policy)
    def train(self):
        if not hasattr(self,'_phase_env'):return super().train()
        self._phase_collecting=False;before_a=parameters(self.policy,'actor').clone();before_v=parameters(self.policy,'critic').clone();before_steps=optimizer_steps(self.policy);phys=self._phase_env.total_physical_steps;idx=self._phase_update_index+1;buf=self.rollout_buffer
        physical_rewards=np.array([x['reward'] for x in self._phase_env.records[-buf.buffer_size:]],np.float64)
        np.savez_compressed(self._phase_folder/f'rollout_{idx:03d}.npz',observations=buf.observations.copy(),actions=buf.actions.copy(),effective_rewards=buf.rewards.copy(),physical_rewards=physical_rewards,values=buf.values.copy(),log_probs=buf.log_probs.copy(),advantages=buf.advantages.copy(),returns=buf.returns.copy(),episode_starts=buf.episode_starts.copy(),last_values=self._phase_last_values,global_decision_end=self.num_timesteps)
        print('UPDATING ACTOR / CRITIC',idx,'decision',self.num_timesteps,flush=True);start=time.perf_counter();super().train();elapsed=time.perf_counter()-start;assert self._phase_env.total_physical_steps==phys
        self._phase_update_index=idx;values=self.logger.name_to_value;row=dict(update=idx,global_decisions=self.num_timesteps,physical_steps=phys,actor_change_norm=float(torch.linalg.vector_norm(parameters(self.policy,'actor')-before_a)),critic_change_norm=float(torch.linalg.vector_norm(parameters(self.policy,'critic')-before_v)),optimizer_steps=optimizer_steps(self.policy)-before_steps,optimizer_steps_total=optimizer_steps(self.policy),optimization_wall_s=elapsed,collection_wall_s=self._phase_collect_wall,advantage_mean=float(buf.advantages.mean()),advantage_std=float(buf.advantages.std()),advantage_min=float(buf.advantages.min()),advantage_max=float(buf.advantages.max()),value_target_mean=float(buf.returns.mean()),value_target_std=float(buf.returns.std()),policy_loss=float(values['train/policy_gradient_loss']),value_loss=float(values['train/value_loss']),approx_kl=float(values['train/approx_kl']),entropy_loss=float(values['train/entropy_loss']),clip_fraction=float(values['train/clip_fraction']),explained_variance=float(values['train/explained_variance']),action_std=float(torch.exp(self.policy.log_std.detach())[0]),parameter_hash=parameter_hash(self.policy),simulator_paused_during_update=True)
        finite=[v for k,v in row.items() if isinstance(v,float) and k!='explained_variance'];assert np.isfinite(finite).all() and row['actor_change_norm']>0 and row['critic_change_norm']>0
        if not np.isfinite(row['explained_variance']):row['explained_variance']=None
        checkpoint=self._phase_checkpoints/f'update_{idx:03d}_{self.num_timesteps:06d}.zip';self.save(checkpoint);row.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        if self._phase_writer is None:self._phase_writer=csv.DictWriter(self._phase_stream,fieldnames=list(row));self._phase_writer.writeheader()
        self._phase_writer.writerow(row);self._phase_stream.flush();self._phase_updates.append(row);write_json(self._phase_folder/'update_events.json',self._phase_updates)
        print('UPDATE DONE',idx,'actor_delta',row['actor_change_norm'],'critic_delta',row['critic_change_norm'],flush=True)
    def finish_trace(self):
        self._phase_collecting=False;self._phase_stream.close();self._phase_hook.remove()

class SamplingCallback(BaseCallback):
    def __init__(self,env,stop_rules):super().__init__();self.real_env=env;self.rules=stop_rules;self.stop_reason=None
    def _on_rollout_start(self):self.model._phase_collecting=True;self.started=time.perf_counter();print('COLLECTING EXPERIENCE','from decision',self.model.num_timesteps,flush=True)
    def _on_step(self):
        row=self.real_env.records[-1];raw=float(self.locals['actions'][0,0]);assert abs(row['raw_action']-raw)<1e-7
        if not np.isfinite([row['reward'],row['raw_action']]).all():self.stop_reason='nonfinite_sample';return False
        recent=self.real_env.episode_summaries[-3:]
        if sum(bool(x['failure']) for x in recent)>=self.rules['protected_failures_in_last3_at_least']:self.stop_reason='persistent_protection_failures';return False
        return True
    def _on_rollout_end(self):
        self.model._phase_collecting=False;self.model._phase_collect_wall=time.perf_counter()-self.started;self.model._phase_last_values=self.locals['values'].detach().cpu().numpy().copy()
