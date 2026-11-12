"""Gymnasium20Hz residual environment, original1kHz MuJoCo owner underneath."""
import json,csv,time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from .action_adapter import LearningCorrection
from .observations import MeasuredState,HistoryObservation
from .rewards import physical_cost,failure_penalty
from ..model import ROOT,write_json
from ..phase1.plant import platform_state
from ..control.phase2c0_experiment import PhysicalOwner
from ..control.phase2b2_experiment import MeasurementAdapter
from ..disturbance.recording import install_colored_capture

def env_config():return json.loads((ROOT/'configs/phase2c1_env.yaml').read_text(encoding='utf-8'))
def owner_config(folder,task,record=False):
    model=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'))
    return dict(folder=str(folder),controller='C3',yaw_policy=dict(kind='observer',gain=1),task_model=model,bandwidth_rad_s=30,compensation_gain=1,leak_per_s=1,condition='nominal',record=record,pace=False,max_sim_seconds=6.,max_wall_seconds=180,exit_on_failure=True,stage='phase2c1',task=task['id'],trajectory='home',disturbance=dict(start_step=task['on_step'],stop_step=task['off_step'],wrench=[0,0,task['Fz_N'],0,0,0],frame='world',kind='step'),actuator_dynamics=dict(enabled=True,tau_s=task['tau_s']))

class TrainingOwner(PhysicalOwner):
    def __init__(self,config,scale,reward_cfg):
        super().__init__(config);core=self.residual.base;self.residual=LearningCorrection(core,scale);self.controller=MeasurementAdapter(self.residual);self.reward_cfg=reward_cfg;self.action_scale=scale;self.context={};self.previous_rl=np.zeros(6);self.last_row=None;self.last_cost={};self.last_c4=self.home.at(0).ff.copy();append=self.recorder.append
        def log(row,state):
            p=np.array([row[f'p_{j}']-row[f'p_ref_{j}'] for j in range(3)]);requested=self.requested_rl_fz if self.rl_was_active else 0.;parts,factor=physical_cost(p,np.deg2rad(row['orientation_error_deg']),self.rl_effect,self.previous_rl,requested,self.action_scale,self.reward_cfg);self.previous_rl=self.rl_effect.copy();self.last_cost=parts
            for key,value in [('rl_applied_command',self.rl_effect),('rl_applied_command_wrench',self.rl_effect_wrench),('c4_same_state_command',self.last_c4),('rl_mapped_target4',np.array([requested,0,0,0]))]:
                for j,x in enumerate(value):row[f'{key}_{j}']=float(x)
            for key,value in parts.items():row['reward_cost_'+key]=value
            row.update(reward_cap_factor=factor,rl_active=self.rl_was_active,rl_normalized_action=self.residual.held_action.normalized,rl_actor_raw=float(self.context.get('raw_action',self.residual.held_action.normalized)),rl_actor_mean=float(self.context.get('actor_mean',self.residual.held_action.normalized)),rl_actor_std=float(self.context.get('actor_std',0)),rl_decision=self.context.get('decision',-1),rl_global_decision=self.context.get('global_decision',-1),policy_version=self.context.get('policy_version',-1),source_label=self.context.get('mode','Baseline Comparison'))
            append(row,state);self.last_row=row.copy()
        self.recorder.append=log
    def apply_actuator_command(self,command):
        self.rl_was_active=not self.failure;self.rl_effect=self.residual.rl_applied_command.copy() if self.rl_was_active else np.zeros(6);self.rl_effect_wrench=self.residual.base.jacobian.T@self.rl_effect;self.last_c4=self.residual.c4_same_state_command.copy() if self.rl_was_active else np.asarray(command).copy();self.requested_rl_fz=self.residual.held_action.normalized*self.action_scale
        return super().apply_actuator_command(command)
    def measured(self):
        s=self.controller.sensor_trace
        if s:p=np.array(s['measured_p']);r=np.array(s['measured_rotation']).reshape(3,3);v=np.array(s['measured_velocity'])
        else:p,r,v=platform_state(self.model,self.data)
        estimate=np.asarray(self.residual.base.diagnostics.get('equivalent_disturbance',np.zeros(6))).copy()
        return MeasuredState(p.copy(),r.copy(),v.copy(),self.data.qpos[self.qadr].copy(),self.data.qvel[self.dof].copy(),self.ref.p.copy(),self.ref.r.copy(),self.ref.twist.copy(),self.last_c4.copy(),estimate,self.previous_rl.copy())

class StewartResidualEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,output_root,tasks=None,config=None,record_episodes=(),history_length=None,mode='Actual Training'):
        super().__init__();self.config=env_config() if config is None else config;self.tasks=tasks or self.config['training_tasks'];self.output_root=Path(output_root);self.output_root.mkdir(parents=True,exist_ok=False);self.history_length=history_length or self.config['history_length'];self.action_space=spaces.Box(-1,1,(1,),np.float32);self.observation_space=spaces.Box(-10,10,(85*self.history_length,),np.float32);self.record_episodes=set(record_episodes);self.mode=mode;self.owner=None;self.episode=-1;self.global_decisions=0;self.total_physical_steps=0;self.records=[];self.episode_summaries=[];self.policy_context={};self.closed=False
        self.decision=0;self.stream=(self.output_root/'training_samples.csv').open('w',newline='',encoding='utf-8');self.writer=None;write_json(self.output_root/'env_config.json',self.config)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if self.owner is not None:self._finish('reset_before_done',False)
        self.episode+=1;self.decision=0;self.episode_return=0.;self.done=False;self.policy_context={};self.current_task=dict(next(t for t in self.tasks if t['id']==options['task_id']) if options and options.get('task_id') else self.tasks[int(self.np_random.integers(len(self.tasks)))])
        folder=self.output_root/f'episode_{self.episode:04d}';self.owner=TrainingOwner(owner_config(folder,self.current_task,self.episode in self.record_episodes),self.config['action_scale_N'],self.config['reward']);self.owner.context=dict(mode=self.mode)
        if self.episode in self.record_episodes:install_colored_capture(self.owner.recorder,self.owner.model)
        self.observer=HistoryObservation(self.config['observation_scales'],self.history_length);state=self.owner.measured();self.obs=self.observer.reset(state,self.owner.home_baseline['p_m'] if isinstance(self.owner.home_baseline['p_m'],np.ndarray) else np.array(self.owner.home_baseline['p_m']),self.owner.geometry.home_rotation,self.owner.home.at(0).q,self.owner.home.at(0).ff);self.episode_obs=[];self.episode_actions=[];self.episode_next=[];self.episode_rewards=[]
        return self.obs.copy(),dict(episode=self.episode,measurement_scope=self.config['measurement_boundary'])
    def step(self,action):
        if self.owner is None or self.done:raise RuntimeError('reset required before interaction')
        a=np.asarray(action,float)
        if a.shape!=(1,) or not np.isfinite(a).all():raise ValueError('Finite one-dimensional action required')
        a=np.clip(a,-1,1);previous_obs=self.obs.copy();self.decision+=1;self.global_decisions+=1;self.owner.residual.held_action.set_action(a);self.owner.context=dict(self.policy_context,decision=self.decision,global_decision=self.global_decisions,mode=self.mode);cost={k:0. for k in ['z','xy','angle','amplitude','slew','request']};measurements=[];start=self.owner.n;failure_time=None;action_steps=0
        for _ in range(self.config['decision_physical_steps']):
            self.owner.step();action_steps+=1;self.total_physical_steps+=1;measurements.append(self.owner.measured())
            for k in cost:cost[k]-=self.owner.last_cost[k]*.001/.05
            if self.owner.failure:failure_time=float(self.owner.data.time);break
            if self.owner.n>=self.config['episode_physical_steps']:break
        hold_steps=0
        if failure_time is not None:
            while self.owner.data.time<self.owner.protect_end-1e-10:
                self.owner.step();hold_steps+=1;self.total_physical_steps+=1;measurements.append(self.owner.measured())
                for k in cost:cost[k]-=self.owner.last_cost[k]*.001/.05
        penalty=failure_penalty(failure_time,self.config['reward']) if failure_time is not None else 0.;reward=float(sum(cost.values())+penalty);terminated=bool(failure_time is not None);truncated=bool(not terminated and self.owner.n>=self.config['episode_physical_steps']);self.done=terminated or truncated;self.episode_return+=reward;self.obs=self.observer.append(self.owner.measured(),float(a[0]),measurements)
        row=dict(episode=self.episode,decision=self.decision,global_decision=self.global_decisions,policy_version=self.policy_context.get('policy_version',-1),physical_start_step=start,physical_end_step=self.owner.n,action_physical_steps=action_steps,hold_physical_steps=hold_steps,physical_time_s=float(self.owner.data.time),raw_action=float(self.policy_context.get('raw_action',a[0])),clipped_action=float(a[0]),actor_mean=float(self.policy_context.get('actor_mean',a[0])),actor_std=float(self.policy_context.get('actor_std',0.)),mapped_Fz_N=float(a[0]*self.config['action_scale_N']),reward=reward,episode_reward=self.episode_return,failure_penalty=penalty,terminated=terminated,truncated=truncated,reason=self.owner.failure or ('time_limit' if truncated else ''),source=self.mode,task_id=self.current_task['id'],**{'reward_'+k:v for k,v in cost.items()})
        self.records.append(row);self.episode_obs.append(previous_obs);self.episode_actions.append(a.copy());self.episode_next.append(self.obs.copy());self.episode_rewards.append(reward)
        if self.writer is None:self.writer=csv.DictWriter(self.stream,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.stream.flush()
        info=dict(episode_id=self.episode,decision_id=self.decision,physical_steps=action_steps+hold_steps,action_physical_steps=action_steps,hold_physical_steps=hold_steps,global_physical_steps=self.total_physical_steps,reward_components=dict(cost,failure=penalty),failure=self.owner.failure,measurement_scope=self.config['measurement_boundary'])
        if self.done:info['episode']={'r':self.episode_return,'l':self.decision};self._finish('protected_failure' if terminated else 'time_limit',True)
        return self.obs.copy(),reward,terminated,truncated,info
    def _finish(self,reason,transition_done):
        if self.owner is None:return
        owner=self.owner;folder=owner.recorder.folder;owner.recorder.close(dict(episode=self.episode,physics_steps=owner.n,simulation_time_s=float(owner.data.time),terminal_reason=reason,failure=owner.failure,controller=owner.controller.name,complete=bool(transition_done and not owner.failure and owner.n==6000),rl_decisions=self.decision,episode_reward=self.episode_return,actual_training=self.mode=='Actual Training',no_live_qpos_assignment=True))
        np.savez_compressed(folder/'decisions.npz',observations=np.asarray(self.episode_obs),actions=np.asarray(self.episode_actions),next_observations=np.asarray(self.episode_next),rewards=np.asarray(self.episode_rewards));self.episode_summaries.append(dict(episode=self.episode,folder=str(folder),reason=reason,physical_steps=owner.n,decisions=self.decision,return_=self.episode_return,complete=bool(transition_done and not owner.failure and owner.n==6000),failure=owner.failure));write_json(self.output_root/'episode_summary.json',self.episode_summaries);self.owner=None
    def close(self):
        if self.closed:return
        self._finish('budget_cutoff' if self.decision else 'closed_without_interaction',False);self.stream.close();self.closed=True;write_json(self.output_root/'interaction_totals.json',dict(decisions=self.global_decisions,physical_steps=self.total_physical_steps,episodes=self.episode_summaries))
