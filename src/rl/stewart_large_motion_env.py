"""50mm extension of original Gym/owner/FF/governor/recording architecture."""
import copy,csv,gc,json
from pathlib import Path
import gymnasium as gym,numpy as np
from gymnasium import spaces
from .stewart_residual_env import StewartResidualEnv
from .large_motion_observations import LargeObservation
from .large_motion_rewards import costs,failure_penalty,CONFIG
from .large_motion_action_adapter import LargeMotionAction
from ..workspace.feedforward_study import make_owner
from ..disturbance.disturbance_profiles import DisturbanceProfile
from ..model import ROOT,write_json

RUN=ROOT/'results/runs/phase2c4_20261105_v1'
FIXED=dict(id='train_fixed',rise_s=6.,tau_s=.005,Fz_N=-5.5,on_s=5.2,off_s=11.)

def large_owner(folder,task,record=False,method='FF-VA'):
    o,r=make_owner(folder,method,.05,task['rise_s'],task['tau_s'],task['Fz_N'],record)
    o.learning_reference=r
    o.profile=DisturbanceProfile(start_step=round(task['on_s']*1000),stop_step=round(task['off_s']*1000),wrench=[0,0,task['Fz_N'],0,0,0],frame='world',kind='step');o.config.update(stage='phase2c4',task=task,disturbance=o.profile.description());write_json(Path(folder)/'config.json',o.config);write_json(Path(folder)/'disturbance_profile.json',o.profile.description());return o,r

class StewartLargeMotionResidualEnv(StewartResidualEnv):
    def __init__(self,output_root,tasks=None,record_episodes=(),history_length=1,mode='Actual Training'):
        # Keep parent bookkeeping/stream/close contract; replace its85-dim frame.
        cfg=dict(history_length=history_length,training_tasks=tasks or [FIXED],decision_physical_steps=50,episode_physical_steps=15000,action_scale_N=.1,reward=copy.deepcopy(CONFIG),measurement_boundary='Ideal measured platform/chain state; known command and causal observer only; no drive/tau/load truth')
        super().__init__(output_root,tasks=tasks or [FIXED],config=cfg,record_episodes=record_episodes,history_length=history_length,mode=mode);self.observation_space=spaces.Box(-10,10,(53*history_length,),np.float32);self.completed_costs=[]
    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        if self.owner is not None:self._finish('reset_before_done',False)
        self.episode+=1;self.decision=0;self.episode_return=0.;self.done=False;self.policy_context={};self.current_task=copy.deepcopy(next(x for x in self.tasks if x['id']==options['task_id']) if options and options.get('task_id') else self.tasks[int(self.np_random.integers(len(self.tasks)))])
        folder=self.output_root/f'episode_{self.episode:04d}';self.owner,self.reference=large_owner(folder,self.current_task,self.episode in self.record_episodes);self.owner.episode=self.episode;self.action_adapter=LargeMotionAction(self.owner);self.horizon_steps=round(self.reference.duration*1000);self.context={};self.last_cost={k:0. for k in ['z','xy','rotation','action','slew']};self.episode_metrics=[]
        if self.episode in self.record_episodes:
            from ..disturbance.recording import install_colored_capture
            install_colored_capture(self.owner.recorder,self.owner.model)
        append=self.owner.recorder.append
        def log(row,state):
            a=self.owner.residual.injection.audit
            for j,x in enumerate(a.get('total_add_rate_N_s',np.zeros(6))):row[f'ff_total_add_rate_N_s_{j}']=float(x)
            parts,factor=costs(row,self.action_adapter.last*.1);self.last_cost=parts
            for k,x in parts.items():row['reward_cost_'+k]=x
            row.update(reward_cap_factor=factor,rl_decision=self.decision,rl_global_decision=self.global_decisions,policy_version=self.policy_context.get('policy_version',-1),source_label=self.mode,rl_actor_raw=self.policy_context.get('raw_action',self.action_adapter.last),rl_actor_mean=self.policy_context.get('actor_mean',self.action_adapter.last),rl_actor_std=self.policy_context.get('actor_std',0.),rl_clipped_action=self.action_adapter.last,rl_requested_Fz_N=self.action_adapter.last*.1)
            effect=a.get('probe_effect_command',np.zeros(6));row['rl_accepted_Fz_N']=float((self.owner.residual.base.jacobian.T@effect)[2]);append(row,state);self.last_row=row.copy()
        self.owner.recorder.append=log;self.observer=LargeObservation(self.history_length);self.obs=self.observer.reset(self.owner);self.episode_obs=[];self.episode_actions=[];self.episode_next=[];self.episode_rewards=[]
        return self.obs.copy(),dict(episode=self.episode,task_id=self.current_task['id'])
    def step(self,action):
        if self.owner is None or self.done:raise RuntimeError('reset required')
        self.action_adapter.set(action);a=self.action_adapter.last;previous=self.obs.copy();self.decision+=1;self.global_decisions+=1;start=self.owner.n;aggregate={k:0. for k in self.last_cost};failure=False;active=0;hold=0;series=[]
        for _ in range(50):
            self.owner.step();active+=1;self.total_physical_steps+=1;series.append(self.last_row)
            for k,x in self.last_cost.items():aggregate[k]-=x*.001/.05
            if self.owner.failure:failure=True;break
            if self.owner.n>=self.horizon_steps:break
        if failure:
            while self.owner.data.time<self.owner.protect_end-1e-10:
                self.owner.step();hold+=1;self.total_physical_steps+=1;series.append(self.last_row)
                for k,x in self.last_cost.items():aggregate[k]-=x*.001/.05
        penalty=failure_penalty(int(np.ceil(max(0,self.horizon_steps-self.owner.n)/50))) if failure else 0.;reward=float(sum(aggregate.values())+penalty);terminated=bool(failure);truncated=bool(not failure and self.owner.n>=self.horizon_steps);self.done=terminated or truncated;self.episode_return+=reward;self.obs=self.observer.append(self.owner)
        accepted=np.array([x['rl_accepted_Fz_N'] for x in series]);ep=np.array([[x[f'p_{j}']-x[f'p_ref_{j}'] for j in range(3)] for x in series]);angles=np.array([x['orientation_error_deg'] for x in series]);projected=np.array([any(abs(x[f'ff_desired_total_add_N_{j}']-x[f'ff_actual_total_add_N_{j}'])>1e-9 for j in range(6)) for x in series]);self.episode_metrics.extend((float(x['time_s']),float(e[2]),float(np.sum(e[:2]**2)),float(x['orientation_error_deg'])) for x,e in zip(series,ep))
        row=dict(episode=self.episode,decision=self.decision,global_decision=self.global_decisions,policy_version=self.policy_context.get('policy_version',-1),physical_start_step=start,physical_end_step=self.owner.n,action_physical_steps=active,hold_physical_steps=hold,physical_time_s=float(self.owner.data.time),raw_action=float(self.policy_context.get('raw_action',a)),clipped_action=a,actor_mean=float(self.policy_context.get('actor_mean',a)),actor_std=float(self.policy_context.get('actor_std',0)),mapped_Fz_N=a*.1,accepted_residual_mean_N=float(accepted.mean()),accepted_residual_rms_N=float(np.sqrt(np.mean(accepted**2))),action_clip_fraction=float(abs(float(self.policy_context.get('raw_action',a)))>1),budget_projection_fraction=float(projected.mean()),Z_error_rmse=float(np.sqrt(np.mean(ep[:,2]**2))*1000),XY_error_rmse=float(np.sqrt(np.mean(np.sum(ep[:,:2]**2,axis=1)))*1000),SO3_error=float(np.sqrt(np.mean(angles**2))),reward=reward,episode_reward=self.episode_return,failure_penalty=penalty,terminated=terminated,truncated=truncated,reason=self.owner.failure or ('time_limit' if truncated else ''),source=self.mode,task_id=self.current_task['id'],**{'reward_'+k:v for k,v in aggregate.items()})
        self.records.append(row);self.episode_obs.append(previous);self.episode_actions.append([a]);self.episode_next.append(self.obs.copy());self.episode_rewards.append(reward)
        if self.writer is None:self.writer=csv.DictWriter(self.stream,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.stream.flush();info=dict(physical_steps=active+hold,global_physical_steps=self.total_physical_steps,reward_components=dict(aggregate,failure=penalty),failure=self.owner.failure)
        if self.done:
            info['episode']={'r':self.episode_return,'l':self.decision};self._finish('protected_failure' if failure else 'time_limit',True);info['is_success']=self.episode_summaries[-1]['success']
        return self.obs.copy(),reward,terminated,truncated,info
    def _finish(self,reason,transition_done):
        if self.owner is None:return
        o=self.owner;folder=o.recorder.folder;complete=bool(transition_done and not o.failure and o.n==self.horizon_steps);a=np.array(self.episode_metrics);windows={}
        for name,t in [('load',self.current_task['on_s']),('unload',self.current_task['off_s'])]:windows[name+'_window_IAE_mm_s']=float(np.sum(abs(a[(a[:,0]>=t)&(a[:,0]<t+1),1]))*.001*1000) if len(a) else 0.
        tail=a[a[:,0]>=self.reference.duration-.5] if len(a) else np.empty((0,4));success=bool(complete and len(tail)>=500 and np.max(np.sqrt(tail[:,1]**2+tail[:,2]))*1000<.75 and np.max(tail[:,3])<.1)
        summary=dict(episode=self.episode,folder=str(folder),reason=reason,physical_steps=o.n,decisions=self.decision,return_=self.episode_return,complete=complete,success=success,failure=o.failure,task_id=self.current_task['id'],observation_clip_fraction=self.observer.clips/max(self.observer.elements,1),observation_peak=self.observer.peak,**windows)
        o.recorder.close(dict(**summary,actual_training=self.mode=='Actual Training',no_live_qpos_assignment=True));np.savez_compressed(folder/'decisions.npz',observations=np.asarray(self.episode_obs),actions=np.asarray(self.episode_actions),next_observations=np.asarray(self.episode_next),rewards=np.asarray(self.episode_rewards));self.episode_summaries.append(summary);write_json(self.output_root/'episode_summary.json',self.episode_summaries);self.owner=None;self.action_adapter=None;del o;gc.collect()
