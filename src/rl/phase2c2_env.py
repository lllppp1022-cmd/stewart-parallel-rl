"""Independent moving-task extension. Original owner step, safety and PPO sampler unchanged."""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from .stewart_residual_env import StewartResidualEnv,TrainingOwner,owner_config
from .observations import HistoryObservation
from .moving_reference import MovingReference
from .action_adapter import LearningCorrection
from ..control.residual_action import ResidualController
from ..control.phase2c0_actions import TaskDiagnostic
from ..control.residual_governor import SharedBudgetGovernor
from ..control.phase2b2_experiment import MeasurementAdapter
from ..model import ROOT,write_json

class StrongBaseAction:
    privileged=False;space='leg';frequency=1000
    def __init__(self,core,scale,baseline='four_d_lead'):
        self.scale_N=float(scale);self.normalized=0.;self.calls=0;self.c4_leg=np.zeros(6);self.base=TaskDiagnostic(dict(kind='four_d_lead'),core);self.baseline=baseline;self.clipped=0
    def set_action(self,value):
        a=np.asarray(value,float)
        if a.shape!=(1,) or not np.isfinite(a).all():raise ValueError('One finite normalized action required')
        self.normalized=float(np.clip(a[0],-1,1))
    def request(self,o,j):
        self.calls+=1
        if self.baseline=='four_d_lead':self.c4_leg=self.base.request(o,j)
        else:
            k=self.base.calls;self.base.calls+=1;w=np.zeros(6);w[5]=np.clip(-o.equivalent_disturbance[5],-.3,.3)
            if self.baseline=='fz_lead':
                if k%50==0:
                    e=o.equivalent_disturbance[2];self.base.held[2]=0 if self.base.last_estimate is None else np.clip(-.02*(e-self.base.last_estimate)*20,-.5,.5);self.base.last_estimate=float(e)
                w[2]=self.base.held[2]
            elif self.baseline!='c4':raise ValueError('Unknown baseline')
            self.c4_leg=np.clip(np.linalg.solve(j.T,w),-.95,.95)
        if self.normalized==0:return self.c4_leg.copy()
        w=np.zeros(6);w[2]=self.normalized*self.scale_N;raw=self.c4_leg+np.linalg.solve(j.T,w);self.clipped+=int(np.any(abs(raw)>1))
        return np.clip(raw,-1,1)

class StrongLearningCorrection(LearningCorrection):
    def __init__(self,core,scale,baseline):
        self.held_action=StrongBaseAction(core,scale,baseline);ResidualController.__init__(self,core,self.held_action);self.name='C4+'+baseline+'+PPO-Fz';self.c4_same_state_command=np.zeros(6);self.rl_applied_command=np.zeros(6);self.comparison_governor=SharedBudgetGovernor()

def moving_owner(config,task,envcfg):
    owner=TrainingOwner(config,envcfg['action_scale_N'],envcfg['reward']);core=owner.residual.base
    owner.residual=StrongLearningCorrection(core,envcfg['action_scale_N'],envcfg.get('baseline','four_d_lead'));owner.controller=MeasurementAdapter(owner.residual)
    owner.active=MovingReference(owner.nominal,task['amplitude_m'],envcfg['reference_cache'],task.get('shift_s',0));owner.ref=owner.active.at(0);owner.active_start=0
    owner.config.update(stage='phase2c2',trajectory='smooth_z_return',motion_task=task,baseline=envcfg.get('baseline','four_d_lead'))
    write_json(Path(config['folder'])/'config.json',owner.config)
    return owner

class MovingResidualEnv(StewartResidualEnv):
    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        if self.owner is not None:self._finish('reset_before_done',False)
        self.episode+=1;self.decision=0;self.episode_return=0.;self.done=False;self.policy_context={};self.current_task=dict(next(t for t in self.tasks if t['id']==options['task_id']) if options and options.get('task_id') else self.tasks[int(self.np_random.integers(len(self.tasks)))])
        folder=self.output_root/f'episode_{self.episode:04d}';record=self.episode in self.record_episodes
        self.owner=moving_owner(owner_config(folder,self.current_task,record),self.current_task,self.config);self.owner.context=dict(mode=self.mode)
        # Display/journal identity only; never enters observations or control.
        self.owner.episode=self.episode
        if record:
            from .phase2c2_recording import install_capture
            install_capture(self.owner)
        self.observer=HistoryObservation(self.config['observation_scales'],self.history_length);s=self.owner.measured();self.obs=self.observer.reset(s,np.array(self.owner.home_baseline['p_m']),self.owner.geometry.home_rotation,self.owner.home.at(0).q,self.owner.home.at(0).ff)
        self.episode_obs=[];self.episode_actions=[];self.episode_next=[];self.episode_rewards=[]
        return self.obs.copy(),dict(episode=self.episode,measurement_scope=self.config['measurement_boundary'])
