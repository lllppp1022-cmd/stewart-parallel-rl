"""Same Phase7 physics/step/observations; independent task RNG and reward hook."""
import copy
import gymnasium as gym
import numpy as np
from .stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv,DynamicMomentPort
from .phase2c8_reward import costs,NAMES
from ..workspace.dynamic_attitude_study import make_owner
from ..model import write_json

class TaskSampler:
    def __init__(self,seed,state=None):
        self.seed=int(seed);self.rng=np.random.default_rng(np.random.SeedSequence([self.seed,8008]));self.draws=0
        if state is not None:self.restore(state)
    def snapshot(self):return dict(seed=self.seed,draws=self.draws,bit_generator=copy.deepcopy(self.rng.bit_generator.state))
    def restore(self,state):
        assert int(state['seed'])==self.seed;self.draws=int(state['draws']);self.rng.bit_generator.state=copy.deepcopy(state['bit_generator'])
    def choose(self,tasks):
        before=self.snapshot();task=copy.deepcopy(tasks[int(self.rng.integers(len(tasks)))]);self.draws+=1
        return task,before,self.snapshot()

class Phase2C8Env(StewartDynamicAttitudeEnv):
    def __init__(self,root,tasks,*,seed=42,version='v3',sampler_state=None,record_episodes=(),source='ACTUAL STOCHASTIC PHASE2C8 TRAINING'):
        super().__init__(root,tasks,record_episodes,source)
        if version not in ('v2_control','v3','v3_return_norm'):raise ValueError(version)
        self.version=version;self.sampler=TaskSampler(seed,sampler_state);self.task_events=[]
    def reset(self,*,seed=None,options=None):
        # Gym/VecEnv may reseed their own RNG. It never owns task selection here.
        gym.Env.reset(self,seed=seed);self.finish('reset_incomplete');self.episode+=1
        self.task,before,after=self.sampler.choose(self.tasks)
        self.task_events.append(dict(episode=self.episode,global_decision=self.global_decisions,task=self.task,requested_gym_seed=seed,sampler_before=before,sampler_after=after))
        write_json(self.root/'task_sequence.json',self.task_events)
        self.owner,self.reference=make_owner(self.root/f'episode_{self.episode:04d}',self.task,self.episode in self.record_episodes)
        self.horizon=round(self.reference.duration*1000);self.owner.episode=self.episode
        injection=self.owner.residual.injection;self.port=DynamicMomentPort(injection.original,self.owner.residual.base);injection.original=self.port
        self.decision=self.clip_count=self.obs_count=0;self.episode_return=0.;self.previous_accepted=np.zeros(2);self.context={};self.done=False;self.last_row=None
        self.physical_diagnostics=dict(peak_so3_error=0.,load_window_IAE=0.,unload_window_IAE=0.,cap_steps=0)
        append=self.owner.recorder.append
        def log(row,state):
            accepted=self.port.accepted if not self.owner.failure else np.zeros(2);slew=(accepted-self.previous_accepted)*1000;self.previous_accepted=accepted.copy()
            self.cost,factor=costs(row,accepted,slew,self.version)
            for name,value in zip(NAMES,self.cost):row[f'dyn_cost_{name}']=value
            row['reward_cap_factor']=factor;row['phase2c8_reward_version']=self.version
            for j,axis in enumerate(('mx','my')):row[f'requested_{axis}_Nm']=self.port.request[j];row[f'accepted_{axis}_Nm']=accepted[j];row[f'slew_{axis}_Nm_s']=slew[j]
            for j in range(6):
                row[f'rl_mapped_leg_{j}_N']=self.port.leg[j];row[f'same_state_base_add_{j}_N']=self.port.base_add[j];row[f'ff_requested_leg_{j}_N']=self.owner.residual.injection.audit.get('ff_leg',np.zeros(6))[j]
            row.update(decision=self.decision,global_decision=self.global_decisions,policy_version=self.context.get('version',-1),rl_source=self.source,projected=self.port.projected,stage=self.task['stage'],task_id=self.task['id'])
            append(row,state);self.last_row=row;self.last_physical_diagnostic_row=row
            self.physical_diagnostics['peak_so3_error']=max(self.physical_diagnostics['peak_so3_error'],row['orientation_error_deg']);self.physical_diagnostics['cap_steps']+=int(factor<1)
            for name,edge in (('load','on'),('unload','off')):
                if round(self.task[edge]*1000)<=row['input_step']<round(self.task[edge]*1000)+1000:self.physical_diagnostics[name+'_window_IAE']+=row['orientation_error_deg']*.001
        self.owner.recorder.append=log;self.obs=self.observe();return self.obs.copy(),dict(task_id=self.task['id'])
