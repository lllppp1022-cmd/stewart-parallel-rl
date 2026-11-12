"""Deterministic causal diagnostic actions; no network, environment or truth input."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .residual_governor import SharedBudgetGovernor

@dataclass(frozen=True)
class ActionObservation:
    p:np.ndarray
    rotation:np.ndarray
    velocity:np.ndarray
    reference_p:np.ndarray
    reference_rotation:np.ndarray
    reference_velocity:np.ndarray
    equivalent_disturbance:np.ndarray

def task_to_leg(action,jacobian,condition):
    action=np.asarray(action,float)
    if action.shape!=(3,) or not np.isfinite(action).all() or not np.isfinite(condition) or condition>50:raise ValueError('Invalid task action / Jacobian condition')
    w=np.zeros(6);w[[2,3,4]]=action;force=np.linalg.solve(jacobian.T,w)
    if not np.isfinite(force).all():raise ValueError('Invalid force mapping')
    return force,w

class CausalAction:
    privileged=False
    def __init__(self,kind='zero',space='task',frequency=20,vector=None):
        if kind not in ('zero','probe','ramp','damping','lead') or space not in ('task','leg') or frequency not in (20,50):raise ValueError('Invalid causal action specification')
        self.kind=kind;self.space=space;self.frequency=frequency;self.vector=np.asarray(vector if vector is not None else np.zeros(3 if space=='task' else 6),float);self.calls=0;self.previous_estimate=None
    def request(self,observation,jacobian):
        tick=self.calls/self.frequency;self.calls+=1
        if self.kind=='zero':return np.zeros(3 if self.space=='task' else 6)
        if self.kind in ('probe','ramp'):
            # Precommitted clock-only probe, never aligned from a load schedule.
            value=self.vector.copy() if .8<=tick<1.8 else np.zeros_like(self.vector)
            if self.kind=='ramp':value=self.vector*np.sin(2*np.pi*.7*tick) if .8<=tick<2.9 else np.zeros_like(self.vector)
            return value
        if self.kind=='damping':
            e=observation.p-observation.reference_p;v=observation.velocity-observation.reference_velocity;angle=Rotation.from_matrix(observation.rotation@observation.reference_rotation.T).as_rotvec()
            task=np.clip([-250*e[2]-50*v[2],-2*angle[0]-2*v[3],-2*angle[1]-2*v[4]],[-.5,-.025,-.025],[.5,.025,.025])
        else:
            estimate=observation.equivalent_disturbance[[2,3,4]];task=np.zeros(3) if self.previous_estimate is None else np.clip(-.02*(estimate-self.previous_estimate)*self.frequency,[-.5,-.025,-.025],[.5,.025,.025]);self.previous_estimate=estimate.copy()
        if self.space=='task':return np.asarray(task)
        # Matched task request mapped at the decision instant, then held in leg space.
        leg,_=task_to_leg(task,jacobian,1.);return np.clip(leg,-.1,.1)

class ResidualController:
    def __init__(self,base,action,allow_privileged=False):
        if getattr(action,'privileged',False) and not allow_privileged:raise ValueError('Privileged diagnostic rejected by standard action interface')
        self.base=base;self.action=action;self.allow_privileged=allow_privileged;self.name=base.name+'+res_'+action.space;self.governor=SharedBudgetGovernor();self.step_index=0;self.target=np.zeros(3 if action.space=='task' else 6);self.trace={}
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset();self.governor.reset();self.step_index=0;self.target[:]=0;self.trace={}
    def command(self,qpos,qvel,p,r,twist,reference,t):
        base,info=self.base.command(qpos,qvel,p,r,twist,reference,t);j=self.base.jacobian;diag=getattr(self.base,'diagnostics',{});c0=diag['base_force_N'].copy() if self.base.name=='C3' else base.copy();estimate=np.asarray(diag.get('equivalent_disturbance',np.zeros(6)))
        if j.shape!=(6,6) or not np.isfinite(j).all() or not np.isfinite(self.base.condition) or self.base.condition>50:raise ValueError('Invalid residual Jacobian')
        observation=ActionObservation(np.array(p),np.array(r),np.array(twist),reference.p.copy(),reference.r.copy(),reference.twist.copy(),estimate.copy())
        stride=1000//self.action.frequency;decision=(self.step_index%stride==0)
        if decision:self.target=np.asarray(self.action.request(observation,j),float)
        if self.action.space=='task':requested,w=task_to_leg(self.target,j,self.base.condition)
        else:
            requested=self.target.copy()
            if requested.shape!=(6,) or not np.isfinite(requested).all():raise ValueError('Invalid leg action')
            w=j.T@requested
        if abs(requested).max()>(5. if self.allow_privileged else 1.):raise ValueError('Action exceeds preflight candidate ceiling')
        cmd,g=self.governor.apply(c0,base,requested)
        # Exactly zero leaves the frozen observer call count/values untouched.
        changed=not np.array_equal(cmd,base)
        if self.base.name=='C3' and changed:self.base.observer.remember_applied_command(diag['position_coordinates'],twist,j.T@cmd)
        self.trace=dict(F_C0=c0,F_base=base.copy(),F_C3=base.copy() if self.base.name=='C3' else np.zeros(6),requested=requested,applied=g['applied_residual'],total_add=g['total_add'],W_requested=w,W_realized_nominal=j.T@g['applied_residual'],W_add_nominal=j.T@g['total_add'],reason=g['reason'],decision=decision,decision_index=self.action.calls-1,measurement_step=self.step_index,observer_command_corrected=changed and self.base.name=='C3',budget_margin=g['budget_margin'])
        self.step_index+=1;return cmd,dict(info,raw=cmd.copy(),clipped=False)
