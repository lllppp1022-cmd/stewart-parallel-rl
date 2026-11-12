"""Frozen C0 + bounded Z/Roll/Pitch equivalent-wrench compensation."""
import numpy as np
from ..phase1.controllers import Controller
from .disturbance_observer import MomentumDisturbanceObserver
from .nominal_task_model import pose_coordinates

class RobustControllerC3:
    name='C3'
    def __init__(self,nominal_model,task_model,bandwidth_rad_s=15.,gain=1.,force_bound_N=5.,force_rate_N_s=20.,dt=.001):
        if not 0<=gain<=1 or not 0<force_bound_N<=5 or not 0<force_rate_N_s<=20:raise ValueError('Invalid bounded compensation')
        self.base=Controller(nominal_model,'C0',dt);self.task_model=task_model;self.observer=MomentumDisturbanceObserver(task_model,bandwidth_rad_s,dt);self.gain=gain;self.bound=force_bound_N;self.rate=force_rate_N_s;self.dt=dt;self.previous_compensation=np.zeros(6);self.diagnostics={}
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset();self.observer.reset();self.previous_compensation[:]=0;self.diagnostics={}
    def command(self,qpos,qvel,p,r,twist,reference,t):
        base,info=self.base.command(qpos,qvel,p,r,twist,reference,t);j=self.base.jacobian
        if not np.isfinite(j).all() or not np.isfinite(self.condition) or self.condition>self.config['jacobian_condition_limit']:raise ValueError('Invalid compensation Jacobian')
        x=pose_coordinates(self.task_model,p,r);estimate=self.observer.update(x,twist);w=np.zeros(6);w[[2,3,4]]=-self.gain*estimate[[2,3,4]]
        desired=np.linalg.solve(j.T,w);bounded=np.clip(desired,-self.bound,self.bound);limited=np.clip(bounded,self.previous_compensation-self.rate*self.dt,self.previous_compensation+self.rate*self.dt);total=np.clip(base+limited,-400,400);actual=total-base
        if np.max(abs(actual-self.previous_compensation))>self.rate*self.dt+1e-10:raise ValueError('Native force headroom incompatible with compensation rate; protection required')
        self.previous_compensation=actual.copy();self.observer.remember_applied_command(x,twist,j.T@total)
        self.diagnostics=dict(base_force_N=base.copy(),compensation_N=actual.copy(),desired_compensation_N=desired,compensation_target_wrench=w,realized_compensation_wrench=j.T@actual,equivalent_disturbance=estimate,observer_clipped=float(self.observer.clipped),compensation_rate_limited=float(np.any(limited!=bounded)),compensation_amplitude_limited=float(np.any(desired!=bounded)),position_coordinates=x.copy())
        info=dict(info,raw=base+limited,clipped=bool(np.any(total!=base+limited)))
        return total,info
