"""Causal bounded LOCAL equivalent-wrench momentum residual, not true-force sensing.

M0 vdot = Wcmd - W0 - K0 x - D0 v + d_eq.
r_k=(r_{k-1}+omega*(M0*(v_k-v_{k-1})-dt*f_nom_{k-1}))/(1+omega*dt).
Only current pose/velocity and the previous known command enter. No acceleration
difference, external command, event time, actual mass or constraint-force inputs.
"""
import numpy as np

class MomentumDisturbanceObserver:
    def __init__(self,model,bandwidth_rad_s=15.,dt=.001):
        self.mass=np.array(model['mass']);self.damping=np.array(model['damping']);self.stiffness=np.array(model['stiffness']);self.equilibrium=np.array(model['equilibrium_wrench']);self.omega=float(bandwidth_rad_s);self.dt=dt
        if any(a.shape!=(6,6) or not np.isfinite(a).all() for a in (self.mass,self.damping,self.stiffness)) or self.equilibrium.shape!=(6,) or not np.isfinite(self.equilibrium).all():raise ValueError('Invalid nominal model')
        if not 0<self.omega<=50 or dt!=.001:raise ValueError('Invalid observer rate/bandwidth')
        if np.linalg.eigvalsh(self.mass).min()<=0:raise ValueError('Nominal mass must be positive definite')
        self.limits=np.array([20,20,20,.8,.8,.8]);self.reset()
    def reset(self):self.estimate=np.zeros(6);self.previous_velocity=None;self.previous_drive=None;self.steps=0;self.clipped=False
    def update(self,position_coordinates,world_velocity):
        x=np.asarray(position_coordinates,float);v=np.asarray(world_velocity,float)
        if x.shape!=(6,) or v.shape!=(6,) or not np.isfinite(np.r_[x,v]).all():raise ValueError('Invalid measured pose/velocity')
        if self.previous_velocity is not None:
            if self.previous_drive is None:raise RuntimeError('Missing previous actual command')
            raw=(self.estimate+self.omega*(self.mass@(v-self.previous_velocity)-self.dt*self.previous_drive))/(1+self.omega*self.dt);self.estimate=np.clip(raw,-self.limits,self.limits);self.clipped=bool(np.any(raw!=self.estimate))
        self.previous_velocity=v.copy();self.steps+=1
        return self.estimate.copy()
    def remember_applied_command(self,position_coordinates,world_velocity,known_command_wrench):
        x=np.asarray(position_coordinates);v=np.asarray(world_velocity);w=np.asarray(known_command_wrench)
        if any(z.shape!=(6,) for z in (x,v,w)) or not np.isfinite(np.r_[x,v,w]).all():raise ValueError('Invalid known input')
        self.previous_drive=w-self.equilibrium-self.stiffness@x-self.damping@v
