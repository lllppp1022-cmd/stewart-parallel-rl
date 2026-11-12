"""Synthetic first-order force response, not a hydraulic/identified driver model."""
import numpy as np

class ActuatorLag:
    def __init__(self,tau_s,initial_force,enabled=True,dt=.001):
        self.tau=np.broadcast_to(np.asarray(tau_s,float),(6,)).copy();self.initial=np.asarray(initial_force,float).copy();self.enabled=bool(enabled);self.dt=float(dt)
        if self.initial.shape!=(6,) or not np.isfinite(np.r_[self.tau,self.initial,self.dt]).all() or np.any(self.tau<0) or np.any(self.tau>.012) or self.dt!=.001 or abs(self.initial).max()>400:raise ValueError('Invalid bounded synthetic actuator dynamics')
        self.alpha=np.ones(6);positive=self.tau>0;self.alpha[positive]=-np.expm1(-dt/self.tau[positive]);self.state=self.initial.copy();self.steps=0
    def advance(self,command):
        c=np.asarray(command,float)
        if c.shape!=(6,) or not np.isfinite(c).all():raise ValueError('Invalid force command')
        c=np.clip(c,-400,400)
        if not self.enabled or not self.tau.any():self.state=c.copy()
        else:
            self.state=np.clip(self.state+self.alpha*(c-self.state),-400,400);self.state[self.tau==0]=c[self.tau==0]
        self.steps+=1;return self.state.copy()

def initialization_description(initial):
    return dict(source='Frozen nominal Home reference equilibrium leg force before first integration',force_N=np.asarray(initial).tolist(),not_zero_start=True,no_live_state_change=True)
