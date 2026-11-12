"""C1-L: independent wrapper retaining the frozen C1 P/D/Ki/antiwindup code."""
import numpy as np
from ..phase1.controllers import Controller

class LeakyPID:
    name='C1-L'
    def __init__(self,nominal_model,leak_per_s=0.,dt=.001):
        if not np.isfinite(leak_per_s) or not 0<=leak_per_s<=5 or dt!=.001:raise ValueError('Invalid leakage / controller period')
        self.base=Controller(nominal_model,'C1',dt);self.leak_per_s=float(leak_per_s);self.dt=dt;self.diagnostics={}
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset();self.diagnostics={}
    def command(self,qpos,qvel,p,r,twist,reference,t):
        if not all(np.all(np.isfinite(x)) for x in (qpos,qvel,p,r,twist,reference.q)):raise ValueError('Invalid feedback')
        before=self.base.integral.copy();leak_delta=-self.dt*self.leak_per_s*before
        # lambda=0 directly calls frozen C1, without even a nominal multiply/add.
        if self.leak_per_s:self.base.integral=before+leak_delta
        e=reference.q-qpos[self.qadr];ev=reference.dq-qvel[self.dof];cfg=self.config
        candidate=np.clip(self.base.integral+e*self.dt,-cfg['integral_limit_m_s'],cfg['integral_limit_m_s']);proposal=reference.ff+cfg['kp_leg_N_per_m']*e+cfg['kd_leg_Ns_per_m']*ev+cfg['ki_leg_N_per_m_s']*candidate
        pushing=((proposal>400)&(e>0))|((proposal<-400)&(e<0))
        force,info=self.base.command(qpos,qvel,p,r,twist,reference,t)
        self.diagnostics=dict(position_error_m=e.copy(),integral_state=self.base.integral.copy(),integral_force_N=cfg['ki_leg_N_per_m_s']*self.base.integral,leak_delta_m_s=leak_delta,leak_force_delta_N=cfg['ki_leg_N_per_m_s']*leak_delta,antiwindup_mask=pushing.astype(float),base_force_N=force.copy(),compensation_N=np.zeros(6))
        return force,info
