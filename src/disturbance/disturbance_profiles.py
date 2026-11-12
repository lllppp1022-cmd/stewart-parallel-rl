"""Integer-step half-open load intervals, independent of controller and GUI clocks."""
from dataclasses import dataclass
import numpy as np,mujoco

@dataclass(frozen=True)
class DisturbanceProfile:
    start_step:int=2000
    stop_step:int=4000
    wrench:tuple=(0.,0.,0.,0.,0.,0.)
    frame:str='world'
    kind:str='step'
    dt:float=.001
    def __post_init__(self):
        w=np.asarray(self.wrench,float)
        if self.frame not in ('world','site-local') or self.kind not in ('step','pulse','composite','zero'):raise ValueError('Invalid frame/kind')
        if w.shape!=(6,) or not np.isfinite(w).all() or np.max(abs(w))>500:raise ValueError('Invalid six-channel wrench or native range')
        if not isinstance(self.start_step,int) or not isinstance(self.stop_step,int) or not 0<=self.start_step<self.stop_step or self.dt!=.001:raise ValueError('Invalid physical-step interval / dt')
    def active(self,k):return self.kind!='zero' and self.start_step<=k<self.stop_step
    def command(self,k,rotation):
        w=np.asarray(self.wrench,float) if self.active(k) else np.zeros(6)
        local=np.r_[rotation.T@w[:3],rotation.T@w[3:]] if self.frame=='world' else w
        if np.max(abs(local))>500:raise ValueError('Transformed command exceeds native range')
        return local
    def description(self):
        from dataclasses import asdict
        return dict(**asdict(self),units=['N']*3+['N m']*3,site='upper_center',reference_point='current upper_center site origin',step_convention='command k drives interval [k*dt,(k+1)*dt); logged row step=k+1',load_time_s=self.start_step*self.dt,unload_time_s=self.stop_step*self.dt)

def generalized_external(model,data,local):
    """Independent Jacobian-transpose and mj_applyFT checks at the pre-step pose."""
    site=model.site('upper_center').id;body=int(model.site_bodyid[site]);r=data.site_xmat[site].reshape(3,3);world=np.r_[r@local[:3],r@local[3:]]
    jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacSite(model,data,jp,jr,site)
    jac=jp.T@world[:3]+jr.T@world[3:];applied=np.zeros(model.nv)
    mujoco.mj_applyFT(model,data,world[:3],world[3:],data.site_xpos[site],body,applied)
    # Native site transmission, expanded from MuJoCo's sparse actuator-moment matrix.
    native=np.zeros(model.nv)
    for i in range(6,12):
        adr=int(data.moment_rowadr[i]);num=int(data.moment_rownnz[i]);cols=data.moment_colind[adr:adr+num]
        native[cols]+=data.actuator_moment[adr:adr+num]*local[i-6]*model.actuator_gainprm[i,0]
    return dict(world=world,jacobian=jac,native=native,apply_ft=applied,maximum_error=float(max(np.max(abs(jac-native)),np.max(abs(jac-applied)))))
