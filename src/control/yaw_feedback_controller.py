"""Nonprivileged C4-PD and finite20Hz diagnostic on measured world SO(3)."""
import numpy as np
from scipy.spatial.transform import Rotation

class YawFeedback:
    privileged=False;frequency=1000;name='C4-PD'
    def __init__(self,kp,kd,limit=.30):
        if kp not in (100.,300.) or not np.isfinite([kp,kd,limit]).all() or not 0<=kd<=100 or not 0<limit<=.30:raise ValueError('Outside frozen finite PD search')
        self.kp=float(kp);self.kd=float(kd);self.limit=limit;self.reset()
    def reset(self):pass
    def request(self,o):
        error=Rotation.from_matrix(o.rotation@o.reference_rotation.T).as_rotvec()
        rate=o.velocity-o.reference_velocity
        return np.array([0.,0.,0.,np.clip(-self.kp*error[2]-self.kd*rate[5],-self.limit,self.limit)])

class CausalFourD:
    """No network:20Hz local PD yaw + finite difference observer lead Z/R/P."""
    privileged=False;frequency=20;name='C3+causal4d20'
    def __init__(self,kp,kd):self.yaw=YawFeedback(kp,kd);self.reset()
    def reset(self):self.previous=None
    def request(self,o):
        a=self.yaw.request(o);est=o.equivalent_disturbance[2:5]
        if self.previous is not None:a[:3]=np.clip(-.02*(est-self.previous)*20,[-.5,-.025,-.025],[.5,.025,.025])
        self.previous=est.copy();return a
