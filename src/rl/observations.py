"""Only whitelisted measured/known values enter fixed-scale history features."""
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class MeasuredState:
    p:np.ndarray
    rotation:np.ndarray
    velocity:np.ndarray
    slides:np.ndarray
    slide_velocity:np.ndarray
    reference_p:np.ndarray
    reference_rotation:np.ndarray
    reference_velocity:np.ndarray
    known_c4_command:np.ndarray
    equivalent_estimate:np.ndarray
    applied_rl_command:np.ndarray

class HistoryObservation:
    def __init__(self,scales,length=3):self.scales=scales;self.length=length;self.history=deque(maxlen=length);self.clip_count=0
    def reset(self,state,home_p,home_r,home_q,home_force):
        self.home_p=home_p.copy();self.home_r=home_r.copy();self.home_q=home_q.copy();self.home_force=home_force.copy();self.history.clear();self.clip_count=0;frame=self.frame(state,0.,[])
        for _ in range(self.length):self.history.append(frame.copy())
        return self.value()
    def frame(self,s,action,interval):
        c=self.scales;ps=np.asarray(c['position_m']);vs=np.r_[np.full(3,c['linear_velocity_m_s']),np.full(3,c['angular_velocity_rad_s'])];es=np.r_[np.full(3,c['estimated_force_N']),np.full(3,c['estimated_moment_Nm'])]
        ep=(s.p-s.reference_p)/ps;er=Rotation.from_matrix(s.rotation@s.reference_rotation.T).as_rotvec()/c['angle_rad']
        base=np.r_[ep,er,s.velocity/vs,(s.slides-self.home_q)/c['slide_m'],s.slide_velocity/c['slide_velocity_m_s'],(s.known_c4_command-self.home_force)/c['known_command_N'],s.equivalent_estimate/es,s.applied_rl_command/c['applied_residual_N'],(s.reference_p-self.home_p)/.005,Rotation.from_matrix(s.reference_rotation@self.home_r.T).as_rotvec()/.01,s.reference_velocity/vs,action]
        if interval:
            errors=np.array([(x.p-x.reference_p)/ps for x in interval]);summary=np.r_[errors.mean(axis=0),abs(errors).max(axis=0),np.mean([x.velocity/vs for x in interval],axis=0),(interval[-1].equivalent_estimate-interval[0].equivalent_estimate)/es,(interval[-1].known_c4_command-interval[0].known_c4_command)/c['known_command_N'],np.mean([x.applied_rl_command/c['applied_residual_N'] for x in interval],axis=0)]
        else:summary=np.zeros(30)
        out=np.r_[base,summary]
        if out.shape!=(85,) or not np.isfinite(out).all():raise ValueError('Invalid causal feature frame')
        self.clip_count+=int(np.count_nonzero(abs(out)>10));return np.clip(out,-10,10).astype(np.float32)
    def append(self,state,action,interval):self.history.append(self.frame(state,action,interval));return self.value()
    def value(self):return np.concatenate(self.history).astype(np.float32)
