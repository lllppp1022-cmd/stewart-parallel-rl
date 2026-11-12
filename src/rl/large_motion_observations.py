"""Explicit53-field causal contract; no plant truth or disturbance schedule."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from ..phase1.plant import platform_state

FIELDS=[('position_error',3),('rotation_error',3),('actual_twist',6),('reference_twist',6),('reference_z_acceleration',1),('relative_slides',6),('slide_velocity',6),('known_command',6),('causal_observer',6),('reference_ff_z',1),('accepted_total_add',6),('last_accepted_residual',1),('actual_z',1),('target_z',1)]

class LargeObservation:
    def __init__(self,length=1):
        if length not in [1,3]:raise ValueError('Only frozen1/3-frame histories')
        self.length=length;self.history=deque(maxlen=length);self.clips=0;self.elements=0;self.peak=0.
    def frame(self,o):
        trajectory=o.learning_reference
        p,r,v=platform_state(o.model,o.data);ref=trajectory.at(float(o.data.time));core=o.residual.base;est=core.diagnostics.get('equivalent_disturbance',np.zeros(6));a=o.residual.injection.audit;add=a.get('actual_total_add',np.zeros(6));effect=a.get('probe_effect_command',np.zeros(6));applied=core.jacobian.T@effect
        values=[(p-ref.p)/[5e-5,5e-5,1e-5],Rotation.from_matrix(r@ref.r.T).as_rotvec()/.00034906585039886593,v/np.r_[np.full(3,.03),np.full(3,.05)],ref.twist/np.r_[np.full(3,.03),np.full(3,.05)],[trajectory.scalar(o.data.time)[2]/.03],(o.data.qpos[o.qadr]-trajectory.at(0).q)/.06,o.data.qvel[o.dof]/.03,(o.last_command-o.home.at(0).ff)/5,np.asarray(est)/[20,20,20,.8,.8,.8],[a.get('ff_wrench',np.zeros(6))[2]/20],add/5,[applied[2]/.1],[(p[2]-o.geometry.home_position[2])/.05],[(ref.p[2]-o.geometry.home_position[2])/.05]]
        out=np.concatenate([np.asarray(x).reshape(-1) for x in values])
        if len(out)!=sum(n for _,n in FIELDS) or len(out)!=53 or not np.isfinite(out).all():raise ValueError('Invalid53-field observation')
        self.clips+=int((abs(out)>10).sum());self.elements+=len(out);self.peak=max(self.peak,float(abs(out).max()));return np.clip(out,-10,10).astype(np.float32)
    def reset(self,o):
        self.history.clear();x=self.frame(o)
        for _ in range(self.length):self.history.append(x.copy())
        return self.value()
    def append(self,o):self.history.append(self.frame(o));return self.value()
    def value(self):return np.concatenate(self.history).copy()

