"""4D world wrench adapter to the frozen residual wrapper and shared governor."""
import numpy as np
from .residual_action import ResidualController

def task4_to_leg(action,jacobian,condition):
    a=np.asarray(action,float);j=np.asarray(jacobian,float)
    if a.shape!=(4,) or j.shape!=(6,6) or not np.isfinite(np.r_[a,j.ravel(),condition]).all() or condition>50:raise ValueError('Invalid four-dimensional wrench/Jacobian')
    w=np.r_[0.,0.,a];force=np.linalg.solve(j.T,w)
    if not np.isfinite(force).all():raise ValueError('Nonfinite four-dimensional mapping')
    return force,w

class FourDAdapter:
    """Map held task wrench at 1kHz using the unchanged available nominal J.

    The frozen wrapper sees leg forces at1kHz; the inner causal policy may only
    update at20/50Hz. This avoids modifying either frozen 3D code or its C3.
    """
    space='leg';frequency=1000
    def __init__(self,policy):
        self.policy=policy;self.privileged=getattr(policy,'privileged',False);self.calls=0;self.target=np.zeros(4);self.decision=False;self.policy_version=0
    def request(self,observation,jacobian):
        self.decision=self.calls%(1000//self.policy.frequency)==0
        if self.decision:self.target=np.asarray(self.policy.request(observation),float);self.policy_version+=1
        self.calls+=1
        return task4_to_leg(self.target,jacobian,1.)[0]

class Task4Controller(ResidualController):
    def __init__(self,base,policy):
        self.policy=policy;self.adapter=FourDAdapter(policy)
        super().__init__(base,self.adapter);self.name=policy.name
    def reset(self):
        super().reset();self.policy.reset();self.adapter.calls=0;self.adapter.target[:]=0;self.adapter.policy_version=0;self.adapter.decision=False

class ZeroOrProbe4:
    privileged=False;frequency=20
    def __init__(self,vector=None):
        self.vector=np.asarray(vector if vector is not None else [0]*4,float)
        if self.vector.shape!=(4,) or not np.isfinite(self.vector).all() or abs(self.vector[3])>.25:raise ValueError('Invalid preflight probe')
        self.name='zero4' if not self.vector.any() else 'probe4';self.reset()
    def reset(self):self.calls=0
    def request(self,o):
        t=self.calls/self.frequency;self.calls+=1
        return self.vector.copy() if .8<=t<1.8 else np.zeros(4)
