"""Learning uses the existing FF wrapper's pre-governor wrench port."""
import numpy as np

class LargeMotionAction:
    def __init__(self,owner,scale=.1):
        if scale!=.1:raise ValueError('Phase2C4 freezes initial authority at0.1N')
        self.owner=owner;self.scale=scale;self.last=0.
    def set(self,action):
        a=np.asarray(action,float)
        if a.shape!=(1,) or not np.isfinite(a).all():raise ValueError('One finite normalised action required')
        self.last=float(np.clip(a[0],-1,1));self.owner.residual.injection.probe_N=self.scale*self.last

