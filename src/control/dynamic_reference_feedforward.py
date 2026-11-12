"""Nominal Home-local Z reference feedforward, not full inverse dynamics."""
import numpy as np

class DynamicReferenceFeedforward:
    def __init__(self,kv,ka):
        if not np.isfinite([kv,ka]).all() or kv<0 or ka<0:raise ValueError('Nonnegative finite physical coefficients required')
        self.kv=float(kv);self.ka=float(ka)
    def wrench(self,reference,t):
        _,v,a=reference.scalar(t);w=np.zeros(6);w[2]=self.kv*v+self.ka*a
        if not np.isfinite(w).all():raise ValueError('Nonfinite reference wrench')
        return w,dict(velocity_m_s=float(v),acceleration_m_s2=float(a),velocity_force_N=self.kv*v,acceleration_force_N=self.ka*a)
    def leg_force(self,reference,t,jacobian):
        j=np.asarray(jacobian,float)
        if j.shape!=(6,6) or not np.isfinite(j).all():raise ValueError('Invalid nominal Jacobian')
        w,parts=self.wrench(reference,t);force=np.linalg.solve(j.T,w)
        if not np.isfinite(force).all():raise ValueError('Invalid feedforward mapping')
        return force,w,parts
