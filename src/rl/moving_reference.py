"""Phase2C2 nominal full-chain IK reference; no writes to live state."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from ..phase1.trajectory import Reference, smooth
from ..kinematics.stewart_geometry import StewartGeometry
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian, singularity_diagnostics
from ..audit_control import static_balance
from ..model import write_json

class MovingReference:
    duration=6.;name='Phase2C2 smooth Z out-and-back'
    def __init__(self,model,amplitude_m,cache,shift_s=0.):
        if not 0<=amplitude_m<=.003: raise ValueError('Research reference ceiling is 3 mm, safety envelope unchanged')
        if abs(shift_s)>.2: raise ValueError('Reference timing outside frozen range')
        self.geometry=StewartGeometry(model);self.amplitude=float(amplitude_m);self.shift=float(shift_s)
        self.times=np.linspace(0,6,301);file=Path(cache)/f'z_{amplitude_m:.6f}_{shift_s:.3f}.npz'
        if file.exists():
            with np.load(file) as z:self.q=z['q'];self.dq=z['dq'];self.ff=z['ff']
        else:
            qs=[];dqs=[];ffs=[];checks=[];seed=None
            for t in self.times:
                p,r,v=self.pose(t);sol=inverse_kinematics(self.geometry,p,r,seed)
                if not sol.success: raise ValueError(f'Full-chain IK rejected at {t}')
                seed=sol.chain_q;j,_=actuator_jacobian(self.geometry,p,r,sol);health=singularity_diagnostics(j)
                if health['near_singular'] or health['normalized_condition']>50:raise ValueError('Singular reference')
                ff,_=static_balance(model,self.geometry,sol,p,r);qs.append(sol.strokes);dqs.append(j@v);ffs.append(ff)
                checks.append(dict(time_s=float(t),stroke_margin_m=sol.minimum_stroke_margin,closure_m=sol.max_position_error,closure_rad=sol.max_rotation_error,condition=health['normalized_condition']))
            self.q=np.array(qs);self.dq=np.array(dqs);self.ff=np.array(ffs);file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,q=self.q,dq=self.dq,ff=self.ff)
            write_json(file.with_suffix('.json'),dict(amplitude_m=amplitude_m,shift_s=shift_s,nodes=checks,source='Frozen nominal model full-chain IK + static balance'))
        self.spline=CubicHermiteSpline(self.times,self.q,self.dq);dense=self.spline(np.linspace(0,6,6001))
        if not np.isfinite(dense).all() or np.any(dense<self.geometry.stroke_ranges[:,0]) or np.any(dense>self.geometry.stroke_ranges[:,1]):raise ValueError('Dense stroke check failed')
    def pose(self,t):
        u=(t-.5-self.shift)/1.5;w=(t-3-self.shift)/1.5
        def ds(x):return (30*x*x-60*x**3+30*x**4)/1.5 if 0<x<1 else 0.
        p=self.geometry.home_position.copy();p[2]+=self.amplitude*(smooth(u)-smooth(w));v=np.zeros(6);v[2]=self.amplitude*(ds(u)-ds(w))
        return p,self.geometry.home_rotation.copy(),v
    def at(self,t):
        t=float(np.clip(t,0,6));p,r,v=self.pose(t)
        return Reference(self.spline(t),self.spline(t,1),p,r,v,np.array([np.interp(t,self.times,self.ff[:,j]) for j in range(6)]))
