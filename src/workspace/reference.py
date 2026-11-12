"""Full-chain Z path parameterised by quintic time; no ideal chord substitute."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from ..model import write_json
from ..phase1.trajectory import Reference,smooth
from ..kinematics.stewart_geometry import StewartGeometry
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian,singularity_diagnostics
from ..audit_control import static_balance
from .geometry import inspect_pose

class LargeReference:
    def __init__(self,model,amplitude,rise_s,cache,query):
        if amplitude not in (0.,.030,.050) or rise_s not in (4.,6.,10.):raise ValueError('Only predeclared references allowed')
        self.amplitude=amplitude;self.rise_s=rise_s;self.duration=3+2*rise_s;self.name=f'L{round(amplitude*1000)}_rise{rise_s:g}';self.geometry=g=StewartGeometry(model)
        file=Path(cache)/f'path_{amplitude:.3f}.npz';self.z=np.linspace(0,amplitude,max(2,round(amplitude/.0005)+1)) if amplitude else np.array([0.,1e-9])
        if file.exists():
            with np.load(file) as z:self.q=z['q'];self.dqdz=z['dqdz'];self.ff=z['ff'];self.chain=z['chain']
        else:
            q=[];dq=[];ff=[];chain=[];checks=[];seed=None
            for dz in self.z:
                row,sol=inspect_pose(model,g,query,float(dz),seed)
                if not row['simulation_geometric_candidate']:raise ValueError(f'Full-chain preflight rejected z={dz}: {row}')
                seed=sol.chain_q;p=g.home_position+[0,0,dz];j,_=actuator_jacobian(g,p,g.home_rotation,sol);u,_=static_balance(model,g,sol,p,g.home_rotation);q.append(sol.strokes);dq.append(j[:,2]);ff.append(u);chain.append(sol.chain_q);checks.append(row)
            self.q=np.array(q);self.dqdz=np.array(dq);self.ff=np.array(ff);self.chain=np.array(chain);file.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,q=self.q,dqdz=self.dqdz,ff=self.ff,chain=self.chain);write_json(file.with_suffix('.json'),dict(path_nodes=checks,source='Full chain IK + independent mesh check + static nominal force'))
        self.spline=CubicHermiteSpline(self.z,self.q,self.dqdz)
        times=np.arange(0,self.duration+.0001,.001);qs=np.array([self.at(t).q for t in times]);ranges=g.stroke_ranges
        if not np.isfinite(qs).all() or np.any(qs<ranges[:,0]) or np.any(qs>ranges[:,1]):raise ValueError('Dense 1ms reference stroke rejected')
        # Independent off-grid IK check: interpolation error is not assumed zero.
        errors=[]
        for dz in (self.z[:-1]+self.z[1:])/2:
            s=inverse_kinematics(g,g.home_position+[0,0,dz],g.home_rotation,self.chain[np.argmin(abs(self.z-dz))]);errors.append(float(max(abs(self.spline(dz)-s.strokes))))
        self.audit=dict(amplitude_m=amplitude,rise_s=rise_s,duration_s=self.duration,max_speed_m_s=1.875*amplitude/rise_s,max_acceleration_m_s2=(10/np.sqrt(3))*amplitude/rise_s**2,max_off_grid_stroke_error_m=max(errors,default=0.),max_static_force_N=float(abs(self.ff).max()),dense_reference_steps=len(times),initial_final_velocity_acceleration_zero=True)
        if self.audit['max_off_grid_stroke_error_m']>1e-7:raise ValueError('Off-grid interpolation mismatch')
    def scalar(self,t):
        T=self.rise_s;u=(t-1)/T;v=(t-2-T)/T
        def d(x):return (30*x*x-60*x**3+30*x**4)/T if 0<x<1 else 0.
        def dd(x):return (60*x-180*x*x+120*x**3)/T**2 if 0<x<1 else 0.
        return self.amplitude*(smooth(u)-smooth(v)),self.amplitude*(d(u)-d(v)),self.amplitude*(dd(u)-dd(v))
    def at(self,t):
        z,dz,_=self.scalar(float(np.clip(t,0,self.duration)));p=self.geometry.home_position+[0,0,z];twist=np.array([0,0,dz,0,0,0.]);q=self.spline(z);dq=self.spline(z,1)*dz;ff=np.array([np.interp(z,self.z,self.ff[:,i]) for i in range(6)])
        return Reference(q,dq,p,self.geometry.home_rotation.copy(),twist,ff)
