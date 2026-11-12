"""Shared platform pose references; exact-chain IK at grid nodes, verified interpolation."""
from dataclasses import dataclass
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicHermiteSpline
from ..model import ROOT,write_json
from ..kinematics.stewart_geometry import StewartGeometry
from ..kinematics.inverse_kinematics import inverse_kinematics,IKResult
from ..kinematics.jacobian import actuator_jacobian,singularity_diagnostics
from ..audit_control import static_balance

def protocol():return json.loads((ROOT/'configs/phase1_protocol.json').read_text(encoding='utf-8'))
def smooth(x):
    t=float(np.clip(x,0,1));return 10*t**3-15*t**4+6*t**5
def desired_pose(geometry,name,t):
    p=geometry.home_position.copy();rv=np.zeros(3);s=smooth((t-1)/2)
    if name=='z':p[2]+=.004*s
    elif name=='roll':rv[0]=np.deg2rad(.4)*s
    elif name=='pitch':rv[1]=np.deg2rad(.4)*s
    elif name=='z_roll':p[2]+=.0025*s;rv[0]=np.deg2rad(.25)*s
    elif name=='roll_pitch':rv[:2]=np.deg2rad([.25,.25])*s
    elif name=='periodic':
        u=float(np.clip((t-1)/4,0,1));envelope=np.sin(np.pi*u)**2;phase=2*np.pi*(t-1)/2
        rv[:2]=np.deg2rad([.2,.15])*envelope*np.array([np.sin(phase),np.cos(phase)])
    elif name!='home':raise ValueError('Unknown trajectory')
    return p,Rotation.from_rotvec(rv).as_matrix()@geometry.home_rotation
def desired_twist(geometry,name,t):
    h=1e-5;pp,rp=desired_pose(geometry,name,t+h);pm,rm=desired_pose(geometry,name,t-h)
    return np.r_[(pp-pm)/(2*h),Rotation.from_matrix(rp@rm.T).as_rotvec()/(2*h)]
@dataclass
class Reference:
    q:np.ndarray
    dq:np.ndarray
    p:np.ndarray
    r:np.ndarray
    twist:np.ndarray
    ff:np.ndarray

class Trajectory:
    def __init__(self,nominal_model,name,cache=None):
        self.name=name;self.geometry=StewartGeometry(nominal_model);self.config=cfg=protocol();self.duration=cfg['duration_s']
        self.times=np.linspace(0,self.duration,round(self.duration/cfg['reference_grid_s'])+1)
        self.q=[];self.dq=[];self.ff=[];self.chain=[];self.validation=[];seed=None;start=time.perf_counter()
        cachefile=Path(cache)/f'{name}.npz' if cache else None
        if cachefile is not None and cachefile.exists():
            z=np.load(cachefile);self.q=z['q'];self.dq=z['dq'];self.ff=z['ff'];self.chain=z['chain'];z.close()
        else:
            for t in self.times:
                p,r=desired_pose(self.geometry,name,t);sol=inverse_kinematics(self.geometry,p,r,seed)
                if not sol.success:raise ValueError(f'IK rejected {name} at {t}')
                seed=sol.chain_q;j,_=actuator_jacobian(self.geometry,p,r,sol);health=singularity_diagnostics(j)
                if health['near_singular'] or health['normalized_condition']>cfg['jacobian_condition_limit']:raise ValueError('Reference Jacobian conditioning rejected')
                ff,_=static_balance(nominal_model,self.geometry,sol,p,r);self.q.append(sol.strokes);self.dq.append(j@desired_twist(self.geometry,name,t));self.ff.append(ff);self.chain.append(sol.chain_q)
                self.validation.append(dict(time_s=float(t),stroke_margin_m=sol.minimum_stroke_margin,closure_translation_m=sol.max_position_error,closure_rotation_rad=sol.max_rotation_error,jacobian_condition=health['normalized_condition']))
            self.q=np.asarray(self.q);self.dq=np.asarray(self.dq);self.ff=np.asarray(self.ff);self.chain=np.asarray(self.chain)
            if cachefile:
                cachefile.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cachefile,q=self.q,dq=self.dq,ff=self.ff,chain=self.chain)
                write_json(cachefile.with_suffix('.json'),dict(name=name,grid_s=cfg['reference_grid_s'],seconds=time.perf_counter()-start,nodes=self.validation,source='Only frozen nominal parameters, full chain IK and static balance; no perturbed physics parameters'))
        self.spline=CubicHermiteSpline(self.times,self.q,self.dq)
        qdense=self.spline(np.arange(0,self.duration+.0001,.001))
        if not np.all(np.isfinite(qdense)) or np.any(qdense<self.geometry.stroke_ranges[:,0]) or np.any(qdense>self.geometry.stroke_ranges[:,1]):raise ValueError('Interpolated reference stroke invalid')
    def at(self,t):
        t=float(np.clip(t,0,self.duration));p,r=desired_pose(self.geometry,self.name,t)
        return Reference(self.spline(t),self.spline(t,1),p,r,desired_twist(self.geometry,self.name,t),np.array([np.interp(t,self.times,self.ff[:,j]) for j in range(6)]))
