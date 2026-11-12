"""Conservative command planner, independent scratch queries; never modifies live data."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicHermiteSpline
from ..phase1.trajectory import Reference,smooth
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian,singularity_diagnostics
from ..audit_control import static_balance,scratch_reference_data

def target_spec(z_mm=0,roll_deg=0,pitch_deg=0):return dict(z_mm=float(z_mm),roll_deg=float(roll_deg),pitch_deg=float(pitch_deg))
def validate_target(spec):
    try:v=np.array([spec['z_mm'],spec['roll_deg'],spec['pitch_deg']],float)
    except (KeyError,TypeError,ValueError):raise ValueError('Expected Z/Roll/Pitch target')
    if not np.all(np.isfinite(v)):raise ValueError('Nonfinite target')
    if np.sum((v/np.array([4.,.4,.4]))**2)>1+1e-10:raise ValueError('Outside restricted command ellipsoid (4mm / 0.4deg / 0.4deg)')
    return target_spec(*v)
def pose_from_spec(geometry,spec):
    spec=validate_target(spec);p=geometry.home_position.copy();p[2]+=spec['z_mm']/1000
    return p,Rotation.from_euler('xyz',[spec['roll_deg'],spec['pitch_deg'],0],degrees=True).as_matrix()@geometry.home_rotation

class SmoothPlan:
    def __init__(self,model,geometry,start,goal,query,duration=2.,samples=41):
        self.duration=duration;self.start=start;self.goal=validate_target(goal);self.p1,self.r1=pose_from_spec(geometry,self.goal);self.rotvec=Rotation.from_matrix(self.r1@start.r.T).as_rotvec();times=np.linspace(0,duration,samples);q=[];dq=[];ff=[];seed=None;clearance=.2;max_condition=0.
        for t in times:
            p,r,twist=self.pose(t);fit=inverse_kinematics(geometry,p,r,seed)
            if not fit.success:raise ValueError('Full chain IK rejected target segment')
            seed=fit.chain_q;j,_=actuator_jacobian(geometry,p,r,fit);health=singularity_diagnostics(j)
            if health['near_singular'] or health['normalized_condition']>50:raise ValueError('Jacobian condition rejected')
            scratch=scratch_reference_data(model,geometry,fit,p,r);gap=query.evaluate(scratch.qpos)['minimum_clearance_m']
            if gap<.005:raise ValueError('Covered proxy clearance rejected')
            force,_=static_balance(model,geometry,fit,p,r);q.append(fit.strokes);dq.append(j@twist);ff.append(force);clearance=min(clearance,gap);max_condition=max(max_condition,health['normalized_condition'])
        q=np.asarray(q);dq=np.asarray(dq);q[0]=start.q;dq[0]=start.dq
        self.times=times;self.ff=np.asarray(ff);self.spline=CubicHermiteSpline(times,q,dq)
        dense=self.spline(np.arange(0,duration+.0001,.001))
        if not np.all(np.isfinite(dense)) or np.any(dense<-.1) or np.any(dense>.35):raise ValueError('Interpolated stroke rejected')
        self.audit=dict(samples=samples,minimum_covered_proxy_m=clearance,maximum_condition=max_condition,duration_s=duration,stroke_min_m=float(dense.min()),stroke_max_m=float(dense.max()))
    def pose(self,t):
        tau=float(np.clip(t/self.duration,0,1));s=smooth(tau);ds=(30*tau**2-60*tau**3+30*tau**4)/self.duration
        return self.start.p+s*(self.p1-self.start.p),Rotation.from_rotvec(s*self.rotvec).as_matrix()@self.start.r,np.r_[ds*(self.p1-self.start.p),ds*self.rotvec]
    def at(self,t):
        t=float(np.clip(t,0,self.duration));p,r,v=self.pose(t)
        return Reference(self.spline(t),self.spline(t,1),p,r,v,np.array([np.interp(t,self.times,self.ff[:,j]) for j in range(6)]))
