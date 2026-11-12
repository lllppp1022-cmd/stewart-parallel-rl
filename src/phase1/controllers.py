"""Frozen nominal-model controllers. No controller receives a perturbed MjModel."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ..model import ids
from ..kinematics.stewart_geometry import StewartGeometry
from ..kinematics.inverse_kinematics import inverse_kinematics,IKResult
from ..kinematics.jacobian import actuator_jacobian,singularity_diagnostics
from .trajectory import protocol

class Controller:
    def __init__(self,nominal_model,name,dt):
        if name not in ('C0','C1','C2'):raise ValueError('Controller must be C0/C1/C2')
        self.name=name;self.dt=dt;self.config=cfg=protocol();self.geometry=g=StewartGeometry(nominal_model);self.qadr,self.dof=ids(nominal_model)
        self.chain_addresses=np.array([[n.qpos_address for n in leg[1:]] for leg in g.legs]);self.integral=np.zeros(6);self.integral_clamp_count=0;self.antiwindup_count=0;self.reset_count=0
        sol=inverse_kinematics(g,g.home_position,g.home_rotation);self.jhome,_=actuator_jacobian(g,g.home_position,g.home_rotation,sol)
        self.k_task=self.jhome.T@(cfg['kp_leg_N_per_m']*np.eye(6))@self.jhome;self.d_task=self.jhome.T@(cfg['kd_leg_Ns_per_m']*np.eye(6))@self.jhome
        self.jacobian=self.jhome.copy();self.condition=singularity_diagnostics(self.jacobian)['normalized_condition'];self.last_jacobian_time=-1.
    def reset(self):self.integral[:]=0;self.reset_count+=1
    def command(self,qpos,qvel,p,r,twist,reference,t):
        cfg=self.config;q=qpos[self.qadr];dq=qvel[self.dof];eq=reference.q-q;ev=reference.dq-dq
        if not all(np.all(np.isfinite(v)) for v in (qpos,qvel,p,r,twist,reference.q,reference.dq,reference.ff)):raise ValueError('Nonfinite state/reference rejected')
        if np.any(reference.q<self.geometry.stroke_ranges[:,0]) or np.any(reference.q>self.geometry.stroke_ranges[:,1]):raise ValueError('Reference stroke out of range')
        if t-self.last_jacobian_time>=cfg['jacobian_refresh_s']-1e-10:
            chains=qpos[self.chain_addresses]
            observed=IKResult(True,chains,q,self.geometry.attachment_chord_lengths(p,r),0.,0.,0.,'Measured coordinates; kinematic parameters remain nominal')
            self.jacobian,_=actuator_jacobian(self.geometry,p,r,observed);health=singularity_diagnostics(self.jacobian);self.condition=health['normalized_condition'];self.last_jacobian_time=t
            if health['near_singular'] or self.condition>cfg['jacobian_condition_limit']:raise ValueError('Near-singular force map')
        if self.name=='C2':
            error=np.r_[reference.p-p,Rotation.from_matrix(reference.r@r.T).as_rotvec()]
            wrench=self.jacobian.T@reference.ff+self.k_task@error+self.d_task@(reference.twist-twist)
            raw=np.linalg.solve(self.jacobian.T,wrench)
        else:
            pd=reference.ff+cfg['kp_leg_N_per_m']*eq+cfg['kd_leg_Ns_per_m']*ev
            if self.name=='C1':
                candidate=self.integral+eq*self.dt;clamped=np.clip(candidate,-cfg['integral_limit_m_s'],cfg['integral_limit_m_s']);self.integral_clamp_count+=int(np.any(candidate!=clamped))
                proposed=pd+cfg['ki_leg_N_per_m_s']*clamped
                pushing=(proposed>cfg['force_limit_N'])&(eq>0) | (proposed<-cfg['force_limit_N'])&(eq<0)
                self.antiwindup_count+=int(np.any(pushing));self.integral=np.where(pushing,self.integral,clamped)
                raw=pd+cfg['ki_leg_N_per_m_s']*self.integral
            else:raw=pd
        if not np.all(np.isfinite(raw)):raise ValueError('Nonfinite force command')
        force=np.clip(raw,-cfg['force_limit_N'],cfg['force_limit_N'])
        return force,dict(raw=raw,integral=self.integral.copy(),condition=self.condition,clipped=bool(np.any(raw!=force)),integral_clamp_count=self.integral_clamp_count,antiwindup_count=self.antiwindup_count,wrench_projection_error=float(np.linalg.norm(self.jacobian.T@(force-raw))))
