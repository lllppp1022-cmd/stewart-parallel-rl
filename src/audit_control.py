"""Phase-0 bounded force/PD sanity check, not a tuned Phase-1 controller.
Live MjData is only advanced by ctrl + mj_step; reference qpos writes are
confined to explicitly separate scratch inverse-static data.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .model import ids,site_gap,equality_residual
from .kinematics.stewart_geometry import StewartGeometry
from .kinematics.inverse_kinematics import inverse_kinematics

def scratch_reference_data(model,geometry,solution,p,r):
    scratch=mujoco.MjData(model)
    for i in range(6):
        for node,q in zip(geometry.legs[i][1:],solution.chain_q[i]):scratch.qpos[node.qpos_address]=q
    a=int(model.joint('upper_joint').qposadr[0]);scratch.qpos[a:a+3]=p
    q=Rotation.from_matrix(r).as_quat();scratch.qpos[a+3:a+7]=q[[3,0,1,2]]
    mujoco.mj_forward(model,scratch);return scratch

def static_balance(model,geometry,solution,p,r):
    if not solution.success:raise ValueError('Unclosed reference rejected')
    scratch=scratch_reference_data(model,geometry,solution,p,r)
    eq=np.asarray(scratch.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    j=np.asarray(scratch.efc_J).reshape(scratch.nefc,model.nv)[eq]
    _,dof=ids(model);b=np.zeros((model.nv,6));b[dof,np.arange(6)]=model.actuator_gear[:6,0]
    lhs=np.c_[b,j.T];rhs=scratch.qfrc_bias-scratch.qfrc_passive
    fit=np.linalg.lstsq(lhs,rhs,rcond=None);u=fit[0][:6];error=float(np.linalg.norm(lhs@fit[0]-rhs))
    if not np.all(np.isfinite(u)) or error>1e-5 or np.any(abs(u)>400):raise ValueError('Unusable static force balance')
    return u,error

class AuditController:
    def __init__(self,model,axis='home',sign=1,config=None):
        self.model=model;self.geometry=g=StewartGeometry(model);self.address,self.dof=ids(model);self.axis=axis
        self.kp=5000.;self.kd=200.;self.move_start=1.;self.move_duration=2.
        if config:
            self.kp=config['kp_N_per_m'];self.kd=config['kd_Ns_per_m'];self.move_start=config['hold_before_s'];self.move_duration=config['move_duration_s']
        self.p0=g.home_position.copy();self.r0=g.home_rotation.copy();self.p1=self.p0.copy();self.r1=self.r0.copy()
        z_amplitude=.005 if config is None else config['z_amplitude_m'];tilt=.5 if config is None else config['tilt_amplitude_deg']
        if axis=='z':self.p1[2]+=z_amplitude*sign
        elif axis in ('roll','pitch'):self.r1=Rotation.from_euler('x' if axis=='roll' else 'y',tilt*sign,degrees=True).as_matrix()@self.r0
        elif axis not in ('home','zero','pulse'):raise ValueError('Unknown audit axis')
        self.home=inverse_kinematics(g,self.p0,self.r0);self.target=inverse_kinematics(g,self.p1,self.r1,self.home.chain_q)
        middle=inverse_kinematics(g,(self.p0+self.p1)/2,Rotation.from_rotvec(Rotation.from_matrix(self.r1@self.r0.T).as_rotvec()/2).as_matrix()@self.r0,self.home.chain_q)
        if not all(x.success for x in (self.home,self.target,middle)):raise ValueError('Target/midpoint closure or stroke check failed')
        self.u0,self.static_residual0=static_balance(model,g,self.home,self.p0,self.r0);self.u1,self.static_residual1=static_balance(model,g,self.target,self.p1,self.r1)
        self.rotvec=Rotation.from_matrix(self.r1@self.r0.T).as_rotvec()
    def reference(self,t):
        tau=float(np.clip((t-self.move_start)/self.move_duration,0,1));s=10*tau**3-15*tau**4+6*tau**5
        ds=(30*tau**2-60*tau**3+30*tau**4)/self.move_duration if 0<tau<1 else 0.
        stroke=self.home.strokes+s*(self.target.strokes-self.home.strokes)
        vel=ds*(self.target.strokes-self.home.strokes)
        return stroke,vel,self.p0+s*(self.p1-self.p0),Rotation.from_rotvec(s*self.rotvec).as_matrix()@self.r0,(1-s)*self.u0+s*self.u1
    def command(self,data,t):
        q,v,_,_,ff=self.reference(t)
        if self.axis=='zero':raw=np.zeros(6)
        else:raw=ff+self.kp*(q-data.qpos[self.address])+self.kd*(v-data.qvel[self.dof])
        if self.axis=='pulse' and 1<=t<1.2:raw[0]+=2.
        if not np.all(np.isfinite(raw)):raise ValueError('Nonfinite force command')
        u=np.zeros(self.model.nu);u[:6]=np.clip(raw,self.model.actuator_ctrlrange[:6,0],self.model.actuator_ctrlrange[:6,1]);return u,raw

def state_row(model,data,controller,k,control):
    q,v,p,r,_=controller.reference(data.time);up=data.body('upper');actual_r=up.xmat.reshape(3,3)
    speed=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('upper').id,speed,0)
    gap=site_gap(model,data);angle=[]
    for i in range(1,7):angle.append(np.linalg.norm(Rotation.from_matrix(data.site(f'site_c1_{i+12}').xmat.reshape(3,3)@data.site(f'site_upper_{i}').xmat.reshape(3,3).T).as_rotvec()))
    margin=np.minimum(data.qpos[controller.address]-controller.geometry.stroke_ranges[:,0],controller.geometry.stroke_ranges[:,1]-data.qpos[controller.address])
    row=dict(step=k,time_s=float(data.time),scenario=controller.axis,position_error_m=float(np.linalg.norm(up.xpos-p)),orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(actual_r@r.T).as_rotvec())),max_site_gap_m=float(max(gap)),max_site_angle_rad=float(max(angle)),max_mixed_equality_residual=float(max(abs(equality_residual(data)))),max_abs_qvel=float(max(abs(data.qvel))),max_abs_qacc=float(max(abs(data.qacc))),minimum_stroke_margin_m=float(min(margin)),contacts=int(data.ncon),warning_count=int(sum(w.number for w in data.warning)))
    for prefix,values in [('stroke_ref_m',q),('stroke_actual_m',data.qpos[controller.address]),('stroke_velocity_mps',data.qvel[controller.dof]),('actuator_length_m',data.actuator_length[:6]),('ctrl_N',control[:6]),('actuator_force_N',data.actuator_force[:6]),('weld_gap_m',gap),('weld_angle_rad',angle),('qpos',data.qpos),('qvel',data.qvel),('qacc',data.qacc),('ref_position_m',p),('actual_position_m',up.xpos),('actual_quat_wxyz',up.xquat),('actual_world_ang_lin_velocity',speed),('all_ctrl',data.ctrl)]:
        for j,x in enumerate(values):row[f'{prefix}_{j+1}']=float(x)
    return row

def advance(model,data,controller,k):
    previous=float(data.time);u,raw=controller.command(data,previous+model.opt.timestep);data.ctrl[:]=u
    mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
    if not np.isclose(data.time,previous+model.opt.timestep,atol=1e-10):raise RuntimeError('Time reset or unexpected integration count')
    if not all(np.all(np.isfinite(x)) for x in (data.qpos,data.qvel,data.qacc,data.actuator_force)):raise RuntimeError('Nonfinite physical state')
    row=state_row(model,data,controller,k,u);row['command_clipped']=bool(np.any(abs(raw)>400))
    if row['max_abs_qvel']>20 or row['max_abs_qacc']>1e5 or row['max_site_gap_m']>.005:raise RuntimeError('Frozen numerical/closure guard exceeded')
    return row
