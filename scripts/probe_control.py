import _bootstrap
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from src.model import load_model,ids,site_gap,equality_residual,ROOT,write_json
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
m=load_model();g=StewartGeometry(m);addr,dof=ids(m);home=inverse_kinematics(g,g.home_position,g.home_rotation)
def reference_data(sol,p,r):
    d=mujoco.MjData(m)
    for i in range(6):
        for node,q in zip(g.legs[i][1:],sol.chain_q[i]):d.qpos[node.qpos_address]=q
    a=int(m.joint('upper_joint').qposadr[0]);d.qpos[a:a+3]=p;qr=Rotation.from_matrix(r).as_quat();d.qpos[a+3:a+7]=qr[[3,0,1,2]]
    mujoco.mj_forward(m,d);return d
def ff(sol,p,r):
    d=reference_data(sol,p,r);j=np.asarray(d.efc_J).reshape(d.nefc,m.nv)[d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]
    b=np.zeros((m.nv,6));b[dof,np.arange(6)]=1
    mat=np.c_[b,j.T];fit=np.linalg.lstsq(mat,d.qfrc_bias-d.qfrc_passive,rcond=None);return fit[0][:6],np.linalg.norm(mat@fit[0]-(d.qfrc_bias-d.qfrc_passive))
u0,res=ff(home,g.home_position,g.home_rotation);print('home ff',u0,'balance residual',res,'damping',m.dof_damping[dof],flush=True)
records=[]
for name,p,r in [('home',g.home_position,g.home_rotation),('z',g.home_position+np.array([0,0,.005]),g.home_rotation),('roll',g.home_position,Rotation.from_euler('x',.5,degrees=True).as_matrix()),('pitch',g.home_position,Rotation.from_euler('y',.5,degrees=True).as_matrix())]:
    sol=inverse_kinematics(g,p,r,home.chain_q);ut,res=ff(sol,p,r);print(name,'ik',sol.success,'targets',sol.strokes,'ff',ut,flush=True)
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);mgap=0;force=0
    for k in range(2500):
        t=k*.002;tau=np.clip((t-1)/2,0,1);s=10*tau**3-15*tau**4+6*tau**5;ds=(30*tau**2-60*tau**3+30*tau**4)/2 if 0<tau<1 else 0
        qr=home.strokes+s*(sol.strokes-home.strokes);vr=ds*(sol.strokes-home.strokes)
        u=(1-s)*u0+s*ut+5000*(qr-d.qpos[addr])+200*(vr-d.qvel[dof]);d.ctrl[:6]=np.clip(u,-400,400);d.ctrl[6:]=0
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d);mgap=max(mgap,max(site_gap(m,d)));force=max(force,max(abs(d.actuator_force[:6])))
    poserr=np.linalg.norm(d.body('upper').xpos-p);roterr=np.linalg.norm(Rotation.from_matrix(d.body('upper').xmat.reshape(3,3)@r.T).as_rotvec())
    row=dict(name=name,pos_error_mm=poserr*1000,rotation_error_deg=np.rad2deg(roterr),max_gap_m=mgap,max_force=force,final_position=d.body('upper').xpos.tolist(),final_strokes=d.qpos[addr].tolist(),target_strokes=sol.strokes.tolist());records.append(row);print(row,flush=True)
write_json(ROOT/'results/setup/probe_control.json',records)
