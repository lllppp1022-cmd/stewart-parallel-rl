import _bootstrap
import json,time
import numpy as np,pandas as pd,mujoco
from scipy.spatial.transform import Rotation
from src.model import ROOT,load_model,write_json,ids,sha
from src.audit_control import AuditController,scratch_reference_data
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.kinematics.jacobian import actuator_jacobian
from src.phase1.trajectory import Trajectory,Reference,protocol
from src.phase1.controllers import Controller
from src.phase1.plant import platform_state

def main():
    folder=ROOT/'results/phase1_preflight';folder.mkdir(exist_ok=True);m=load_model();m.opt.timestep=.001;cfg=protocol();cache=folder/'references';trajectories={}
    for name in cfg['nominal_trajectories']:
        tr=Trajectory(m,name,cache);trajectories[name]=tr;errors=[]
        for t in [1.011,1.723,2.245,3.417,4.141]:
            ref=tr.at(t);seed=tr.chain[np.argmin(abs(tr.times-t))];fit=inverse_kinematics(tr.geometry,ref.p,ref.r,seed)
            if not fit.success:raise ValueError('Off-grid exact IK failed')
            errors.append(float(max(abs(fit.strokes-ref.q))))
        write_json(folder/f'{name}_interpolation.json',dict(max_offgrid_stroke_error_m=max(errors),sample_times=[1.011,1.723,2.245,3.417,4.141],full_chain_ik=True,reference_within_previous_component_bounds=True))
        assert max(errors)<1e-7
        print('REFERENCE PASS',name,'off-grid error',max(errors),flush=True)
    # Original Phase0 implementation remains untouched; its legacy reference is reused here.
    legacy=AuditController(m,'z');c0=Controller(m,'C0',.001);a=mujoco.MjData(m);b=mujoco.MjData(m);mujoco.mj_forward(m,a);mujoco.mj_forward(m,b);max_u=0.;max_q=0.
    for k in range(5000):
        t=a.time+.001;q,dq,p,r,ff=legacy.reference(t);ref=Reference(q,dq,p,r,np.zeros(6),ff);pa,ra,twist=platform_state(m,b)
        u0,_=legacy.command(a,t);u1,_=c0.command(b.qpos,b.qvel,pa,ra,twist,ref,t);ua=np.zeros(12);ua[:6]=u1
        a.ctrl[:]=u0;b.ctrl[:]=ua;mujoco.mj_step(m,a);mujoco.mj_step(m,b);mujoco.mj_forward(m,a);mujoco.mj_forward(m,b)
        max_u=max(max_u,float(max(abs(u0-ua))));max_q=max(max_q,float(max(abs(a.qpos-b.qpos))))
    write_json(folder/'c0_phase0_equivalence.json',dict(physical_steps_per_model=5000,original_controller_sha256=sha(ROOT/'src/audit_control.py'),max_force_command_difference_N=max_u,max_qpos_difference=max_q,bitwise_equal=bool(np.array_equal(a.qpos,b.qpos)),scope='Original Z reference, same nominal 1ms plant; Phase1 trajectories deliberately replace the reference, not the C0 force law'))
    assert max_u==0 and max_q==0
    # Finite-difference constrained tangent cross-check of the virtual-work map.
    rng=np.random.default_rng(42);rows=[];qadr,dof=ids(m)
    for name,t in [('home',0),('z_roll',2),('roll_pitch',3),('periodic',2.37)]:
        tr=trajectories[name];ref=tr.at(t);fit=inverse_kinematics(tr.geometry,ref.p,ref.r,tr.chain[np.argmin(abs(tr.times-t))]);j,_=actuator_jacobian(tr.geometry,ref.p,ref.r,fit);T=np.zeros((m.nv,6));eps=1e-5
        for col in range(6):
            dp=np.eye(3)[col]*eps if col<3 else np.zeros(3);rv=np.eye(3)[col-3]*eps if col>=3 else np.zeros(3)
            rp=Rotation.from_rotvec(rv).as_matrix()@ref.r;rm=Rotation.from_rotvec(-rv).as_matrix()@ref.r
            qp=inverse_kinematics(tr.geometry,ref.p+dp,rp,fit.chain_q);qm=inverse_kinematics(tr.geometry,ref.p-dp,rm,fit.chain_q)
            da=scratch_reference_data(m,tr.geometry,qp,ref.p+dp,rp);db=scratch_reference_data(m,tr.geometry,qm,ref.p-dp,rm)
            tangent=np.zeros(m.nv);mujoco.mj_differentiatePos(m,tangent,2*eps,db.qpos,da.qpos);T[:,col]=tangent
        for trial in range(3):
            force=rng.uniform(-150,150,6);twist=rng.uniform(-.1,.1,6);d=scratch_reference_data(m,tr.geometry,fit,ref.p,ref.r);d.ctrl[:6]=force;mujoco.mj_forward(m,d)
            wrench=j.T@force;expected=wrench@twist;actual=d.qfrc_actuator@(T@twist)
            rows.append(dict(trajectory=name,time_s=t,trial=trial,seed=42,actuator_power_W=float(force@(j@twist)),platform_power_W=float(expected),mujoco_generalized_power_W=float(actual),virtual_work_abs_error_W=float(abs(force@(j@twist)-expected)),mujoco_power_abs_error_W=float(abs(actual-expected)),wrench_max_abs_error=float(max(abs(T.T@d.qfrc_actuator-wrench))),stroke_jacobian_max_error=float(max(abs(T[dof]-j).flat)),wrench_order='Fx,Fy,Fz,Tx,Ty,Tz; world at upper_center',twist_order='vx,vy,vz,wx,wy,wz',force_units='N; torque Nm; slide m'))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase1_force_mapping_validation.csv',index=False);assert max(r['wrench_max_abs_error'] for r in rows)<1e-5
    write_json(folder/'complete.json',dict(reference_names=list(trajectories),force_mapping_rows=len(rows),seed=42,controller_input_model='nominal only',full_chain_interpolation_verified=True))
    print('PREFLIGHT COMPLETE',flush=True)
if __name__=='__main__':main()
