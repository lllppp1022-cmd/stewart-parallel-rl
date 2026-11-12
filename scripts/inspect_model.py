import _bootstrap
import json,platform,sys,subprocess,importlib,importlib.metadata
import numpy as np,mujoco,pandas as pd
from scipy.spatial.transform import Rotation
from src.model import *
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.kinematics.jacobian import *
from src.audit_control import scratch_reference_data

def axis_center(data,model,names):
    blocks=[];rhs=[]
    for name in names:
        j=model.joint(name).id;u=data.xaxis[j];a=data.xanchor[j];p=np.eye(3)-np.outer(u,u);blocks.append(p);rhs.append(p@a)
    center=np.linalg.lstsq(np.concatenate(blocks),np.concatenate(rhs),rcond=None)[0]
    return center,max(float(np.linalg.norm(a@center-b)) for a,b in zip(blocks,rhs))
def main():
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=StewartGeometry(m)
    snapshot=dict(nq=m.nq,nv=m.nv,nu=m.nu,nbody=m.nbody,njnt=m.njnt,neq=m.neq,ngeom=m.ngeom,nefc=d.nefc,timestep_s=m.opt.timestep,integrator=int(m.opt.integrator),gravity=m.opt.gravity.tolist(),joint_types=dict(free=int(sum(m.jnt_type==0)),ball=int(sum(m.jnt_type==1)),slide=int(sum(m.jnt_type==2)),hinge=int(sum(m.jnt_type==3))),initial_site_gap_m=site_gap(m,d).tolist(),initial_max_mixed_equality_residual=float(max(abs(equality_residual(d)))),upper_mass_kg=float(m.body('upper').mass[0]),upper_principal_inertia_kg_m2=m.body('upper').inertia.tolist(),upper_inertial_position_m=m.body('upper').ipos.tolist(),initial_position_m=d.body('upper').xpos.tolist(),initial_quat_wxyz=d.body('upper').xquat.tolist(),robot_geoms_collision_disabled=bool(np.all(m.geom_contype[:-1]==0)) if m.geom('floor').id==m.ngeom-1 else [m.geom(i).name for i in range(m.ngeom) if m.geom_contype[i]==0],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),source_url='https://github.com/duccuongvu/stewart_platform_mujoco',license='MIT, Copyright (c) 2026 vr-cuongvu',source_files={str(p.relative_to(UPSTREAM)):sha(p) for p in UPSTREAM.rglob('*') if p.is_file() and '.git' not in p.relative_to(UPSTREAM).parts})
    snapshot['actuators']=[]
    snapshot['robot_geoms_collision_disabled']=bool(all(m.geom_contype[i]==0 and m.geom_conaffinity[i]==0 for i in range(m.ngeom) if m.geom(i).name!='floor'))
    for i in range(m.nu):
        aid=int(m.actuator_trnid[i,0]);typ=int(m.actuator_trntype[i]);snapshot['actuators'].append(dict(id=i,name=m.actuator(i).name,transmission_type=typ,target_joint=m.joint(aid).name if typ==0 else None,target_site=m.site(aid).name if typ==4 else None,gear=m.actuator_gear[i].tolist(),ctrlrange=m.actuator_ctrlrange[i].tolist(),forcerange=m.actuator_forcerange[i].tolist(),gain_type=int(m.actuator_gaintype[i]),gain_prm=m.actuator_gainprm[i].tolist(),bias_type=int(m.actuator_biastype[i]),bias_prm=m.actuator_biasprm[i].tolist(),dyn_type=int(m.actuator_dyntype[i])))
    snapshot['joints']=[dict(name=m.joint(i).name,type=int(m.jnt_type[i]),qpos_address=int(m.jnt_qposadr[i]),dof_address=int(m.jnt_dofadr[i]),limited=bool(m.jnt_limited[i]),range=m.jnt_range[i].tolist()) for i in range(m.njnt)]
    snapshot['equalities']=[dict(name=m.equality(i).name,type=int(m.eq_type[i]),obj1=int(m.eq_obj1id[i]),obj2=int(m.eq_obj2id[i]),solref=m.eq_solref[i].tolist(),solimp=m.eq_solimp[i].tolist()) for i in range(m.neq)]
    snapshot['leg_topology']=[[node.body_name for node in leg] for leg in g.legs]
    anchors=[]
    for i in range(1,7):
        b,bskew=axis_center(d,m,[f'joint_c2_{i}',f'joint_lb_{i}']);top,tskew=axis_center(d,m,[f'joint_c1_{i+6}',f'joint_c2_{i+6}',f'joint_c1_{i+12}']);local=g.home_rotation.T@(top-g.home_position)
        for label,point,skew,readme in [('base',b,bskew,g.readme_base[i-1]),('top_local',local,tskew,g.readme_top[i-1])]:
            anchors.append(dict(leg=i,frame=label,x_m=point[0],y_m=point[1],z_m=point[2],axis_common_center_max_distance_m=skew,readme_same_index_difference_m=np.linalg.norm(point-readme),is_exact_common_center=skew<1e-6))
    pd.DataFrame(anchors).to_csv(ROOT/'results/phase0_anchor_audit.csv',index=False)
    snapshot['geometry_reference']=dict(base_mount_body_origins_m=g.base_points.tolist(),platform_weld_sites_local_m=g.top_sites.tolist(),readme_B_m=g.readme_base.tolist(),readme_T_m=g.readme_top.tolist(),stroke_ranges_m=g.stroke_ranges.tolist(),anchor_warning='README points/order differ from compiled MJCF. Upper spherical-joint proxy axes have ~0.25mm skew; exact chain IK is used, not ideal chord-length-as-stroke substitution.')
    write_json(ROOT/'results/phase0_model_inspection.json',snapshot)
    rows=[];home=inverse_kinematics(g,g.home_position,g.home_rotation)
    poses=[('home',g.home_position,g.home_rotation),('z_up',g.home_position+[0,0,.005],g.home_rotation),('z_down',g.home_position-[0,0,.005],g.home_rotation),('roll',g.home_position,Rotation.from_euler('x',.5,degrees=True).as_matrix()),('pitch',g.home_position,Rotation.from_euler('y',.5,degrees=True).as_matrix())]
    rng=np.random.default_rng(42)
    for k in range(2):poses.append((f'random42_{k}',g.home_position+rng.uniform(-.002,.002,3),Rotation.from_rotvec(rng.uniform(-.003,.003,3)).as_matrix()))
    for name,p,r in poses:
        solution=inverse_kinematics(g,p,r,home.chain_q);scratch=scratch_reference_data(m,g,solution,p,r);fkerr=0.;roterr=0.
        for i in range(6):
            t=g.forward_leg(i,solution.chain_q[i]);body=scratch.body(f'cardan1_{i+13}');fkerr=max(fkerr,np.linalg.norm(t[:3,3]-body.xpos));roterr=max(roterr,np.linalg.norm(Rotation.from_matrix(t[:3,:3]@body.xmat.reshape(3,3).T).as_rotvec()))
        j,cc=actuator_jacobian(g,p,r,solution);health=singularity_diagnostics(j)
        for epsilon in [1e-4,1e-5,1e-6]:
            fd=finite_difference_jacobian(g,p,r,solution,epsilon)
            rows.append(dict(case=name,ik_success=solution.success,closure_position_error_m=solution.max_position_error,closure_rotation_error_rad=solution.max_rotation_error,fk_mujoco_position_error_m=fkerr,fk_mujoco_rotation_error_rad=roterr,scratch_weld_gap_m=max(site_gap(m,scratch)),actuator_length_stroke_max_error_m=float(max(abs(scratch.actuator_length[:6]-solution.strokes))),minimum_stroke_margin_m=solution.minimum_stroke_margin,fd_epsilon=epsilon,jacobian_max_abs_error=float(max(abs(j-fd).flat)),normalized_condition=health['normalized_condition'],near_singular=health['near_singular']))
        print('kinematics',name,'closed',solution.success,'FK',fkerr,flush=True)
    bad=inverse_kinematics(g,g.home_position+[0,0,10],g.home_rotation,home.chain_q)
    snapshot['unreachable_test']=dict(target_z_m=float(g.home_position[2]+10),rejected=not bad.success,reason=bad.reason);write_json(ROOT/'results/phase0_model_inspection.json',snapshot)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase0_kinematics_validation.csv',index=False)
    deps={n:importlib.metadata.version(n) for n in ['mujoco','numpy','scipy','gymnasium','matplotlib','pandas','pytest','imageio','imageio-ffmpeg','pillow']}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    write_json(ROOT/'results/environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),gpu=gpu,dependencies=deps,torch_installed=importlib.util.find_spec('torch') is not None,sb3_installed=importlib.util.find_spec('stable_baselines3') is not None))
    (ROOT/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    print('Inspection saved',flush=True)
if __name__=='__main__':main()
