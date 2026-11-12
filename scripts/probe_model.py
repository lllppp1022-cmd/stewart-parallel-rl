import _bootstrap
import mujoco,numpy as np,json
from src.model import load_model,write_json,ROOT,site_gap,equality_residual
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.kinematics.jacobian import actuator_jacobian,singularity_diagnostics
m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=StewartGeometry(m)
print('model',m.nq,m.nv,m.nu,'equality',m.neq,'bodies',m.nbody,'timestep',m.opt.timestep,flush=True)
print('initial gap',site_gap(m,d),'residual',np.max(abs(equality_residual(d))),flush=True)
print('initial top',d.body('upper').xpos,'readme distances',g.readme_lengths(g.home_position,g.home_rotation),flush=True)
sol=inverse_kinematics(g,g.home_position,g.home_rotation)
print('IK',sol.success,'stroke',sol.strokes,'pe/re',sol.max_position_error,sol.max_rotation_error,flush=True)
try:j,cc=actuator_jacobian(g,g.home_position,g.home_rotation,sol);print('J',j,'conditions',cc,singularity_diagnostics(j),flush=True)
except Exception as e:print('J failure',repr(e),flush=True)
out=[]
for k in range(1000):
    d.ctrl[:]=0;mujoco.mj_step(m,d)
    if k%100==0:out.append(dict(step=k,time=float(d.time),z=float(d.body('upper').xpos[2]),gap=float(max(site_gap(m,d))),max_velocity=float(max(abs(d.qvel))),max_acceleration=float(max(abs(d.qacc)))))
    if not np.all(np.isfinite(d.qpos)) or np.max(abs(d.qvel))>100:print('ABORT',k,flush=True);break
print(json.dumps(out,indent=2),flush=True)
write_json(ROOT/'results/setup/probe_model.json',dict(samples=out,initial_home=g.home_position.tolist(),ik_success=sol.success,home_chain_q=sol.chain_q.tolist()))
