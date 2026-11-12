"""Home-local six-axis model from NOMINAL inertia and closed-chain tangent space.

No actual-plant dynamic internals enter this model. Not a global nonlinear model.
"""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ..phase1.plant import create_plant
from ..phase1.controllers import Controller
from ..phase1.trajectory import Trajectory
from ..model import ROOT

def tangent(model,data):
    equality=np.asarray(data.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    if mujoco.mj_isSparse(model):
        jac=np.zeros((data.nefc,model.nv))
        for row in range(data.nefc):
            adr=int(data.efc_J_rowadr[row]);num=int(data.efc_J_rownnz[row]);jac[row,data.efc_J_colind[adr:adr+num]]=data.efc_J[adr:adr+num]
    else:jac=np.asarray(data.efc_J).reshape(data.nefc,model.nv)
    e=jac[equality];jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacSite(model,data,jp,jr,model.site('upper_center').id);site=np.r_[jp,jr]
    if e.shape!=(36,42) or np.linalg.matrix_rank(e)!=36:raise ValueError('Unexpected closed-chain constraint rank')
    t=np.linalg.solve(np.r_[e,site],np.r_[np.zeros((36,6)),np.eye(6)])
    return t,float(np.max(abs(e@t))),float(np.max(abs(site@t-np.eye(6))))

def build_model():
    m=create_plant('nominal');d=mujoco.MjData(m);mujoco.mj_forward(m,d);t,ec,jc=tangent(m,d);mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass);reduced=t.T@mass@t;damping=t.T@np.diag(m.dof_damping)@t
    q=d.qpos.copy();stiffness=np.zeros((6,6));eps=1e-6
    for j in range(6):
        sides=[]
        for sign in (-1,1):
            # Independent nominal kinematic perturbations only, never live execution.
            scratch=mujoco.MjData(m);scratch.qpos[:]=q;mujoco.mj_integratePos(m,scratch.qpos,t[:,j],sign*eps);mujoco.mj_forward(m,scratch);tj,_,_=tangent(m,scratch);sides.append(tj.T@(scratch.qfrc_bias-scratch.qfrc_passive))
        stiffness[:,j]=(sides[1]-sides[0])/(2*eps)
    ctrl=Controller(m,'C0',.001);ref=Trajectory(m,'home',ROOT/'results/phase1_preflight/references').at(0);w0=ctrl.jhome.T@ref.ff;nominal_gravity=t.T@(d.qfrc_bias-d.qfrc_passive)
    if np.linalg.eigvalsh(reduced).min()<=0 or max(ec,jc)>1e-9:raise ValueError('Invalid constrained nominal mass / tangent')
    return dict(mass=reduced.tolist(),damping=damping.tolist(),stiffness=stiffness.tolist(),equilibrium_wrench=w0.tolist(),projected_nominal_bias=nominal_gravity.tolist(),home_position=d.site('upper_center').xpos.tolist(),home_rotation=d.site('upper_center').xmat.reshape(3,3).tolist(),constraint_tangent_error=ec,site_tangent_error=jc,gravity_balance_error=float(np.max(abs(w0-nominal_gravity))),mass_eigenvalues=np.linalg.eigvalsh(reduced).tolist(),coordinate_order=['x_m','y_m','z_m','world_rot_x_rad','world_rot_y_rad','world_rot_z_rad'],valid_region='Home-local approximation; nominal accepted +/-4mm Z and +/-0.4deg Roll/Pitch only after residual validation',derivation='T satisfies E*T=0 and J_site*T=I; M0=T.T*Mfull*T, D0=T.T*diag(damping)*T, K0=central derivative of projected nominal bias. Coriolis, variable mass and compliant closure modes remain modelling residuals.')

def pose_coordinates(model,p,r):
    return np.r_[np.asarray(p)-model['home_position'],Rotation.from_matrix(np.asarray(r)@np.asarray(model['home_rotation']).T).as_rotvec()]
