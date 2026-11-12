import numpy as np
from scipy.spatial.transform import Rotation
from .inverse_kinematics import inverse_kinematics
from .stewart_geometry import valid_pose

def chord_jacobian(geometry,p,r):
    p,r=valid_pose(p,r);offset=geometry.top_sites@r.T;v=p+offset-geometry.base_points;n=np.linalg.norm(v,axis=1)
    if np.any(n<1e-9):raise ValueError('Degenerate chord')
    u=v/n[:,None];return np.c_[u,np.cross(offset,u)]
def actuator_jacobian(geometry,p,r,solution):
    """dl/d(world translation, world rotation); exact chain implicit Jacobian."""
    p,r=valid_pose(p,r)
    if not solution.success:raise ValueError('IK must close before Jacobian evaluation')
    rows=[];chain_conditions=[]
    for i in range(6):
        t,axes,anchors,_=geometry.forward_leg(i,solution.chain_q[i],True);end=t[:3,3]
        g=np.zeros((6,6))
        for j in range(6):
            if j==2:g[:3,j]=axes[j]
            else:g[:3,j]=np.cross(axes[j],end-anchors[j]);g[3:,j]=axes[j]
        skew=np.column_stack([np.cross(np.eye(3)[j],r@geometry.top_sites[i]) for j in range(3)])
        mapping=np.block([[np.eye(3),skew],[np.zeros((3,3)),np.eye(3)]])
        chain_conditions.append(float(np.linalg.cond(g)));rows.append(np.linalg.solve(g,mapping)[2])
    return np.asarray(rows),chain_conditions
def finite_difference_jacobian(geometry,p,r,solution,epsilon=1e-6):
    out=np.zeros((6,6))
    for j in range(6):
        dp=np.zeros(3);dr=np.zeros(3)
        if j<3:dp[j]=epsilon
        else:dr[j-3]=epsilon
        plus=inverse_kinematics(geometry,np.asarray(p)+dp,Rotation.from_rotvec(dr).as_matrix()@r,solution.chain_q)
        minus=inverse_kinematics(geometry,np.asarray(p)-dp,Rotation.from_rotvec(-dr).as_matrix()@r,solution.chain_q)
        if not plus.success or not minus.success:raise ValueError('Finite difference IK did not close')
        out[:,j]=(plus.strokes-minus.strokes)/(2*epsilon)
    return out
def singularity_diagnostics(jacobian,characteristic_length=.35):
    # Translation columns dimensionless; scale rotational columns by length to compare units.
    j=np.asarray(jacobian,float).copy();j[:,3:]/=characteristic_length
    sv=np.linalg.svd(j,compute_uv=False);condition=float(sv[0]/sv[-1]) if sv[-1]>1e-14 else None
    return dict(singular_values=sv.tolist(),normalized_condition=condition,near_singular=bool(sv[-1]<1e-5 or condition is None or condition>1e5),characteristic_length_m=characteristic_length)
