"""Solve each actual six-joint chain to its platform weld attachment.
This is a kinematic reference, not a dynamic qpos controller.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from .stewart_geometry import valid_pose
@dataclass
class IKResult:
    success:bool
    chain_q:np.ndarray
    strokes:np.ndarray
    chord_lengths:np.ndarray
    max_position_error:float
    max_rotation_error:float
    minimum_stroke_margin:float
    reason:str

def inverse_kinematics(geometry,position,orientation,initial=None):
    p,r=valid_pose(position,orientation)
    seed=np.zeros((6,6)) if initial is None else np.asarray(initial,float).copy()
    if seed.shape!=(6,6) or not np.all(np.isfinite(seed)):raise ValueError('Invalid IK seed')
    qs=[];pe=[];re=[];success=True
    for i in range(6):
        tp=p+r@geometry.top_sites[i];tr=r@geometry.top_site_rotations[i]
        def error(q):
            t=geometry.forward_leg(i,q);return np.r_[t[:3,3]-tp,Rotation.from_matrix(t[:3,:3]@tr.T).as_rotvec()]
        lower=np.full(6,-np.inf);upper=np.full(6,np.inf);lower[2],upper[2]=geometry.stroke_ranges[i]
        q0=seed[i].copy();q0[2]=np.clip(q0[2],lower[2]+1e-10,upper[2]-1e-10)
        fit=least_squares(error,q0,bounds=(lower,upper),xtol=1e-13,ftol=1e-13,gtol=3e-15,max_nfev=160)
        e=error(fit.x);qs.append(fit.x);pe.append(np.linalg.norm(e[:3]));re.append(np.linalg.norm(e[3:]));success &= bool(fit.success and pe[-1]<1e-7 and re[-1]<1e-7)
    qs=np.asarray(qs);strokes=qs[:,2];margin=float(np.min(np.minimum(strokes-geometry.stroke_ranges[:,0],geometry.stroke_ranges[:,1]-strokes)))
    success=bool(success and margin>=0)
    return IKResult(success,qs,strokes,geometry.attachment_chord_lengths(p,r),max(pe),max(re),margin,'ok' if success else 'closure_unreachable_or_solver_failed')
