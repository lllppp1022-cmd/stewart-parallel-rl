"""Independent NumPy/SciPy FK of actual MJCF joint/attachment transforms.
No MjData or mj_forward in FK; model is only a static parameter source.
README ideal points are retained separately and never silently substituted.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

def rotation(quat_wxyz):
    q=np.asarray(quat_wxyz,float);return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
def transform(p,r):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t
def valid_pose(position,orientation):
    p=np.asarray(position,float);r=np.asarray(orientation,float)
    if p.shape!=(3,) or r.shape!=(3,3) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):raise ValueError('Expected finite position(3) and rotation(3,3)')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1,atol=1e-8):raise ValueError('Invalid SO(3) orientation')
    return p,r
@dataclass
class JointNode:
    body_name:str
    fixed:np.ndarray
    joint_type:int
    axis:np.ndarray
    anchor:np.ndarray
    qpos_address:int

class StewartGeometry:
    def __init__(self,model):
        self.legs=[];self.stroke_ranges=[];self.top_sites=[];self.base_points=[];self.top_site_rotations=[]
        self.home_position=np.asarray(model.body('upper').pos).copy();self.home_rotation=rotation(model.body('upper').quat)
        self.readme_base=np.array([[.483,.1294,.067],[-.1294,.483,.067],[-.3536,.3536,.067],[-.3536,-.3536,.067],[-.1294,-.483,.067],[.483,-.1294,.067]])
        self.readme_top=np.array([[.2475,.2475,-.067],[.0906,.3381,-.067],[-.3381,.0906,-.067],[-.3381,-.0906,-.067],[.0906,-.3381,-.067],[.2475,-.2475,-.067]])
        for i in range(1,7):
            names=[f'cardan1_{i}',f'cardan2_{i}',f'legbot_{i}',f'legup_{i}',f'cardan1_{i+6}',f'cardan2_{i+6}',f'cardan1_{i+12}'];nodes=[]
            for name in names:
                b=model.body(name);j=int(b.jntadr[0]) if b.jntnum[0] else -1
                nodes.append(JointNode(name,transform(b.pos,rotation(b.quat)),int(model.jnt_type[j]) if j>=0 else -1,model.jnt_axis[j].copy() if j>=0 else np.zeros(3),model.jnt_pos[j].copy() if j>=0 else np.zeros(3),int(model.jnt_qposadr[j]) if j>=0 else -1))
            self.legs.append(nodes);self.base_points.append(nodes[0].fixed[:3,3]);s=model.site(f'site_upper_{i}')
            self.top_sites.append(s.pos.copy());self.top_site_rotations.append(rotation(s.quat));self.stroke_ranges.append(model.joint(f'joint_lu_{i}').range.copy())
        self.base_points=np.asarray(self.base_points);self.top_sites=np.asarray(self.top_sites);self.stroke_ranges=np.asarray(self.stroke_ranges)
    def forward_leg(self,index,q,return_chain=False):
        q=np.asarray(q,float)
        if q.shape!=(6,) or not np.all(np.isfinite(q)):raise ValueError('Expected six finite chain coordinates')
        t=np.eye(4);axes=[];anchors=[];frames=[];k=0
        for node in self.legs[index]:
            t=t@node.fixed
            if node.joint_type>=0:
                axes.append(t[:3,:3]@node.axis);anchors.append(t[:3,3]+t[:3,:3]@node.anchor)
                if node.joint_type==3:
                    rr=Rotation.from_rotvec(node.axis*q[k]).as_matrix();t=t@transform(node.anchor-rr@node.anchor,rr)
                elif node.joint_type==2:t=t@transform(node.axis*q[k],np.eye(3))
                else:raise ValueError('Unexpected leg joint type')
                k+=1
            frames.append(t.copy())
        return (t,np.asarray(axes),np.asarray(anchors),frames) if return_chain else t
    def attachment_chord_lengths(self,p,r):
        p,r=valid_pose(p,r);return np.linalg.norm(p+self.top_sites@r.T-self.base_points,axis=1)
    def readme_lengths(self,p,r):
        p,r=valid_pose(p,r);return np.linalg.norm(p+self.readme_top@r.T-self.readme_base,axis=1)
