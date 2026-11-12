"""Explicit convex-mesh distance queries on independent MjData, native flags retained."""
import itertools
import numpy as np,mujoco
from ..phase1.plant import create_plant
from ..kinematics.stewart_geometry import StewartGeometry
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian,singularity_diagnostics
from ..audit_control import scratch_reference_data

def branch(name):
    for part in ['cardan1','cardan2','legbot','legup']:
        if name.startswith(part+'_'):
            i=int(name.split('_')[-1]);leg=(i-1)%6
            node=0 if part=='cardan1' and i<=6 else 4 if part=='cardan1' and i<=12 else 6 if part=='cardan1' else 1 if part=='cardan2' and i<=6 else 5 if part=='cardan2' else 2 if part=='legbot' else 3
            return leg,node
    return None

def exclusion(a,b):
    if a==b:return 'same body'
    if set([a,b])=={'base','world'}:return 'fixed base resting on floor'
    aa,bb=branch(a),branch(b)
    if aa and bb and aa[0]==bb[0]:
        u,v=sorted([aa[1],bb[1]])
        if (u<=2 and v<=2) or (u>=3 and v>=3):return 'same connected universal/upper attachment assembly; concave sockets filled by hull, not independent obstacles'
        if (u,v)==(2,3):return 'intentional telescoping cylinder/rod nesting'
    for fixture,other in [(a,bb),(b,aa)]:
        if other and ((fixture=='base' and other[1]<=2) or (fixture=='upper' and other[1]>=3)):return 'intended fixture/attachment contact; connected assembly excluded, not blanket self-collision disable'
    return None

class ExpandedGeometryQuery:
    def __init__(self):
        self.model=create_plant();self.data=mujoco.MjData(self.model);self.pairs=[];self.exclusions=[]
        for a,b in itertools.combinations(range(self.model.ngeom),2):
            ba=self.model.body(int(self.model.geom_bodyid[a])).name;bb=self.model.body(int(self.model.geom_bodyid[b])).name;why=exclusion(ba,bb);item=dict(geom1=a,body1=ba,geom2=b,body2=bb)
            if why:self.exclusions.append(dict(item,reason=why))
            else:self.pairs.append((a,b,ba,bb))
        self.nativeccd=not bool(self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
        if not self.nativeccd:raise RuntimeError('Distance audit requires original native GJK pipeline')
    def evaluate(self,qpos,details=False):
        q=np.asarray(qpos,float)
        if q.shape!=(self.model.nq,) or not np.isfinite(q).all():raise ValueError('Bad query state')
        self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data);minimum=.15;near=None;bad=[];cross=.15;platform=.15
        for a,b,ba,bb in self.pairs:
            d=float(mujoco.mj_geomDistance(self.model,self.data,a,b,.15,None))
            if not np.isfinite(d):raise ValueError('Nonfinite mesh distance')
            if d<minimum:minimum=d;near=[ba,bb]
            if branch(ba) and branch(bb) and branch(ba)[0]!=branch(bb)[0]:cross=min(cross,d)
            if set([ba,bb])=={'upper','base'}:platform=d
            if details and d<.005:bad.append(dict(body1=ba,body2=bb,distance_m=d))
        return dict(minimum_m=minimum,minimum_is_lower_bound=minimum>=.15,nearest_pair=near,cross_branch_m=cross,cross_branch_is_lower_bound=cross>=.15,platform_gap_m=platform,platform_gap_is_lower_bound=platform>=.15,distmax_m=.15,violating_pairs=bad,query_pairs=len(self.pairs),coverage='ALL native visual mesh convex hulls + floor, excluding explicitly connected assemblies; not exact concave-socket or hardware certification')

def check_native_joint_limits(model,qpos):
    margins=[];bad=[];unlimited=[]
    for j in range(model.njnt):
        if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
        if not model.jnt_limited[j]:unlimited.append(model.joint(j).name);continue
        value=qpos[int(model.jnt_qposadr[j])];lo,hi=model.jnt_range[j];margin=min(value-lo,hi-value);margins.append(float(margin))
        if margin<0:bad.append(model.joint(j).name)
    return dict(valid=not bad,minimum_native_margin=min(margins) if margins else None,violating_joints=bad,unlimited_hinges=unlimited)

def inspect_pose(model,geometry,query,dz,seed=None,details=False):
    p=geometry.home_position.copy();p[2]+=dz;sol=inverse_kinematics(geometry,p,geometry.home_rotation,seed);row=dict(delta_z_m=float(dz),IK_REACHABLE=sol.success,JOINT_LIMIT_VALID=False,CONDITION_NUMBER_VALID=False,GEOMETRY_CLEARANCE_VALID=False,COLLISION_COVERAGE_INCOMPLETE=True,DYNAMICALLY_VALIDATED=False,closure_translation_m=sol.max_position_error,closure_rotation_rad=sol.max_rotation_error,slide_margin_m=sol.minimum_stroke_margin)
    if sol.success:
        scratch=scratch_reference_data(model,geometry,sol,p,geometry.home_rotation);j,_=actuator_jacobian(geometry,p,geometry.home_rotation,sol);health=singularity_diagnostics(j);lim=check_native_joint_limits(model,scratch.qpos);gap=query.evaluate(scratch.qpos,details)
        row.update(JOINT_LIMIT_VALID=lim['valid'],CONDITION_NUMBER_VALID=health['normalized_condition'] is not None and health['normalized_condition']<=50,GEOMETRY_CLEARANCE_VALID=gap['minimum_m']>=.005,jacobian_condition=health['normalized_condition'],expanded_mesh_gap_m=gap['minimum_m'],cross_branch_gap_m=gap['cross_branch_m'],platform_gap_m=gap['platform_gap_m'],nearest_pair=gap['nearest_pair'],minimum_is_lower_bound=gap['minimum_is_lower_bound'],unlimited_passive_count=len(lim['unlimited_hinges']),passive_limit_margin_rad=None,violations=gap['violating_pairs'])
        for i in range(6):
            row[f'slide_{i+1}_m']=float(sol.strokes[i])
            for k in [0,1,3,4,5]:row[f'passive_{i+1}_{k}_rad']=float(sol.chain_q[i,k])
        row['simulation_geometric_candidate']=all(row[k] for k in ['IK_REACHABLE','JOINT_LIMIT_VALID','CONDITION_NUMBER_VALID','GEOMETRY_CLEARANCE_VALID'])
    else:row['simulation_geometric_candidate']=False
    return row,sol
