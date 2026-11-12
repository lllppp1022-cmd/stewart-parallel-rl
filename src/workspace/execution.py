"""Independent safety envelope around frozen integration and frozen controllers."""
import types,copy
import numpy as np
from ..control.phase2c0_experiment import PhysicalOwner
from ..control.phase2b2_experiment import actual_force_map
from ..dynamics.execution_hook import PLANT_STEP
from ..phase1.executor import check_safety
from ..kinematics.jacobian import singularity_diagnostics
from .geometry import ExpandedGeometryQuery,check_native_joint_limits

class LargeOwner(PhysicalOwner):
    def __init__(self,config,reference,query,workspace):
        super().__init__(config);self.active=reference;self.ref=reference.at(0);self.expanded_query=query;self.large_limits=copy.deepcopy(workspace);self.expanded_gap=query.evaluate(self.data.qpos);self.latest_query_step=-1
        core=self.residual.base;target=core.base if core.name=='C3' else core;target.config=copy.deepcopy(target.config)
        # Gains/physical constraints unchanged. Asymmetric Z checked separately.
        target.config['actual_z_limit_m']=max(abs(x) for x in workspace['actual_z_bounds_m'])
        self.safety_config=copy.deepcopy(target.config)
        def safety(m,d,p,r,g,wt,wr,condition,proxy,cfg):
            why=check_safety(m,d,p,r,g,wt,wr,condition,proxy,cfg)
            if why:return why
            dz=float(p[2]-g.home_position[2]);lo,hi=self.large_limits['actual_z_bounds_m']
            if not lo<=dz<=hi:return 'audited_asymmetric_large_z_envelope'
            if not check_native_joint_limits(m,d.qpos)['valid']:return 'native_passive_or_slide_limit'
            if self.n%20==0 or self.latest_query_step<0:self.expanded_gap=self.expanded_query.evaluate(d.qpos);self.latest_query_step=self.n
            if self.expanded_gap['minimum_m']<.005:
                if not self.failure:self.event('expanded_geometry_violation',nearest_pair=self.expanded_gap['nearest_pair'],gap_m=self.expanded_gap['minimum_m'])
                return 'expanded_convex_mesh_clearance'
            return None
        namespace=dict(PLANT_STEP.__globals__);namespace['check_safety']=safety
        self._large_step=types.FunctionType(PLANT_STEP.__code__,namespace,'independent_large_step')
        append=self.recorder.append
        def log(row,state):
            lim=check_native_joint_limits(self.model,self.data.qpos);core=self.residual.base
            row.update(expanded_mesh_gap_m=self.expanded_gap['minimum_m'],expanded_gap_is_lower_bound=self.expanded_gap['minimum_is_lower_bound'],expanded_query_step=self.latest_query_step,unlimited_passive_count=len(lim['unlimited_hinges']),native_limit_margin=lim['minimum_native_margin'],reference_z_from_home_m=self.ref.p[2]-self.geometry.home_position[2],actual_z_from_home_m=self.data.site_xpos[self.site_id,2]-self.geometry.home_position[2],reference_z_acceleration_m_s2=reference.scalar(self.data.time)[2],workspace_version='phase2c3a_independent')
            append(row,state)
        self.recorder.append=log
    def step(self):
        self.true_map=actual_force_map(self.model,self.data);health=singularity_diagnostics(self.true_map)
        if not self.failure and (health['normalized_condition'] is None or health['normalized_condition']>50):self.hold('actual_jacobian_condition',failed=True)
        self._large_step(self)
