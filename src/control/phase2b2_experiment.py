"""Additive Phase2B2 experiment adapters. Frozen physics step and controllers reused."""
import json
from collections import deque
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ..model import ROOT,write_json,ids
from ..phase1.plant import plant_metadata
from ..kinematics.jacobian import singularity_diagnostics
from .phase2b1_experiment import RobustOwner
from .residual_action import CausalAction,ResidualController

def actual_force_map(m,d):
    eq=np.asarray(d.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    if mujoco.mj_isSparse(m):
        jac=np.zeros((d.nefc,m.nv))
        for i in range(d.nefc):
            a=int(d.efc_J_rowadr[i]);n=int(d.efc_J_rownnz[i]);jac[i,d.efc_J_colind[a:a+n]]=d.efc_J[a:a+n]
    else:jac=np.asarray(d.efc_J).reshape(d.nefc,m.nv)
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,m.site('upper_center').id);t=np.linalg.solve(np.r_[jac[eq],jp,jr],np.r_[np.zeros((36,6)),np.eye(6)])
    return t[ids(m)[1]] # Unit direct-force transmission, excluding actual gain.

class MeasurementAdapter:
    def __init__(self,base,spec=None):
        self.base=base;self.spec=spec or {}
        values=[self.spec.get(k,0.) for k in ('position_std_m','rotation_std_rad','velocity_std')]
        if not np.isfinite(values).all() or any(x<0 for x in values) or sum(x!=0 for x in values)+bool(self.spec.get('delay_steps',0))>1:raise ValueError('Single small measurement factor required')
        if any(v>limit for v,limit in zip(values,[2e-6,2e-5,2e-5])):raise ValueError('Measurement candidate exceeds pilot range')
        self.queue=deque(maxlen=8);self.rng=np.random.default_rng(self.spec.get('seed',42));self.n=0;self.sensor_trace={}
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset();self.queue.clear();self.n=0;self.sensor_trace={}
    def command(self,qpos,qvel,p,r,v,ref,t):
        ps=float(self.spec.get('position_std_m',0));rs=float(self.spec.get('rotation_std_rad',0));vs=float(self.spec.get('velocity_std',0));delay=int(self.spec.get('delay_steps',0))
        if not 0<=delay<=5:raise ValueError('Delay outside preflight range')
        noise=self.rng.normal(size=12) if ps or rs or vs else np.zeros(12)
        measured_p=p+noise[:3]*ps if ps else p.copy();measured_r=Rotation.from_rotvec(noise[3:6]*rs).as_matrix()@r if rs else r.copy();measured_v=v+noise[6:]*vs if vs else v.copy()
        self.queue.append((self.n,measured_p.copy(),measured_r.copy(),measured_v.copy(),noise.copy()));sample=self.queue[max(0,len(self.queue)-1-delay)]
        self.sensor_trace=dict(capture_step=sample[0],read_step=self.n,noise=sample[4],measured_p=sample[1],measured_rotation=sample[2].ravel(),measured_velocity=sample[3]);self.n+=1
        return self.base.command(qpos,qvel,sample[1],sample[2],sample[3],ref,t)

class PrivilegedOracle:
    privileged=True;space='leg';frequency=1000
    def __init__(self,owner,base):self.owner=owner;self.base=base;self.calls=0
    def request(self,observation,j):
        self.calls+=1;local=self.owner.profile.command(self.owner.n,self.owner.data.site_xmat[self.owner.site_id].reshape(3,3));r=self.owner.data.site_xmat[self.owner.site_id].reshape(3,3);world=np.r_[r@local[:3],r@local[3:]];ideal=np.linalg.solve(j.T,-world)
        own=np.asarray(self.base.diagnostics['compensation_N']) if self.base.name=='C3' else np.zeros(6)
        return ideal-own

class ResidualOwner(RobustOwner):
    def __init__(self,config,*channels):
        super().__init__(config,*channels);self.perturbation=config.get('actual_perturbation',{});m=self.model;b=m.body('upper').id;before=plant_metadata(m,'nominal')
        if len(self.perturbation)>1:raise ValueError('Only one actual-plant factor per trial')
        factor=self.perturbation.get('mass_scale',1.);m.body_mass[b]*=factor;m.body_inertia[b]*=factor;m.body_ipos[b,0]+=self.perturbation.get('com_x_m',0.)
        gains=np.asarray(self.perturbation.get('gain_scales',[1.]*6));m.actuator_gainprm[:6,0]*=gains
        if self.perturbation:mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_forward(m,self.data)
        after=plant_metadata(m,'phase2b2_actual');after['nominal_plant']=before;after['actual_perturbation']=self.perturbation;after['external_channel_policy']='Explicit independent disturbance_profile only';write_json(Path(config['folder'])/'plant.json',after)
        self.snapshot();write_json(Path(config['folder'])/'initial_state.json',dict(qpos=self.data.qpos.tolist(),qvel=self.data.qvel.tolist(),integration_state=self.state.tolist()))
        action_spec=config.get('action',dict(kind='zero',space='task',frequency=20));base=self.controller
        if config.get('use_residual',True):
            privileged=action_spec['kind']=='oracle'
            if privileged and config['stage']!='oracle_diagnostic':raise ValueError('Oracle prohibited outside privileged diagnostic stage')
            action=PrivilegedOracle(self,base) if privileged else CausalAction(**action_spec)
            base=ResidualController(base,action,allow_privileged=privileged)
        self.residual=base if isinstance(base,ResidualController) else None;self.controller=MeasurementAdapter(base,config.get('measurement'));self.true_map=actual_force_map(m,self.data)
        append=self.recorder.append
        def log(row,state):
            trace=self.residual.trace if self.residual else {};active=not self.failure;force=np.array([row[f'force_{j}'] for j in range(6)]);core=self.residual.base if self.residual else base
            f0=trace.get('F_C0',getattr(core,'diagnostics',{}).get('base_force_N',self.last_force));fb=trace.get('F_base',self.last_force);native_gain=self.model.actuator_gainprm[:6,0]
            values=dict(F_C0=f0,F_C3=trace.get('F_C3',self.last_force if self.config['controller']=='C3' else np.zeros(6)),F_residual_requested=trace.get('requested',np.zeros(6)),F_residual_applied=trace.get('applied',np.zeros(6)),F_add_total=trace.get('total_add',self.last_force-f0),W_requested=trace.get('W_requested',np.zeros(6)),W_realized=trace.get('W_realized_nominal',np.zeros(6)),W_add_realized=trace.get('W_add_nominal',np.zeros(6)),W_physical_residual=self.true_map.T@(force-np.clip(native_gain*fb,-400,400)),W_physical_add=self.true_map.T@(force-np.clip(native_gain*f0,-400,400)),true_kinematic_map=self.true_map.ravel())
            sensor=self.controller.sensor_trace
            for key,n in [('noise',12),('measured_p',3),('measured_rotation',9),('measured_velocity',6)]:values['sensor_'+key]=sensor.get(key,np.zeros(n))
            for key,value in values.items():
                for j,x in enumerate(value):row[f'{key}_{j}']=float(x)
            row.update(budget_applicable=active and self.config['controller'] in ('C0','C3'),action_rejection_reason=trace.get('reason','baseline' if active else 'PROTECTED HOLD: see failure/events'),action_decision=trace.get('decision',False),action_decision_index=trace.get('decision_index',-1),observer_command_corrected=trace.get('observer_command_corrected',False),sensor_capture_step=sensor.get('capture_step',-1),sensor_read_step=sensor.get('read_step',-1),actual_jacobian_condition=float(singularity_diagnostics(self.true_map)['normalized_condition']))
            append(row,state)
        self.recorder.append=log;write_json(Path(config['folder'])/'action_profile.json',dict(**action_spec,privileged=action_spec['kind']=='oracle',label='PRIVILEGED ORACLE DIAGNOSTIC' if action_spec['kind']=='oracle' else 'NONPRIVILEGED CAUSAL DIAGNOSTIC',shared_additional_limit_N=5,shared_additional_rate_N_s=20))
    def step(self):
        self.true_map=actual_force_map(self.model,self.data);health=singularity_diagnostics(self.true_map)
        if not self.failure and (health['normalized_condition'] is None or health['normalized_condition']>50):self.hold('actual_jacobian_condition',failed=True)
        super().step()
    def run(self):
        result=super().run();events=[json.loads(x) for x in (Path(self.config['folder'])/'events.jsonl').read_text(encoding='utf-8').splitlines()];write_json(Path(self.config['folder'])/'safety_events.json',[x for x in events if x['type'] in ('protect_hold','exception','target_rejected')]);return result
