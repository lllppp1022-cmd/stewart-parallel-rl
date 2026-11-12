"""Phase2C3B-only reference timing, experiment construction and full-rate journal."""
import copy,json
from pathlib import Path
import numpy as np
from ..model import ROOT,write_json
from .reference import LargeReference
from .execution import LargeOwner
from .geometry import ExpandedGeometryQuery
from ..phase1.plant import create_plant
from ..control.phase2b2_experiment import MeasurementAdapter
from ..control.dynamic_reference_feedforward import DynamicReferenceFeedforward
from ..control.large_motion_feedforward_wrapper import LargeMotionFeedforwardWrapper
from ..rl.stewart_residual_env import owner_config

RUN=ROOT/'results/runs/phase2c3b_20261105_v1'

class StudyReference(LargeReference):
    def __init__(self,model,amplitude,rise_s,cache,query):
        if rise_s not in (4.,6.,8.,10.):raise ValueError('Frozen timings only')
        super().__init__(model,amplitude,rise_s if rise_s!=8. else 10.,cache,query)
        if rise_s==8.:
            self.rise_s=8.;self.duration=19.;self.name='L50_rise8_independent_timing';self.audit.update(rise_s=8.,duration_s=19.,max_speed_m_s=1.875*amplitude/8,max_acceleration_m_s2=10/np.sqrt(3)*amplitude/64,dense_reference_steps=19001)
            qs=np.array([self.at(t).q for t in np.arange(0,19.0001,.001)])
            if np.any(qs<self.geometry.stroke_ranges[:,0]) or np.any(qs>self.geometry.stroke_ranges[:,1]):raise ValueError('Independent timing stroke validation failed')

def make_owner(folder,method,amplitude,rise,tau=0.,load=0.,record=False):
    parameters=json.loads((ROOT/'results/phase2c3b_feedforward_parameters.json').read_text(encoding='utf-8'));q=ExpandedGeometryQuery();ref=StudyReference(create_plant(),amplitude,rise,RUN/'references',q);task=dict(id=Path(folder).name,tau_s=tau,Fz_N=load,on_step=round((1+.7*rise)*1000),off_step=round((2+1.5*rise)*1000));oc=owner_config(folder,task,record);oc.update(stage='phase2c3b',max_sim_seconds=ref.duration,max_wall_seconds=360,controller='C0' if method=='C0' else 'C3')
    if method=='C0':oc['yaw_policy']=dict(kind='zero4')
    if method in ('fourDlead','FF-VA-Lead'):oc['diagnostic']=dict(kind='four_d_lead')
    workspace=json.loads((ROOT/'configs/phase2c3b_feedforward_study.json').read_text(encoding='utf-8'))['workspace'];o=LargeOwner(oc,ref,q,workspace);o.config['trajectory']=ref.name;o.config['study_method']=method
    if method.startswith('FF-'):
        ff=DynamicReferenceFeedforward(parameters['kv_N_s_m'],0. if method=='FF-V' else parameters['ka_kg']);o.residual=LargeMotionFeedforwardWrapper(o.residual,ref,ff,method);o.controller=MeasurementAdapter(o.residual)
    append=o.recorder.append
    def log(row,state):
        inj=getattr(o.residual,'injection',None);a=inj.audit if inj and not o.failure else {};core=o.residual.base
        vectors=dict(ff_requested_leg_N=a.get('ff_leg',np.zeros(6)),ff_requested_wrench=a.get('ff_wrench',np.zeros(6)),ff_probe_wrench=a.get('probe_wrench',np.zeros(6)),ff_baseline_request_N=a.get('baseline_request',o.residual.trace.get('requested',np.zeros(6))),ff_c4_compensation_N=a.get('c4_compensation',np.array(core.diagnostics.get('compensation_N',np.zeros(6))) if core.name=='C3' else np.zeros(6)),ff_desired_total_add_N=a.get('desired_total_add',o.residual.trace.get('total_add',np.zeros(6))),ff_actual_total_add_N=a.get('actual_total_add',o.residual.trace.get('total_add',np.zeros(6))),ff_effect_command_N=a.get('ff_effect_command',np.zeros(6)),ff_probe_effect_N=a.get('probe_effect_command',np.zeros(6)),ff_realized_wrench=a.get('realized_wrench',np.zeros(6)),ff_total_add_rate_N_s=a.get('total_add_rate_N_s',np.zeros(6)))
        for k,v in vectors.items():
            for i,value in enumerate(v):row[f'{k}_{i}']=float(value)
        row.update(ff_velocity_force_N=float(a.get('velocity_force_N',0)),ff_acceleration_force_N=float(a.get('acceleration_force_N',0)),ff_budget_reason=a.get('budget_reason','frozen_baseline' if not o.failure else 'protected_hold'),ff_budget_peak_fraction=float(np.max(abs(vectors['ff_actual_total_add_N']))/5),study_method=method)
        append(row,state)
    o.recorder.append=log;write_json(Path(folder)/'config.json',o.config);write_json(Path(folder)/'reference_preflight.json',ref.audit);write_json(Path(folder)/'feedforward_parameters.json',dict(method=method,**parameters));return o,ref
