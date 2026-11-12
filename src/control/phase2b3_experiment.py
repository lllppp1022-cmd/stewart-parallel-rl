"""Additive Phase2B3 control installation; frozen physics, C3 and governor reused."""
import json
from pathlib import Path
import numpy as np
from .phase2b2_experiment import ResidualOwner,MeasurementAdapter
from .task_residual_4d import Task4Controller,ZeroOrProbe4
from .yaw_feedback_controller import YawFeedback,CausalFourD
from .yaw_disturbance_compensation import YawObserverCompensation,C4WithLead
from .residual_action import ResidualController
from ..model import write_json

def policy(spec):
    kind=spec['kind']
    if kind in ('zero4','probe4'):return ZeroOrProbe4(spec.get('vector'))
    if kind=='pd':return YawFeedback(spec['kp'],spec['kd'])
    if kind=='observer':return YawObserverCompensation(spec['gain'])
    if kind=='causal4':return CausalFourD(spec['kp'],spec['kd'])
    if kind=='c4lead':return C4WithLead(policy(spec['base']))
    raise ValueError('Unknown Phase2B3 policy')

class ScaledReference:
    def __init__(self,base,scale):
        if scale!=.85:raise ValueError('Only frozen slowed reference supported')
        self.base=base;self.scale=scale;self.duration=base.duration/scale
    def at(self,t):
        from dataclasses import replace
        r=self.base.at(t*self.scale);return replace(r,dq=r.dq*self.scale,twist=r.twist*self.scale)

class GainOracle:
    privileged=True;space='leg';frequency=1000
    def __init__(self,owner,base):self.owner=owner;self.base=base;self.calls=0
    def request(self,o,j):
        self.calls+=1;owner=self.owner;diag=self.base.diagnostics;c0=np.asarray(diag['base_force_N']);fb=c0+diag['compensation_N'];r=owner.data.site_xmat[owner.site_id].reshape(3,3)
        local=owner.profile.command(owner.n,r);external=np.r_[r@local[:3],r@local[3:]]
        # Privileged actual gain, actual tangent and true current load. Never a policy input.
        g=owner.model.actuator_gainprm[:6,0];ideal=np.linalg.solve(owner.true_map.T*g[None,:],j.T@c0-external)
        return ideal-fb

class YawOwner(ResidualOwner):
    def __init__(self,config,*channels):
        if config.get('measurement') or set(config.get('actual_perturbation',{})) - {'gain_scales'}:raise ValueError('Phase2B3 gain-only, no noise/delay/other factors')
        if config.get('reference_scale',1.) not in (1.,.85):raise ValueError('Unsupported reference')
        spec=config.get('yaw_policy',dict(kind='zero4'));basecfg=dict(config,use_residual=False,action=dict(kind='zero',space='leg',frequency=20))
        super().__init__(basecfg,*channels);self.config=dict(basecfg,yaw_policy=spec)
        core=self.controller.base
        if spec['kind']=='oracle':
            if config['stage']!='oracle_diagnostic':raise ValueError('Privileged oracle excluded from formal methods')
            wrapper=ResidualController(core,GainOracle(self,core),allow_privileged=True);wrapper.name='PRIVILEGED ORACLE DIAGNOSTIC'
        else:wrapper=Task4Controller(core,policy(spec))
        self.residual=wrapper;self.controller=MeasurementAdapter(wrapper)
        if config.get('reference_scale',1.)!=1:self.active=ScaledReference(self.active,config['reference_scale'])
        append=self.recorder.append
        def log(row,state):
            adapter=getattr(wrapper,'adapter',None);target=adapter.target if adapter else np.zeros(4)
            for k,v in enumerate(target):row[f'task4_target_{k}']=float(v)
            row['task4_decision']=adapter.decision if adapter else True;row['task4_policy_version']=adapter.policy_version if adapter else 0
            # Physical-force difference uses actual current force and same-state nominal cmd.
            force=np.array([row[f'force_{i}'] for i in range(6)]);command=np.array([row[f'f_cmd_{i}'] for i in range(6)])
            w=self.true_map.T@(force-command)
            for i,value in enumerate(w):row[f'gain_wrench_error_{i}']=float(value)
            append(row,state)
        self.recorder.append=log
        write_json(Path(config['folder'])/'config.json',self.config)
        write_json(Path(config['folder'])/'action_profile.json',dict(spec=spec,label='PRIVILEGED ORACLE DIAGNOSTIC' if spec['kind']=='oracle' else 'CLASSICAL CAUSAL YAW CONTROL',reference='world / upper_center',order=['Fz_N','Mx_Nm','My_Nm','Mz_Nm'],shared_extra_N=5,shared_extra_rate_N_s=20))
