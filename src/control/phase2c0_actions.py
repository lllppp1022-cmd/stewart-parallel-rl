"""Finite local diagnostics, no networks; all nonprivileged inputs are causal."""
import numpy as np

class TaskDiagnostic:
    privileged=False;frequency=1000;space='leg'
    def __init__(self,spec,core):self.spec=spec;self.core=core;self.name=spec['kind'];self.calls=0;self.previous=None;self.filtered_rate=np.zeros(6);self.last_estimate=None;self.held=np.zeros(6)
    def request(self,o,j):
        k=self.calls;self.calls+=1;w=np.zeros(6);w[5]=np.clip(-o.equivalent_disturbance[5],-.30,.30)
        kind=self.spec['kind']
        if kind=='probe':
            if 2.2<=k*.001<3.2:w+=np.asarray(self.spec['wrench'],float)
        elif kind=='xy_feedback':
            e=o.p-o.reference_p;v=o.velocity-o.reference_velocity;w[:2]=np.clip(-300*e[:2]-10*v[:2],-.04,.04)
        elif kind=='four_d_lead':
            if k%50==0:
                estimate=o.equivalent_disturbance[2:6];a=np.zeros(4) if self.last_estimate is None else np.clip(-.02*(estimate-self.last_estimate)*20,[-.5,-.025,-.025,-.015],[.5,.025,.025,.015]);self.last_estimate=estimate.copy();self.held=np.r_[0,0,a]
            w+=self.held;w[5]=np.clip(w[5],-.30,.30)
        elif kind!='fixed_command_lead':raise ValueError('Unknown causal diagnostic')
        residual=np.linalg.solve(j.T,w)
        if kind=='fixed_command_lead':
            d=self.core.diagnostics;current=d['base_force_N']+d['compensation_N']+residual
            rate=np.zeros(6) if self.previous is None else (current-self.previous)/.001;self.previous=current.copy();self.filtered_rate+=(-np.expm1(-.001/.004))*(rate-self.filtered_rate)
            residual+=np.clip(.003*self.filtered_rate,-.1,.1)
        # Predeclared local command ceiling; the shared governor remains tighter.
        return np.clip(residual,-.95,.95)

class PrivilegedPhysical:
    privileged=True;frequency=1000;space='leg';name='PRIVILEGED PHYSICAL DIAGNOSTIC'
    def __init__(self,core,owner):self.core=core;self.owner=owner;self.calls=0
    def request(self,o,j):
        self.calls+=1;x=self.owner;d=self.core.diagnostics;r=x.data.site_xmat[x.site_id].reshape(3,3);local=x.profile.command(x.n,r);external=np.r_[r@local[:3],r@local[3:]]
        desired=np.linalg.solve(x.true_map.T*x.model.actuator_gainprm[:6,0][None,:],j.T@d['base_force_N']-external)
        alpha=x.lag.alpha if x.lag.enabled else np.ones(6);command=(desired-(1-alpha)*x.lag.state)/alpha;fb=d['base_force_N']+d['compensation_N']
        return np.clip(command-fb,-5,5)

def make_action(spec,core,owner):
    if spec['kind']=='privileged':
        if owner.config['stage']!='privileged_diagnostic':raise ValueError('Privileged physical diagnostic excluded from formal methods')
        return PrivilegedPhysical(core,owner)
    return TaskDiagnostic(spec,core)
