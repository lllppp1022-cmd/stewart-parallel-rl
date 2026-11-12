"""Inject a known-reference request BEFORE the ONE frozen shared-budget governor.

The frozen baseline computes its original action/observer exactly once. Its
governor interface is wrapped, not stacked with a second independent budget.
The original final-command observer correction therefore sees the true command.
"""
import numpy as np
from .residual_governor import SharedBudgetGovernor

class ReferenceBudgetInjection:
    def __init__(self,original,core,reference,feedforward):
        self.original=original;self.core=core;self.reference=reference;self.feedforward=feedforward;self.time=0.;self.probe_N=0.;self.audit={};self.counter=SharedBudgetGovernor()
    def __getattr__(self,name):return getattr(self.original,name)
    def reset(self):self.original.reset();self.audit={};self.time=0.;self.probe_N=0.
    def apply(self,c0,base,residual):
        j=self.core.jacobian;ff,w,parts=self.feedforward.leg_force(self.reference,self.time,j);probe_w=np.array([0.,0.,self.probe_N,0.,0.,0.]);probe=np.linalg.solve(j.T,probe_w);extra=ff+probe
        previous=self.original.previous.copy();self.counter.previous=previous.copy();without_ff,_=self.counter.apply(c0,base,residual)
        self.counter.previous=previous.copy();without_probe,_=self.counter.apply(c0,base,residual+ff)
        command,g=self.original.apply(c0,base,residual+extra)
        self.audit=dict(ff_leg=ff,ff_wrench=w,probe_wrench=probe_w,baseline_request=residual.copy(),c4_compensation=base-c0,desired_total_add=(base-c0)+residual+extra,actual_total_add=command-c0,ff_effect_command=command-without_ff,probe_effect_command=command-without_probe,realized_wrench=j.T@(command-c0),budget_reason=g['reason'],budget_margin=g['budget_margin'],total_add_rate_N_s=((command-c0)-previous)/.001,**parts)
        return command,g

class LargeMotionFeedforwardWrapper:
    def __init__(self,baseline,reference,feedforward,name):
        self.baseline=baseline;self.injection=ReferenceBudgetInjection(baseline.governor,baseline.base,reference,feedforward);baseline.governor=self.injection;self.name=name
    def __getattr__(self,name):return getattr(self.baseline,name)
    @property
    def base(self):return self.baseline.base
    @property
    def trace(self):return self.baseline.trace
    def reset(self):self.baseline.reset()
    def command(self,*args):
        self.injection.time=float(args[-1]);return self.baseline.command(*args)
