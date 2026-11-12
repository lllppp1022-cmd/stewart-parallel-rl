"""One-dimensional Actor through the frozen four-dimensional wrench mapping."""
import numpy as np
from ..control.residual_action import ResidualController
from ..control.task_residual_4d import task4_to_leg
from ..control.yaw_disturbance_compensation import YawObserverCompensation
from ..control.residual_governor import SharedBudgetGovernor

class HeldFzAction:
    privileged=False;space='leg';frequency=1000
    def __init__(self,scale_N):self.scale_N=float(scale_N);self.normalized=0.;self.calls=0;self.yaw=YawObserverCompensation(1);self.task=np.zeros(4);self.c4_leg=np.zeros(6)
    def set_action(self,value):
        a=np.asarray(value,float)
        if a.shape!=(1,) or not np.isfinite(a).all():raise ValueError('Actor must supply one finite normalized action')
        self.normalized=float(np.clip(a[0],-1,1))
    def request(self,o,j):
        self.calls+=1;baseline=self.yaw.request(o);self.c4_leg=task4_to_leg(baseline,j,1.)[0];self.task=baseline.copy();self.task[0]+=self.normalized*self.scale_N
        return task4_to_leg(self.task,j,1.)[0]

class LearningCorrection(ResidualController):
    def __init__(self,core,scale_N):
        self.held_action=HeldFzAction(scale_N);super().__init__(core,self.held_action);self.name='C4+Actor-Fz';self.c4_same_state_command=np.zeros(6);self.rl_applied_command=np.zeros(6);self.comparison_governor=SharedBudgetGovernor()
    def command(self,*args):
        previous=self.governor.previous.copy();command,info=super().command(*args);self.comparison_governor.previous=previous
        self.c4_same_state_command,_=self.comparison_governor.apply(self.trace['F_C0'],self.trace['F_base'],self.held_action.c4_leg)
        self.rl_applied_command=command-self.c4_same_state_command
        return command,info
