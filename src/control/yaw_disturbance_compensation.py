"""C4-Observer reads only the frozen C3 causal local Mz estimate."""
import numpy as np

class YawObserverCompensation:
    privileged=False;frequency=1000;name='C4-Observer'
    def __init__(self,gain=1.,limit=.30):
        if gain not in (.5,1.) or not 0<limit<=.30:raise ValueError('Outside frozen observer search')
        self.gain=gain;self.limit=limit
    def reset(self):pass
    def request(self,o):return np.array([0.,0.,0.,np.clip(-self.gain*o.equivalent_disturbance[5],-self.limit,self.limit)])

class C4WithLead:
    privileged=False;frequency=1000;name='C4+causal4d20'
    def __init__(self,base):self.base=base;self.reset()
    def reset(self):self.base.reset();self.k=0;self.previous=None;self.held=np.zeros(4)
    def request(self,o):
        if self.k%50==0:
            estimate=o.equivalent_disturbance[2:6]
            self.held=np.zeros(4) if self.previous is None else np.clip(-.02*(estimate-self.previous)*20,[-.5,-.025,-.025,-.015],[.5,.025,.025,.015])
            self.previous=estimate.copy()
        self.k+=1
        result=self.base.request(o)+self.held
        result[3]=np.clip(result[3],-.30,.30)
        return result
