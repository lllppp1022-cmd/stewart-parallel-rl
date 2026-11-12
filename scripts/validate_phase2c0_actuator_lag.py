"""Standalone analytic ZOH validation, no robot or extra mj_step."""
import _bootstrap
import numpy as np,pandas as pd
from src.dynamics.actuator_lag import ActuatorLag
from src.model import ROOT,write_json
if __name__=='__main__':
    rows=[]
    for tau in [0,.002,.005,.01]:
        initial=np.full(6,85.);lag=ActuatorLag(tau,initial);expected=initial.copy()
        for k in range(1000):
            command=np.full(6,85.+(.1 if 100<=k<300 else -.1 if 400<=k<700 else 0));drive=lag.advance(command);expected=command.copy() if tau==0 else command+(expected-command)*np.exp(-.001/tau)
            np.testing.assert_allclose(drive,expected,rtol=0,atol=2e-13);rows.append(dict(tau_s=tau,step=k,time_s=(k+1)*.001,command_N=command[0],drive_N=drive[0],analytic_N=expected[0],error_N=abs(drive[0]-expected[0])))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c0_actuator_lag_validation.csv',index=False);write_json(ROOT/'results/phase2c0_lag_unit_validation.json',dict(samples=len(rows),maximum_error_N=max(x['error_N'] for x in rows),mj_step_calls=0,dt_s=.001,units='N and seconds',method='Exact ZOH, endpoint applied throughout next numerical integration step, one update per step'))
    print('Actuator analytic ZOH passed',len(rows))
