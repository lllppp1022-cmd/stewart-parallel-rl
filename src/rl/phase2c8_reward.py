"""Physical Reward-V3, with no event labels or true disturbance inputs."""
import numpy as np
NAMES=('orientation','omega','xyz','action','slew')
def costs(row,accepted,slew,version='v3',alpha=.5,beta=.5,epsilon=.01):
    e=float(row['orientation_error_deg'])/.006
    orientation=e*e if version=='v2_control' else alpha*(np.hypot(e,epsilon)-epsilon)+beta*e*e
    omega=np.array([row[f'twist_actual_{j}']-row[f'twist_ref_{j}'] for j in (3,4,5)])
    xyz=np.array([row[f'p_{j}']-row[f'p_ref_{j}'] for j in range(3)])*1000
    values=np.array([orientation,.2*np.sum((omega/.003)**2),.1*np.sum((xyz/[.2,.2,.05])**2),.001*np.sum((accepted/.05)**2),.000025*np.sum((slew/2.5)**2)])
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Nonfinite or negative physical cost')
    factor=min(1.,100/max(values.sum(),1e-30));return values*factor,factor
