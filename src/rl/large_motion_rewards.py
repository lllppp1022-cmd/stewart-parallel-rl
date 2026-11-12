"""Physical-time cost, no load flags; bounded failure ranking including discount."""
import numpy as np

CONFIG=dict(z_scale_m=1e-5,xy_scale_m=5e-5,angle_scale_rad=.00034906585039886593,z_weight=1.,xy_weight=.1,rotation_weight=.05,action_weight=.01,slew_weight=.005,cost_cap=100.,failure_extra=100.,gamma=.99)

def costs(row,requested,cfg=CONFIG):
    e=np.array([row[f'p_{i}']-row[f'p_ref_{i}'] for i in range(3)]);slew=np.array([row[f'ff_total_add_rate_N_s_{i}'] for i in range(6)])
    p=dict(z=cfg['z_weight']*(e[2]/cfg['z_scale_m'])**2,xy=cfg['xy_weight']*sum((e[:2]/cfg['xy_scale_m'])**2),rotation=cfg['rotation_weight']*(np.deg2rad(row['orientation_error_deg'])/cfg['angle_scale_rad'])**2,action=cfg['action_weight']*(requested/.1)**2,slew=cfg['slew_weight']*float(np.mean((slew/20)**2)))
    factor=min(1.,cfg['cost_cap']/max(sum(p.values()),1e-30));p={k:float(x*factor) for k,x in p.items()}
    if not np.isfinite(list(p.values())).all():raise ValueError('Nonfinite cost')
    return p,factor

def failure_penalty(remaining_decisions,cfg=CONFIG):
    # At any prefix exceeds100 units/decision maximum discounted future cost,
    # plus original200ms protective hold. Strict even if termination is immediate.
    g=cfg['gamma'];bound=cfg['cost_cap']*(1-g**(max(0,remaining_decisions)+4))/(1-g)
    return -float(cfg['failure_extra']+bound)

