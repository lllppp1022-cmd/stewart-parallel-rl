"""Physical integral reward on every interval, no disturbance flag or truth input."""
import numpy as np

def physical_cost(position_error_m,so3_error_rad,applied_rl_N,previous_applied_rl_N,requested_fz_N,action_scale_N,cfg):
    e=np.asarray(position_error_m);a=np.asarray(applied_rl_N);rate=(a-np.asarray(previous_applied_rl_N))/.001
    parts=dict(z=cfg['z_weight']*(e[2]/cfg['z_scale_m'])**2,xy=cfg['xy_weight']*np.sum((e[:2]/cfg['xy_scale_m'])**2),angle=cfg['angle_weight']*(so3_error_rad/cfg['angle_scale_rad'])**2,amplitude=cfg['amplitude_weight']*np.mean((a/cfg['residual_scale_N'])**2),slew=cfg['slew_weight']*np.mean((rate/cfg['residual_rate_scale_N_s'])**2),request=cfg['request_weight']*(requested_fz_N/action_scale_N)**2)
    total=float(sum(parts.values()));factor=min(1.,cfg['maximum_physical_cost']/max(total,1e-30));parts={k:float(v*factor) for k,v in parts.items()}
    if not np.isfinite(list(parts.values())).all():raise ValueError('Nonfinite physical reward')
    return parts,factor

def failure_penalty(failure_time_s,cfg,horizon_s=6.):
    # Worse than the maximum possible remaining cost from this same prefix,
    # including under gamma<=1. No future physical outcome is queried.
    remaining=max(0.,horizon_s-failure_time_s)/.05
    return -float(cfg['failure_base_penalty']+cfg['maximum_physical_cost']*np.ceil(remaining))
