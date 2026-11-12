"""One shared additional-force budget relative to frozen C0, not per-controller budgets."""
import numpy as np

class SharedBudgetGovernor:
    def __init__(self,amplitude_N=5.,rate_N_s=20.,dt=.001):
        if amplitude_N!=5 or rate_N_s!=20 or dt!=.001:raise ValueError('This study freezes the original shared budget')
        self.amplitude=amplitude_N;self.rate=rate_N_s;self.dt=dt;self.previous=np.zeros(6)
    def reset(self):self.previous[:]=0
    def apply(self,c0,base,residual):
        c0=np.asarray(c0,float);base=np.asarray(base,float);residual=np.asarray(residual,float)
        if any(x.shape!=(6,) for x in (c0,base,residual)) or not np.isfinite(np.r_[c0,base,residual]).all():raise ValueError('Invalid force request')
        lo=np.maximum.reduce([np.full(6,-5.),self.previous-.02,-400-c0]);hi=np.minimum.reduce([np.full(6,5.),self.previous+.02,400-c0])
        if np.any(lo>hi+1e-10):raise ValueError('Shared budget/native headroom infeasible')
        desired=(base-c0)+residual
        # Original C3 already has ulp-level cancellation in total-base. Keep the
        # exact original output for zero inputs when numerically feasible.
        if not np.any(residual) and np.all(desired>=lo-1e-10) and np.all(desired<=hi+1e-10):cmd=base.copy();reason='zero_exact_pass_through'
        else:
            projected=np.clip(desired,lo,hi);cmd=np.clip(c0+projected,-400,400);reason='projected_shared_budget' if np.any(projected!=desired) else 'accepted'
        total=cmd-c0
        if abs(total).max()>5+1e-10 or abs(total-self.previous).max()>.02+1e-10:raise ValueError('Shared constraint numerical violation')
        self.previous=total.copy()
        return cmd,dict(total_add=total,applied_residual=cmd-base,requested_total_add=desired,reason=reason,budget_margin=5-abs(total))
