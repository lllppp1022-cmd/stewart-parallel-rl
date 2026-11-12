"""Bounded physical interface gate, independent of policy outcomes."""
import time
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from phase2c7_protocol import RUN,config,read,write_json
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv,dynamic_cost
from src.workspace.dynamic_attitude_study import make_owner


def main():
    admission=read(RUN/'admission.json')
    assert admission['passed']
    task=next(t for t in admission['admitted'] if t['id']=='C2')
    target=RUN/'environment_gate'
    target.mkdir(exist_ok=False)
    start=time.perf_counter()
    env=StewartDynamicAttitudeEnv(target/'gym',[task])
    try:
        check_env(env,skip_render_check=True)
        obs,_=env.reset(seed=12)
        original=obs.copy()
        initial=env.owner.data.qpos.copy()
        env.step(np.array([.5,-.5]))
        reset,_=env.reset(seed=12)
        np.testing.assert_array_equal(reset,original)
        np.testing.assert_array_equal(initial,env.owner.data.qpos)
        np.testing.assert_array_equal(env.port.request,[0,0])
        # Hidden plant state and future exogenous schedule never enter observe.
        before=env.observe()
        drive=env.owner.lag.state.copy()
        env.owner.lag.state+=.001
        np.testing.assert_array_equal(before,env.observe())
        env.owner.lag.state[:]=drive
        env.owner.hold('interface_gate_protect',failed=True)
        _,reward,term,trunc,info=env.step(np.zeros(2))
        assert term and not trunc and np.isfinite(reward)
        assert info['physical_steps']>=20
    finally:env.close()
    # Full 15 s combined path: direct frozen coordinator vs exact zero RL port.
    owner,ref=make_owner(target/'direct',task)
    try:
        for _ in range(round(ref.duration*1000)):
            owner.step()
            assert not owner.failure
    finally:owner.recorder.close(dict(source='ZERO EQUIVALENCE DIRECT BASELINE'))
    baseline=RUN/'preflight'/task['id']/'baseline'
    a=pd.read_csv(target/'direct'/'physics.csv.gz')
    b=pd.read_csv(baseline/'episode_0000'/'physics.csv.gz')
    cols=[c for c in a if c.startswith(('qpos_','qvel_','F_command_','F_drive_','force_','p_','twist_actual_'))]
    assert len(cols)>20 and len(a)==len(b)==15000
    np.testing.assert_array_equal(a[cols].to_numpy(),b[cols].to_numpy())
    # Recompute every reward component from the actual recorded physical rows.
    total=[]
    prev=np.zeros(2)
    for row in b.to_dict('records'):
        accepted=np.array([row['accepted_mx_Nm'],row['accepted_my_Nm']])
        cost,_=dynamic_cost(row,accepted,(accepted-prev)*1000)
        total.append(cost.sum())
        prev=accepted
    decisions=pd.read_csv(baseline/'training_samples.csv')
    np.testing.assert_allclose(-np.asarray(total).reshape(-1,20).mean(1),decisions.reward,rtol=1e-10,atol=1e-10)
    random=pd.read_csv(RUN/'preflight'/task['id']/'random'/'episode_0000'/'physics.csv.gz')
    accepted=random[['accepted_mx_Nm','accepted_my_Nm']].to_numpy()
    effect=float(abs(random[[f'force_{i}' for i in range(6)]].to_numpy()-b[[f'force_{i}' for i in range(6)]].to_numpy()).max())
    assert abs(accepted).max()>1e-6 and effect>1e-4
    write_json(RUN/'training_gate.json',dict(passed=True,gym=True,reset_exact=True,zero_residual_bitwise=True,
        equivalence_physical_steps=len(a),equivalence_columns=cols,reward_recomputed=True,
        failure_semantics=True,hidden_drive_not_observed=True,actual_force_effect_N=effect,
        accepted_peak_Nm=float(abs(accepted).max()),wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
