import inspect
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase2c7_protocol import ROOT,RUN,config,read
from src.workspace.dynamic_attitude_study import DynamicReference
from src.rl.stewart_dynamic_attitude_env import DIM,FIELDS,StewartDynamicAttitudeEnv,dynamic_cost


@pytest.mark.parametrize('task',config('curriculum')['C'])
def test_combined_reference_derivatives(task):
    ref=DynamicReference(task['reference'])
    for t in (1.8,3.4,8.8,11.2):
        h=1e-5;a,b=ref.at(t-h),ref.at(t+h);center=ref.at(t)
        np.testing.assert_allclose((b.p-a.p)/(2*h),center.twist[:3],atol=1e-9)
        np.testing.assert_allclose(Rotation.from_matrix(b.r@a.r.T).as_rotvec()/(2*h),center.twist[3:],atol=1e-9)
        np.testing.assert_allclose((b.q-a.q)/(2*h),center.dq,atol=1e-8)
    np.testing.assert_allclose(ref.at(0).p,ref.at(ref.duration).p,atol=1e-14)
    np.testing.assert_allclose(ref.at(7.5).p[2]-ref.at(0).p[2],.05,atol=1e-14)


def test_observation_no_privileged_physics():
    assert DIM==58 and config('env')['observation_dim']==58
    body=inspect.getsource(StewartDynamicAttitudeEnv.observe)
    for forbidden in ('self.task','o.profile','o.lag','actuator_force','external_world','F_drive'):
        assert forbidden not in body
    assert 'reference_z' in FIELDS


@pytest.mark.parametrize('axis,scale',[(0,.0002),(1,.0002),(2,.00005)])
def test_reward_xyz_scales(axis,scale):
    row={f'{prefix}_{j}':0. for prefix in ('p','p_ref','twist_actual','twist_ref') for j in range(6)}
    row['orientation_error_deg']=0.;row[f'p_{axis}']=scale
    cost,factor=dynamic_cost(row,np.zeros(2),np.zeros(2))
    np.testing.assert_allclose(cost,[0,0,.1,0,0]);assert factor==1


def test_reward_cap_and_so3():
    row={f'{prefix}_{j}':0. for prefix in ('p','p_ref','twist_actual','twist_ref') for j in range(6)}
    row['orientation_error_deg']=.006
    cost,_=dynamic_cost(row,np.zeros(2),np.zeros(2));assert cost[0]==1
    row['orientation_error_deg']=20
    cost,_=dynamic_cost(row,np.zeros(2),np.zeros(2));assert cost.sum()==pytest.approx(100)


def test_task_partition_and_budget():
    signature=lambda t:json.dumps({k:v for k,v in t.items() if k not in ('id','stage')},sort_keys=True)
    c=config('curriculum');training=sum([c[k] for k in ('A','B','C')],[])
    assert set(map(signature,training)).isdisjoint(map(signature,config('eval')['final']))
    assert config('ppo')['total']==32768 and config('ppo')['seeds']==[42,123,2026]
    assert max(np.linalg.norm(t['torque']) for t in training)<=1.2+1e-12


def test_environment_real_step_reset_and_action(tmp_path):
    env=StewartDynamicAttitudeEnv(tmp_path/'env',[config('curriculum')['A'][0]])
    try:
        first,_=env.reset(seed=42)
        _,reward,term,trunc,info=env.step(np.array([2.,-2.]))
        np.testing.assert_array_equal(env.port.request,[.05,-.05])
        assert info['physical_steps']==20 and env.owner.data.time==pytest.approx(.020)
        assert np.isfinite(reward) and not term and not trunc
        np.testing.assert_array_equal(first,env.reset(seed=42)[0])
        with pytest.raises(ValueError):env.step(np.array([np.nan,0]))
    finally:env.close()


def test_no_live_qpos_writes():
    for module in ('src/rl/stewart_dynamic_attitude_env.py','src/workspace/dynamic_attitude_study.py'):
        body=(ROOT/module).read_text()
        assert 'data.qpos[' not in body or module.endswith('stewart_dynamic_attitude_env.py')
        assert 'data.qpos[:]=' not in body and 'data.qpos[:] =' not in body
