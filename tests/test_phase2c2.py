import copy,json
import numpy as np,pytest
from src.model import ROOT
from src.rl.moving_reference import MovingReference
from src.rl.phase2c2_env import MovingResidualEnv,moving_owner
from src.rl.stewart_residual_env import owner_config,env_config
from src.control.phase2c0_experiment import PhysicalOwner

TASK=dict(id='unit',tau_s=.005,Fz_N=-5.5,on_step=1600,off_step=3500,amplitude_m=.003)
def config():
    c=env_config();c.update(reference_cache=str(ROOT/'results/runs/phase2c2_20261030_v1/references'),baseline='four_d_lead',training_tasks=[TASK]);return c

def test_full_zero_equals_original_four_d_lead_through_motion_and_load(tmp_path):
    c=config();a=moving_owner(owner_config(tmp_path/'new',TASK),TASK,c)
    b=PhysicalOwner(dict(owner_config(tmp_path/'old',TASK),diagnostic=dict(kind='four_d_lead')));b.active=MovingReference(b.nominal,.003,c['reference_cache']);b.ref=b.active.at(0)
    try:
        for _ in range(2200):
            a.step();b.step();np.testing.assert_array_equal(a.data.qpos,b.data.qpos);np.testing.assert_array_equal(a.data.qvel,b.data.qvel);np.testing.assert_array_equal(a.data.ctrl,b.data.ctrl)
        assert not a.failure and not b.failure
    finally:a.recorder.close(dict(test='zero_equivalence'));b.recorder.close(dict(test='zero_equivalence'))

def test_reference_returns_home_dense_limits_and_velocity(tmp_path):
    o=moving_owner(owner_config(tmp_path/'owner',TASK),TASK,config());r=o.active
    try:
        for t in [0,6]:np.testing.assert_allclose(r.at(t).p,r.geometry.home_position,atol=1e-15);np.testing.assert_allclose(r.at(t).twist,0,atol=1e-15)
        assert r.at(2.5).p[2]-r.geometry.home_position[2]==pytest.approx(.003)
        for t in [.8,1.5,3.5,4.]:
            numeric=(r.at(t+1e-5).p-r.at(t-1e-5).p)/2e-5;np.testing.assert_allclose(numeric,r.at(t).twist[:3],atol=1e-9)
        with pytest.raises(ValueError):MovingReference(o.nominal,.005,config()['reference_cache'])
    finally:o.recorder.close(dict(test='reference'))

def test_env_actions_physics_observation_and_invalid_input(tmp_path):
    c=config();e=MovingResidualEnv(tmp_path/'env',config=c);obs,_=e.reset(seed=42)
    try:
        assert obs.shape==(255,) and e.observation_space.contains(obs)
        with pytest.raises(ValueError):e.step([np.nan])
        assert e.owner.n==0
        for a in [1.,-1.,0.]:
            obs,r,te,tr,info=e.step([a]);assert info['physical_steps']==50 and not te and not tr and np.isfinite(r)
            assert max(abs(e.owner.residual.trace['total_add']))<=5+1e-12
            assert abs(e.owner.residual.trace['requested']).max()<=1
        assert e.total_physical_steps==150 and e.owner.n==150
    finally:e.close()

def test_split_and_nominal_boundary():
    from src.rl.observations import MeasuredState
    p=json.loads((ROOT/'results/runs/phase2c2_20261030_v1/protocol_frozen.json').read_text(encoding="utf-8"));tasks=[p['train'],p['validation'],*p['final']]
    assert len({(x['amplitude_m'],x['shift_s'],x['tau_s'],x['on_step']) for x in tasks})==4
    assert not {'tau_s','external_wrench','F_drive','actual_gain'}&set(MeasuredState.__dataclass_fields__)
    assert p['original_z_envelope_m']==.005 and max(p['amplitude_candidates_m'])==.003

def test_future_episode_metadata_matches_canonical_directory(tmp_path):
    e=MovingResidualEnv(tmp_path/'env',config=config())
    try:
        a,_=e.reset(seed=42);assert e.owner.episode==0
        b,_=e.reset(seed=42);assert e.owner.episode==1;np.testing.assert_array_equal(a,b)
        e.step([0.]);assert e.owner.last_row['episode']==1
    finally:e.close()
