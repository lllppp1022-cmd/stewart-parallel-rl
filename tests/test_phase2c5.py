import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.model import ROOT, sha
from src.rl.stewart_attitude_env import (
    DIM, GAMMA, MomentPort, StewartAttitudeResidualEnv, components, failure_penalty,
)
from src.workspace.attitude_study import RUN, AttitudeReference


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("task",["R1","R2","R3"])
def test_analytic_spatial_reference(task):
    ref=AttitudeReference(task)
    for time in (.5,1.75,3.,4.25,5.5):
        np.testing.assert_array_equal(ref.at(time).twist,np.zeros(6))
        np.testing.assert_array_equal(ref.angular_acceleration(time),np.zeros(3))
    t=1.2
    h=1e-5
    r0=ref.at(t-h).r
    r1=ref.at(t+h).r
    np.testing.assert_allclose(Rotation.from_matrix(r1@r0.T).as_rotvec()/(2*h),
                               ref.at(t).twist[3:],atol=1e-10)
    assert abs(ref.axis).max() <= np.deg2rad(.35)


def test_offline_and_dynamic_gates():
    d=pd.read_csv(ROOT/"results/phase2c5_attitude_workspace.csv")
    assert len(d)==42 and d.geometric.all()
    assert d.passive_unlimited.eq(30).all() and d.clearance_m.min()>=.005 and d.condition.max()<50
    t=pd.read_csv(ROOT/"results/phase2c5_attitude_tracking.csv")
    assert len(t)==9 and t.success.all() and t.follows_both_signs.all()
    assert t.weld_translation_m.max()<1e-4 and t.weld_angle_rad.max()<1e-4
    w=load(ROOT/"configs/phase2c5_attitude_workspace.yaml")
    old=load(ROOT/"configs/phase2c3a_large_workspace.json")
    assert w["roll_pitch_guard_deg"]==old["actual_roll_pitch_limit_deg"] and w["no_wider_position_or_attitude_guard"]


@pytest.mark.parametrize("direction",[0,1,2,3])
def test_measured_boundary_and_retained_failures(direction):
    d=load(RUN/f"strong_{direction}.json")
    assert all(r["external_max_error"]<1e-9 for r in d)
    assert next(r for r in d if r["amplitude_Nm"]==2.4)["recoverable"]
    fail=next(r for r in d if r["amplitude_Nm"]==4.8)
    assert not fail["recoverable"] and fail["failure"]=="restricted_position_envelope"
    assert fail["load_right_censored"] and fail["unload_right_censored"]


def test_same_state_force_mapping_and_shared_slew():
    from types import SimpleNamespace
    from src.control.residual_governor import SharedBudgetGovernor
    core=SimpleNamespace(jacobian=np.diag([1.,1.,1.,.2,.25,1.]))
    original=SharedBudgetGovernor()
    port=MomentPort(original,core)
    port.request=np.array([.05,-.05])
    c0=np.full(6,90.)
    command,_=port.apply(c0,c0,np.zeros(6))
    np.testing.assert_allclose(core.jacobian.T@port.leg,[0,0,0,.05,-.05,0],atol=1e-14)
    assert abs(command-c0).max()<=.02+1e-12
    assert port.original is original


def test_observation_contract_and_reward():
    text=inspect.getsource(StewartAttitudeResidualEnv.observe)
    for term in ("self.task","o.profile","o.lag","actuator_force","external_world","F_drive"):
        assert term not in text
    assert DIM==52
    zero=components(0.,np.zeros(3),0.,np.zeros(2),np.zeros(2))
    assert np.all(zero==0)
    assert components(.01,np.zeros(3),0.,np.zeros(2),np.zeros(2))[0]>0
    for remaining in (0,1,300):
        assert failure_penalty(remaining)<-sum(100*GAMMA**k for k in range(remaining))
    gate=load(RUN/"training_gate.json")
    assert gate["passed"] and gate["zero"]["gym_check"] and gate["zero"]["reset_checks"]==2
    assert len(gate["random"])==8 and all(x["reward_error"]<1e-8 for x in gate["random"])


def test_zero_full_rgb_and_qt_equivalence():
    a=np.load(RUN/"validation/zero/episode_0000/integration_states.npz")["states"]
    b=np.load(RUN/"validation/frozen/integration_states.npz")["states"]
    np.testing.assert_array_equal(a,b)
    assert len(a)==6000 and load(RUN/"qt_equivalence.json")["bitwise_steps"]==2000


def test_protection_and_time_semantics(tmp_path):
    cfg=load(ROOT/"configs/phase2c5_attitude_env.yaml")
    e=StewartAttitudeResidualEnv(tmp_path/"fault",[cfg["train"][0]],cfg["method"],source="Fault injection test")
    try:
        obs,_=e.reset(seed=42)
        assert obs.shape==(52,)
        _,_,term,trunc,info=e.step([0,0])
        assert not term and not trunc and info["physical_steps"]==20
        e.owner.hold("test_injected_protection",failed=True)
        _,reward,term,trunc,info=e.step([0,0])
        assert term and not trunc and reward<0 and info["physical_steps"]>=199
    finally:
        e.close()


@pytest.mark.parametrize("seed",[42,123,2026])
def test_pilot_updates_saved_and_real(seed):
    folder=RUN/"training"/f"seed{seed}"
    summary=load(folder/"summary.json")
    assert summary["decisions"]==8192 and summary["physical_steps"]==163840
    assert summary["updates"]==16 and summary["initial_hash"]!=summary["final_hash"]
    assert sha(ROOT/"models/phase2c5"/f"seed{seed}"/"final.zip")==summary["model_sha"]
    rows=load(folder/"updates.json")
    assert all(r["actor_delta"]>0 and r["critic_delta"]>0 for r in rows)
    for k,path in enumerate(sorted(folder.glob("rollout_*.npz"))):
        with np.load(path) as z:
            assert z["observations"].shape[-1]==52 and z["actions"].shape[-1]==2
            assert np.isfinite(z["rewards"]).all()
    assert k==15


def test_complete_independent_evaluation_and_videos():
    d=pd.read_csv(ROOT/"results/phase2c5_classical_vs_ppo.csv")
    assert len(d)==24 and d.groupby("task_id").size().eq(6).all()
    assert d[d.method.str.startswith("PPO")].deterministic.all()
    assert not d[d.method=="initial_random"].deterministic.any()
    for kind in ("attitude_motion","ppo_training","ppo_evaluation"):
        r=load(RUN/f"video_{kind}.json")
        assert r["passed"] and all(v["state_and_time_verified"] for v in r["proof"])
    historical=load(RUN/"history_audit.json")
    assert historical["only_transient_missing"] and not historical["changed"]
