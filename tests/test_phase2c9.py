import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_phase2c9_dynamics import REFS
from src.rl.phase2c8_env import Phase2C8Env
from src.rl.stewart_adaptive_env import HistoryObservation, StewartAdaptiveEnv


def task(kind, value):
    return dict(id="check", stage="validation", reference=REFS[0][1],
                torque=[.6, -.8], on=.08, off=.16, plant=dict(kind=kind, value=value))


def trace(env):
    rows = []
    for _ in range(12):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(2))
        owner = env.unwrapped.owner
        rows.append(np.r_[obs[-58:], reward, owner.data.qpos, owner.data.qvel, owner.data.ctrl])
        assert not terminated and not truncated
    return rows


def test_nominal_matches_frozen_environment(tmp_path):
    args = dict(tasks=[task("nominal", 1)], seed=42, version="v2_control")
    old = Phase2C8Env(tmp_path / "old", **args)
    new = StewartAdaptiveEnv(tmp_path / "new", **args)
    try:
        np.testing.assert_array_equal(old.reset()[0], new.reset()[0])
        np.testing.assert_array_equal(trace(old), trace(new))
    finally:
        old.close()
        new.close()


def test_hidden_mass_and_history_do_not_change_control(tmp_path):
    args = dict(tasks=[task("mass", 1.10)], seed=42, version="v2_control")
    current = StewartAdaptiveEnv(tmp_path / "current", **args)
    history = HistoryObservation(StewartAdaptiveEnv(tmp_path / "history", **args), 10)
    try:
        initial = current.reset()[0]
        np.testing.assert_array_equal(history.reset()[0], np.tile(initial, 10))
        owner = current.owner
        body = owner.model.body("upper").id
        assert owner.model.body_mass[body] == 22
        assert owner.nominal.body_mass[body] == 20
        np.testing.assert_allclose(owner.model.body_inertia[body], owner.nominal.body_inertia[body] * 1.10)
        np.testing.assert_array_equal(trace(current), trace(history))
        np.testing.assert_array_equal(history.reset()[0], np.tile(initial, 10))
        assert history.unwrapped.owner.n == 0
    finally:
        current.close()
        history.close()
