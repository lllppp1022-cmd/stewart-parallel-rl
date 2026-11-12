"""Phase 2C-9 plant-only perturbations on the frozen dynamic environment."""
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from ..dynamics.actuator_lag import ActuatorLag
from ..model import write_json
from ..phase1.plant import plant_metadata
from .phase2c8_env import Phase2C8Env


class StewartAdaptiveEnv(Phase2C8Env):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        owner = self.owner
        model = owner.model
        assert model is not owner.nominal and owner.n == 0 and owner.data.time == 0
        body = model.body("upper").id
        spec = self.task["plant"]
        kind, value = spec["kind"], spec["value"]
        if kind == "mass":
            # Fixed-shape density scaling preserves COM and principal axes.
            model.body_mass[body] *= value
            model.body_inertia[body] *= value
        elif kind == "com_x":
            # Rigid translation of the inertial distribution; inertia about COM stays fixed.
            model.body_ipos[body, 0] += value
        elif kind == "lag":
            initial = owner.home.at(0).ff
            for name in ("lag", "shadow_base", "shadow_c0"):
                setattr(owner, name, ActuatorLag(value, initial))
        elif kind != "nominal":
            raise ValueError(kind)
        if kind in ("mass", "com_x"):
            mujoco.mj_setConst(model, mujoco.MjData(model))
            mujoco.mj_forward(model, owner.data)
        write_json(owner.recorder.folder / "phase2c9_plant.json", dict(
            specification=spec, actual=plant_metadata(model, kind),
            nominal=plant_metadata(owner.nominal, "nominal"),
            actual_drive_tau_s=owner.lag.tau.tolist()))
        return obs, info


class HistoryObservation(gym.ObservationWrapper):
    """Newest frame last; reset padding is the initial causal observation."""

    def __init__(self, env, frames):
        super().__init__(env)
        if frames not in (5, 10):
            raise ValueError("Expected 5 or 10 frames")
        self.buffer = deque(maxlen=frames)
        self.observation_space = gym.spaces.Box(-10, 10, (58 * frames,), np.float32)

    def reset(self, **kwargs):
        self.buffer.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        if not self.buffer:
            self.buffer.extend(observation.copy() for _ in range(self.buffer.maxlen - 1))
        self.buffer.append(observation.copy())
        return np.concatenate(self.buffer)
