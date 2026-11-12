"""Causal 50 Hz residual moment control over the frozen 1 kHz owner."""
import csv
import gc
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ..control.residual_governor import SharedBudgetGovernor
from ..model import write_json
from ..phase1.plant import platform_state
from ..workspace.attitude_study import make_attitude_owner

STRIDE = 20
HORIZON = 6000
GAMMA = .99**.4
CAP = 100.
SCALE = .05
FIELDS = ("SO3_rad[3]", "angular_velocity[3]", "reference_angular_velocity[3]",
          "position_error[3]", "linear_velocity[3]", "relative_slides[6]",
          "slide_velocity[6]", "known_command[6]", "observer[6]", "base_add[6]",
          "previous_request[2]", "previous_accepted[2]", "reference_rotation[3]")
DIM = 52


class MomentPort:
    """Extend the existing governor input, with a same-state audit counterfactual."""
    def __init__(self, original, controller):
        self.original = original
        self.controller = controller
        self.counter = SharedBudgetGovernor()
        self.reset()

    def reset(self):
        self.original.reset()
        self.request = np.zeros(2)
        self.accepted = np.zeros(2)
        self.leg = np.zeros(6)
        self.base_add = np.zeros(6)
        self.projected = False

    def apply(self, c0, base, residual):
        j = self.controller.jacobian
        self.counter.previous = self.original.previous.copy()
        self.leg = np.linalg.solve(j.T, np.r_[np.zeros(3), self.request, 0.])
        if np.any(self.request):
            baseline, _ = self.counter.apply(c0, base, residual)
            command, info = self.original.apply(c0, base, residual + self.leg)
        else:
            command, info = self.original.apply(c0, base, residual)
            baseline = command
        self.base_add = baseline - c0
        self.accepted = (j.T @ (command - baseline))[3:5]
        self.projected = info["reason"] == "projected_shared_budget"
        return command, info


def components(rotation_deg, omega_error, xyz_mm, accepted, slew):
    parts = np.array([(rotation_deg/.05)**2, .2*np.sum((omega_error/.02)**2),
                      .1*(xyz_mm/.2)**2, .01*np.sum((accepted/SCALE)**2),
                      .005*np.sum((slew/2.5)**2)])
    return parts * min(1., CAP/max(parts.sum(), 1e-30))


def failure_penalty(remaining):
    return -(100 + CAP*(1-GAMMA**(remaining+10))/(1-GAMMA))


class StewartAttitudeResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, root, tasks, method, record_episodes=(), source="ACTUAL STOCHASTIC ATTITUDE PPO TRAINING"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.tasks, self.method = tasks, method
        self.source = source
        self.record_episodes = set(record_episodes)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (DIM,), np.float32)
        self.owner = None
        self.episode = -1
        self.total_physical_steps = 0
        self.global_decisions = 0
        self.records, self.episodes = [], []
        self.context = {}
        self.stream = (self.root/"training_samples.csv").open("w", newline="", encoding="utf-8")
        self.writer = None

    def observe(self):
        o = self.owner
        p, r, v = platform_state(o.model, o.data)
        ref = self.reference.at(o.data.time)
        estimate = getattr(o.residual.base, "diagnostics", {}).get("equivalent_disturbance", np.zeros(6))
        frame = np.r_[Rotation.from_matrix(r @ ref.r.T).as_rotvec()/.001,
                      v[3:]/.02, ref.twist[3:]/.02, (p-ref.p)/.0002, v[:3]/.01,
                      (o.data.qpos[o.qadr]-self.reference.at(0).q)/.01, o.data.qvel[o.dof]/.02,
                      (o.last_command-o.home.at(0).ff)/5, np.asarray(estimate)/[20,20,20,.8,.8,.8],
                      self.port.base_add/5, self.port.request/SCALE, self.port.accepted/SCALE,
                      Rotation.from_matrix(ref.r @ o.geometry.home_rotation.T).as_rotvec()/.01]
        if frame.shape != (DIM,) or not np.isfinite(frame).all():
            raise ValueError(f"Invalid causal observation {frame.shape}")
        self.clip_count += int((abs(frame)>10).sum())
        self.obs_count += DIM
        return np.clip(frame, -10, 10).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finish("reset_incomplete")
        self.episode += 1
        self.task = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        self.owner, self.reference = make_attitude_owner(
            self.root/f"episode_{self.episode:04d}", self.task.get("reference", "Home"), self.method,
            self.task["torque"], self.task["on"], self.task["off"], self.episode in self.record_episodes)
        self.owner.episode = self.episode
        self.port = MomentPort(self.owner.residual.governor, self.owner.residual.base)
        self.owner.residual.governor = self.port
        self.decision = self.clip_count = self.obs_count = 0
        self.episode_return = 0.
        self.previous_accepted = np.zeros(2)
        self.context = {}
        self.done = False
        self.last_row = None
        append = self.owner.recorder.append

        def log(row, state):
            accepted = self.port.accepted if not self.owner.failure else np.zeros(2)
            slew = (accepted-self.previous_accepted)*1000
            self.previous_accepted = accepted.copy()
            omega = np.array([row[f"twist_actual_{j}"]-row[f"twist_ref_{j}"] for j in (3,4,5)])
            self.cost = components(row["orientation_error_deg"], omega, row["position_error_mm"], accepted, slew)
            for j, axis in enumerate(("mx","my")):
                row[f"requested_{axis}_Nm"] = self.port.request[j]
                row[f"accepted_{axis}_Nm"] = accepted[j]
                row[f"slew_{axis}_Nm_s"] = slew[j]
            for j in range(6):
                row[f"rl_mapped_leg_{j}_N"] = self.port.leg[j]
                row[f"same_state_base_add_{j}_N"] = self.port.base_add[j]
            for name, value in zip(("orientation","omega","xyz","action","slew"), self.cost):
                row[f"cost_{name}"] = value
            row.update(decision=self.decision, global_decision=self.global_decisions,
                       policy_version=self.context.get("version", -1), rl_source=self.source,
                       projected=self.port.projected)
            append(row, state)
            self.last_row = row

        self.owner.recorder.append = log
        self.obs = self.observe()
        return self.obs.copy(), {"task_id": self.task["id"]}

    def step(self, action):
        if self.owner is None or self.done:
            raise RuntimeError("Reset before step")
        a = np.asarray(action, float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Expected two finite actions")
        clipped = np.clip(a, -1, 1)
        self.port.request = SCALE*clipped
        self.decision += 1
        self.global_decisions += 1
        start = self.owner.n
        total = np.zeros(5)
        accepted = []
        projected = []
        for _ in range(min(STRIDE, HORIZON-start)):
            self.owner.step()
            total += self.cost
            accepted.append(self.port.accepted.copy())
            projected.append(self.port.projected)
            if self.owner.failure:
                break
        failure = bool(self.owner.failure)
        while failure and self.owner.data.time < self.owner.protect_end-1e-10:
            self.owner.step()
            total += self.cost
        n = self.owner.n-start
        self.total_physical_steps += n
        terminal_penalty = failure_penalty(max(0, 300-self.decision)) if failure else 0.
        reward = float(-total.sum()/STRIDE + terminal_penalty)
        self.done = failure or self.owner.n >= HORIZON
        self.episode_return += reward
        self.obs = self.observe()
        raw = self.context.get("raw", a)
        row = dict(episode=self.episode, decision=self.decision, global_decision=self.global_decisions,
                   physical_steps=n, reward=reward, terminal_penalty=terminal_penalty,
                   raw_roll=raw[0], raw_pitch=raw[1], clipped_roll=clipped[0], clipped_pitch=clipped[1],
                   accepted_roll=float(np.mean(accepted, axis=0)[0]), accepted_pitch=float(np.mean(accepted, axis=0)[1]),
                   projection_fraction=float(np.mean(projected)), policy_version=self.context.get("version", -1),
                   terminated=failure, truncated=self.done and not failure, task_id=self.task["id"],
                   so3_deg=self.last_row["orientation_error_deg"], xyz_mm=self.last_row["position_error_mm"],
                   episode_return=self.episode_return,
                   action_box_clipped=bool(np.any(abs(raw)>1)))
        for name,value in zip(("orientation","omega","xyz","action","slew"),total):
            row[f"reward_{name}"] = -value/STRIDE
        self.records.append(row)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        info = {"failure": self.owner.failure, "physical_steps": n}
        if self.done:
            info["episode"] = {"r": self.episode_return, "l": self.decision}
            self.finish("protected_failure" if failure else "time_limit")
        return self.obs.copy(), reward, failure, self.done and not failure, info

    def finish(self, reason):
        if self.owner is None:
            return
        owner = self.owner
        row = dict(episode=self.episode, task=self.task, reason=reason, physical_steps=owner.n,
                   complete=owner.n==HORIZON and not owner.failure, failure=owner.failure,
                   folder=str(owner.recorder.folder), episode_return=self.episode_return,
                   clip_fraction=self.clip_count/max(1, self.obs_count), source=self.source)
        owner.recorder.close(row)
        self.episodes.append(row)
        write_json(self.root/"episodes.json", self.episodes)
        self.owner = None
        del owner
        gc.collect()

    def close(self):
        self.finish("budget_or_viewer_cutoff")
        self.stream.close()
