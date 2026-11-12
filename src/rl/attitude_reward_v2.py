"""Phase 2C-6 reward wrapper. Physics, observations and decisions use Phase 2C-5."""
import numpy as np

from .stewart_attitude_env import CAP, SCALE, StewartAttitudeResidualEnv

NAMES = ("orientation", "omega", "xyz", "action", "slew")
SCALES = (.006, .003, .2, SCALE, 2.5)
WEIGHTS = (1., .2, .1, .001, .000025)


def v2_components(rotation_deg, omega_rad_s, xyz_mm, accepted_nm, slew_nm_s):
    magnitudes = np.array([
        rotation_deg**2, np.sum(omega_rad_s**2), xyz_mm**2,
        np.sum(accepted_nm**2), np.sum(slew_nm_s**2),
    ])
    parts = magnitudes * np.asarray(WEIGHTS) / np.square(SCALES)
    return parts * min(1., CAP / max(parts.sum(), 1e-30))


class AttitudeRewardV2Env(StewartAttitudeResidualEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished_diagnostics = []

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        append = self.owner.recorder.append
        self.physical_diagnostics = dict(peak_so3_error=0., load_window_IAE=0., unload_window_IAE=0.)

        def log(row, state):
            accepted = self.port.accepted if not self.owner.failure else np.zeros(2)
            slew = (accepted - self.previous_accepted) * 1000
            omega = np.array([row[f"twist_actual_{i}"] - row[f"twist_ref_{i}"] for i in (3, 4, 5)])
            parts = v2_components(row["orientation_error_deg"], omega, row["position_error_mm"], accepted, slew)
            for name, value in zip(NAMES, parts):
                row[f"v2_cost_{name}"] = value
            append(row, state)
            self.cost = parts
            angle = row["orientation_error_deg"]
            self.physical_diagnostics["peak_so3_error"] = max(self.physical_diagnostics["peak_so3_error"], angle)
            for name in ("load", "unload"):
                t = self.task["on" if name == "load" else "off"]
                if t <= row["time_s"] < t + 1:
                    self.physical_diagnostics[name+"_window_IAE"] += angle*.001
            self.last_physical_diagnostic_row = row

        self.owner.recorder.append = log
        return result

    def finish(self, reason):
        if self.owner is not None:
            self.finished_diagnostics.append(self.physical_diagnostics.copy())
        super().finish(reason)
