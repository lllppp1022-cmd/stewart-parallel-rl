"""Phase 2C-5 references and trials on the frozen large-workspace owner."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from ..audit_control import scratch_reference_data, static_balance
from ..disturbance.disturbance_profiles import DisturbanceProfile
from ..disturbance.recording import install_colored_capture
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian, singularity_diagnostics
from ..kinematics.stewart_geometry import StewartGeometry
from ..model import ROOT, write_json
from ..phase1.plant import create_plant
from ..phase1.trajectory import Reference
from ..rl.stewart_residual_env import owner_config
from .execution import LargeOwner
from .geometry import ExpandedGeometryQuery, check_native_joint_limits

RUN = ROOT / "results/runs/phase2c5_20261106_v1"
ANGLES = {"Home": (0, 0), "R1": (.35, 0), "R2": (0, .35), "R3": (.25, .25), "Hold": (.2, -.2)}


def inspect_attitude(model, geometry, query, z, roll, pitch):
    p = geometry.home_position + [0, 0, z]
    r = Rotation.from_rotvec(np.deg2rad([roll, pitch, 0])).as_matrix() @ geometry.home_rotation
    sol = inverse_kinematics(geometry, p, r)
    row = dict(z_m=z, roll_component_deg=roll, pitch_component_deg=pitch, ik=sol.success,
               closure_m=sol.max_position_error, closure_rad=sol.max_rotation_error,
               slide_margin_m=sol.minimum_stroke_margin, geometric=False)
    if sol.success:
        data = scratch_reference_data(model, geometry, sol, p, r)
        joints = check_native_joint_limits(model, data.qpos)
        gap = query.evaluate(data.qpos)
        j, _ = actuator_jacobian(geometry, p, r, sol)
        condition = singularity_diagnostics(j)["normalized_condition"]
        row.update(native_margin=joints["minimum_native_margin"], passive_unlimited=len(joints["unlimited_hinges"]),
                   clearance_m=gap["minimum_m"], nearest_pair=str(gap["nearest_pair"]), condition=condition,
                   geometric=bool(joints["valid"] and condition is not None and condition <= 50
                                  and gap["minimum_m"] >= .005 and sol.max_position_error < 1e-4
                                  and sol.max_rotation_error < 1e-4))
        for leg in range(6):
            for axis in range(6):
                row[f"chain_{leg}_{axis}"] = sol.chain_q[leg, axis]
        for axis in (3, 4):
            unit = np.eye(6)[axis]
            row[f"pure_axis_{axis}_static_add_Nm"] = 5 / abs(np.linalg.solve(j.T, unit)).max()
    return row, sol, p, r


@lru_cache(maxsize=8)
def reference_path(task):
    model = create_plant()
    geometry = StewartGeometry(model)
    query = ExpandedGeometryQuery()
    axis = np.asarray(ANGLES[task])
    grid = np.linspace(-1, 1, 41) if task != "Home" else np.array([-1., 1.])
    strokes, forces = [], []
    for s in grid:
        audit, sol, p, r = inspect_attitude(model, geometry, query, 0., *(s * axis))
        if not audit["geometric"]:
            raise ValueError(f"Reference geometry rejected: {audit}")
        strokes.append(sol.strokes)
        forces.append(static_balance(model, geometry, sol, p, r)[0])
    return geometry, np.deg2rad(np.r_[axis, 0]), CubicSpline(grid, strokes), CubicSpline(grid, forces)


class AttitudeReference:
    duration = 6.

    def __init__(self, task):
        self.name = task
        self.geometry, self.axis, self.strokes, self.forces = reference_path(task)

    def scalar(self, time):
        if self.name == "Hold":
            x = np.clip(time - .5, 0., 1.)
            return 10*x**3-15*x**4+6*x**5, 30*x*x*(1-x)**2, 60*x-180*x*x+120*x**3
        if self.name == "Home" or time < .5 or time >= 5.5:
            return 0., 0., 0.
        segment = min(int((time - .5) / 1.25), 3)
        x = (time - .5) / 1.25 - segment
        knots = [0., 1., 0., -1., 0.]
        delta = knots[segment + 1] - knots[segment]
        return (knots[segment] + delta * (10*x**3 - 15*x**4 + 6*x**5),
                delta * 30*x*x*(1-x)**2 / 1.25,
                delta * (60*x - 180*x*x + 120*x**3) / 1.25**2)

    def angular_acceleration(self, time):
        return self.axis * self.scalar(time)[2]

    def at(self, time):
        s, ds, _ = self.scalar(time)
        r = Rotation.from_rotvec(self.axis * s).as_matrix() @ self.geometry.home_rotation
        return Reference(self.strokes(s), self.strokes(s, 1)*ds, self.geometry.home_position.copy(),
                         r, np.r_[np.zeros(3), self.axis * ds], self.forces(s))


def make_attitude_owner(folder, task="Home", method="C4", torque=(0., 0.),
                        on=2., off=3.5, record=False):
    reference = AttitudeReference(task)
    # LargeOwner's Z telemetry hook calls scalar()[2]. The attitude reference
    # remains physically fixed at Home Z, so expose a separate Z-only hook.
    class OwnerReference(AttitudeReference):
        def scalar(self, time):
            return 0., 0., 0.

        def at(self, time):
            return reference.at(time)

    workspace = json.loads((ROOT / "configs/phase2c3a_large_workspace.json").read_text())
    config = owner_config(folder, dict(id=task, tau_s=.005, Fz_N=0.,
                                      on_step=round(on*1000), off_step=round(off*1000)), record)
    config["disturbance"]["wrench"] = [0., 0., 0., *torque, 0.]
    config.update(stage="phase2c5", max_sim_seconds=6., controller="C0" if method == "C0" else "C3")
    if method == "C0":
        config["yaw_policy"] = {"kind": "zero4"}
    if method == "fourDlead":
        config["diagnostic"] = {"kind": "four_d_lead"}
    owner = LargeOwner(config, OwnerReference(task), ExpandedGeometryQuery(), workspace)
    owner.config["trajectory"] = task
    write_json(Path(folder) / "config.json", owner.config)
    if record:
        install_colored_capture(owner.recorder, owner.model)
    return owner, reference
