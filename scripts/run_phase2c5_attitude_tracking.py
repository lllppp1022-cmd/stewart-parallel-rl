"""Geometry first, then actual six-leg force-driven reference tracking."""
import _bootstrap
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.kinematics.stewart_geometry import StewartGeometry
from src.model import ROOT, sha, write_json
from src.phase1.plant import create_plant
from src.workspace.attitude_study import RUN, inspect_attitude, make_attitude_owner
from src.workspace.geometry import ExpandedGeometryQuery


def freeze():
    RUN.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for phase in ("phase2c3b", "phase2c4"):
        path = ROOT / f"results/{phase}_artifact_manifest.json"
        frozen.update({k: v["sha256"] for k, v in json.loads(path.read_text(encoding="utf-8"))["files"].items()})
        frozen[str(path.relative_to(ROOT))] = sha(path)
    old = ROOT / "results/runs/phase2c4_20261105_v1/history_freeze.json"
    frozen.update(json.loads(old.read_text(encoding="utf-8")))
    write_json(RUN / "history_freeze.json", frozen)
    write_json(RUN / "protocol.json", dict(
        physical_dt_s=.001, protection_unchanged=True,
        reference_components_deg=[.35, .35, .25], selection="Lowest mean SO3 RMSE among methods passing all three active trajectories",
        tracking_gate="All 6000 steps, no protection, each requested axis reaches +/-80% and final 0.5s <0.1deg/<0.75mm",
        tau_s=.005, recovery_deg=.1, recovery_dwell_s=.5))


def geometry():
    m = create_plant()
    g, q = StewartGeometry(m), ExpandedGeometryQuery()
    points = [(0., 0.)] + [(s*a, 0.) if axis == 0 else (0., s*a)
                            for a in [.1, .35, .5, 1., 2.] for s in [-1, 1] for axis in [0, 1]]
    points += [(i*a, j*a) for a in [.25, .5, 1., 2.] for i in [-1, 1] for j in [-1, 1]]
    rows = [inspect_attitude(m, g, q, z, r, p)[0] for z, group in
            [(0., points), (.05, [(0., 0.), (.5, 0.), (0., -.5), (.25, .25), (1., 1.)])]
            for r, p in group]
    pd.DataFrame(rows).to_csv(ROOT / "results/phase2c5_attitude_workspace.csv", index=False)
    print("Geometry", sum(r["geometric"] for r in rows), "/", len(rows), flush=True)


def trial(folder, task, method, torque=(0., 0.), on=2., off=3.5, record=False):
    owner, _ = make_attitude_owner(folder, task, method, torque, on, off, record)
    try:
        for _ in range(6000):
            owner.step()
            if owner.failure and owner.data.time >= owner.protect_end - 1e-10:
                break
    finally:
        owner.recorder.close(dict(steps=owner.n, failure=owner.failure, task=task, method=method))
    failure = owner.failure
    del owner
    gc.collect()
    return summarize(folder, task, method, torque, on, off, failure)


def summarize(folder, task, method, torque, on, off, failure):
    d = pd.read_csv(folder / "physics.csv.gz")
    tail = d.tail(500)
    angles = d[["rpy_0", "rpy_1", "rpy_2"]].to_numpy()
    ref = d[["rpy_ref_0", "rpy_ref_1", "rpy_ref_2"]].to_numpy()
    force = d[[f"force_{i}" for i in range(6)]].to_numpy()
    extra = d[[f"F_command_{i}" for i in range(6)]].to_numpy() - d[[f"F_C0_{i}" for i in range(6)]].to_numpy()
    active = d.failure.isna().to_numpy()
    peak_add = abs(extra[active]).max(initial=0.)
    slew = abs(np.diff(np.vstack([np.zeros((1, 6)), extra[active]]), axis=0)).max(initial=0.) * 1000
    direction = all(angles[:, i].max() >= .8 * ref[:, i].max() and
                    angles[:, i].min() <= .8 * ref[:, i].min()
                    for i in range(2) if abs(ref[:, i]).max() > .01)
    row = dict(folder=str(folder), task=task, method=method, mx_Nm=torque[0], my_Nm=torque[1],
               on_s=on, off_s=off, steps=len(d), failure=failure, follows_both_signs=direction,
               success=bool(len(d) == 6000 and not failure and direction
                            and tail.orientation_error_deg.max() < .1 and tail.position_error_mm.max() < .75
                            and peak_add <= 5+1e-8 and slew <= 20+1e-6 and abs(force).max() < 400
                            and not d.warning_count.max()),
               roll_min_deg=angles[:, 0].min(), roll_max_deg=angles[:, 0].max(),
               pitch_min_deg=angles[:, 1].min(), pitch_max_deg=angles[:, 1].max(),
               yaw_peak_deg=abs(angles[:, 2]).max(), so3_rmse_deg=np.sqrt(np.mean(d.orientation_error_deg**2)),
               so3_peak_deg=d.orientation_error_deg.max(), xyz_peak_mm=d.position_error_mm.max(),
               omega_peak_deg_s=np.rad2deg(np.linalg.norm(d[[f"twist_actual_{i}" for i in (3,4,5)]], axis=1).max()),
               force_peak_N=abs(force).max(), force_rate_N_s=abs(np.diff(force, axis=0)).max(initial=0.)*1000,
               force_energy_N2_s=np.sum(force**2)*.001, add_peak_N=peak_add, add_rate_N_s=slew,
               condition_peak=d.actual_jacobian_condition.max(), clearance_m=d.expanded_mesh_gap_m.min(),
               weld_translation_m=d.weld_translation_m.max(), weld_angle_rad=d.weld_angle_rad.max())
    expected = ((d.input_step >= round(on*1000)) & (d.input_step < round(off*1000))).to_numpy()[:, None] * np.array([0,0,0,*torque,0])
    row["external_max_error"] = abs(d[[f"external_world_{i}" for i in range(6)]].to_numpy()-expected).max()
    row["success"] = bool(row["success"] and row["external_max_error"] < 1e-9)
    for name, t in [("load", on), ("unload", off)]:
        mask = (d.time_s >= t) & (d.time_s < t+1)
        row[f"{name}_iae_deg_s"] = d.orientation_error_deg[mask].sum()*.001
        end = off if name == "load" else 6.
        w = d[(d.time_s >= t) & (d.time_s <= end)]
        good = (w.orientation_error_deg < .1).to_numpy()
        # Recovery requires a clean 0.5 s dwell extending to the window end.
        suffix = np.logical_and.accumulate(good[::-1])[::-1]
        ids = np.flatnonzero(suffix & (w.time_s.to_numpy() <= end-.5))
        recovered = bool(len(ids) and len(d) == 6000 and not failure)
        row[f"{name}_recovery_s"] = float(w.time_s.iloc[ids[0]]-t) if recovered else None
        row[f"{name}_right_censored"] = not recovered
    write_json(folder / "metrics.json", row)
    print(task, method, torque, row["success"], row["so3_peak_deg"], failure, flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "track", "collect"])
    parser.add_argument("--worker", type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.stage == "prepare":
        freeze()
        geometry()
    elif args.stage == "track":
        method = ["C0", "C4", "fourDlead"][args.worker]
        rows = [trial(RUN/"tracking"/method/task, task, method, record=(method=="C4")) for task in ["R1","R2","R3"]]
        write_json(RUN/f"tracking_{method}.json", rows)
    else:
        rows = sum([json.loads((RUN/f"tracking_{method}.json").read_text())
                    for method in ["C0","C4","fourDlead"]], [])
        d = pd.DataFrame(rows)
        d.to_csv(ROOT/"results/phase2c5_attitude_tracking.csv", index=False)
        candidates = {m: g.so3_rmse_deg.mean() for m, g in d.groupby("method") if g.success.all()}
        write_json(RUN/"tracking_gate.json", dict(passed=bool(candidates), scores=candidates,
                   selected=min(candidates, key=candidates.get) if candidates else None))


if __name__ == "__main__":
    main()
