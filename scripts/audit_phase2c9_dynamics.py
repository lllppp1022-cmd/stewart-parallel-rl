"""Bounded paired preflight. Checkpoint training is outside this entry point."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phase2c8_protocol import ROOT, cfg, read, write_json
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv
from validate_phase2c7_dynamic_tasks import metrics

RUN = ROOT / "results/runs/phase2c9_20261109_v1"
PLANTS = [("nominal", 1), ("mass", 1.03), ("mass", 1.10),
          ("com_x", .0005), ("com_x", .0015), ("lag", .002), ("lag", .006)]
REFS = [("home", cfg("training")["curriculum"]["A"][0]["reference"]),
        ("combined", cfg("evaluation")["final"][-1]["reference"])]


def trial(index, record=False, adaptive=False):
    ref_id, ref = REFS[index // len(PLANTS)]
    kind, value = PLANTS[index % len(PLANTS)]
    spec = dict(kind=kind, value=value)
    task = dict(id=ref_id, reference=ref, torque=[.6, -.8], on=4.4, off=6.2,
                stage="preflight", plant=spec)
    group = "classical_diagnostic" if adaptive else "video_capture" if record else "preflight_v2"
    folder = RUN / group / f"{index:02d}"
    if folder.exists():
        raise FileExistsError(folder)
    env = StewartAdaptiveEnv(folder, [task], seed=6401, version="v2_control",
                             record_episodes=(0,) if record else (),
                             source=f"PHASE2C9 {group.upper()}")
    frames = []
    try:
        obs, _ = env.reset(seed=6401)
        frames.append(obs)
        integral = np.zeros(2)
        for _ in range(800):
            action = np.zeros(2)
            if adaptive:
                error = obs[:2] * .001
                velocity_error = (obs[3:5] - obs[6:8]) * .02
                pd = -25 * error - velocity_error
                proposed = np.clip(np.exp(-.02) * integral - 10 * .02 * error, -.025, .025)
                integral = np.where(np.abs(pd + proposed) <= .05, proposed, np.exp(-.02) * integral)
                action = np.clip((pd + integral) / .05, -1, 1)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(obs)
            if terminated or truncated:
                break
    finally:
        env.close()
    episode = env.episodes[0]
    result = metrics(Path(episode["folder"]), episode)
    result.update(index=index, reference_id=ref_id, kind=kind, value=value,
                  method="classical_leaky_pi_diagnostic" if adaptive else "frozen_zero_residual")
    write_json(folder / "metrics.json", result)
    np.save(folder / "observations.npy", np.stack(frames))
    print(index, ref_id, spec, result["success"], result["failure"], flush=True)


def collect():
    paths = sorted((RUN / "preflight_v2").glob("*/metrics.json"))
    data = pd.DataFrame([read(p) for p in paths])
    assert len(data) == len(REFS) * len(PLANTS)
    data.to_csv(ROOT / "results/phase2c9_parameter_preflight.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=int, nargs="?")
    parser.add_argument("--record-index", type=int, choices=range(len(PLANTS) * len(REFS)))
    parser.add_argument("--classical-index", type=int, choices=range(len(PLANTS) * len(REFS)))
    args = parser.parse_args()
    if args.record_index is not None:
        trial(args.record_index, record=True)
    elif args.classical_index is not None:
        trial(args.classical_index, adaptive=True)
    elif args.worker is None:
        collect()
    else:
        for index in range(args.worker, len(REFS) * len(PLANTS), 4):
            trial(index)
