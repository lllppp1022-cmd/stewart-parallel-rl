"""Run Phase 2C-6 paired physics trials without writing to Phase 2C-5."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

from phase2c6_protocol import OLD, ROOT, RUN, config, read, sha, write_json
from run_phase2c5_attitude_tracking import summarize
from src.phase1.plant import platform_state
from src.rl.attitude_reward_v2 import AttitudeRewardV2Env
from src.rl.ppo_trace import parameter_hash
from src.rl.stewart_attitude_env import SCALE, StewartAttitudeResidualEnv


def recovery(time, error, start, end, band, complete, dwell=.5):
    inside = (time >= start) & (time <= end + 1e-9)
    t, e = time[inside], error[inside]
    if not complete or not len(t) or t[-1] < end-.001001:
        return "right_censored", None
    outside = np.flatnonzero(e >= band)
    if not len(outside):
        return "not_triggered", None
    after = outside[-1] + 1
    if after < len(t) and end-t[after] >= dwell-1e-9:
        return "recovered", float(t[after]-start)
    return "right_censored", None


def trial(task, method, group, *, record=False, v2=True):
    folder = RUN/group/task["id"]/method
    if (folder/"comparison.json").exists():
        return read(folder/"comparison.json")
    env_class = AttitudeRewardV2Env if v2 else StewartAttitudeResidualEnv
    env = env_class(folder, [task], "fourDlead", [0] if record else [],
                    source="Phase 2C-6 deterministic evaluation" if method != "random" else
                    "Phase 2C-6 bounded random safety probe")
    obs, _ = env.reset(seed=6401)
    model = model_file = before = None
    if method.startswith(("V1_", "V2_", "C2_")):
        kind, seed = method.split("_")
        parent = "phase2c5" if kind == "V1" else "phase2c6/" + ("reward_v2" if kind == "V2" else "curriculum_v2")
        model_file = ROOT/"models"/parent/f"seed{seed}"/"final.zip"
        model = PPO.load(model_file, device="cpu")
        before = parameter_hash(model.policy)
    rng = np.random.default_rng(6401)
    gains = config("eval")["gains"]
    try:
        for _ in range(300):
            now = env.owner.data.time
            action = np.zeros(2)
            if model is not None:
                action = model.predict(obs, deterministic=True)[0]
                env.context["version"] = model.num_timesteps // 512
            elif method == "classical50":
                _, r, v = platform_state(env.owner.model, env.owner.data)
                ref = env.reference.at(now)
                err = Rotation.from_matrix(r@ref.r.T).as_rotvec()
                action = (-gains["kp"]*err[:2]-gains["kd"]*(v[3:5]-ref.twist[3:5]))/SCALE
            elif method == "random":
                action = rng.uniform(-1, 1, 2)
            elif method in ("xp", "xm", "yp", "ym") and task["on"]-1e-9 <= now < task["off"]-1e-9:
                action["xy".index(method[0])] = 1 if method[1] == "p" else -1
            env.context["raw"] = action
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    finally:
        env.close()
    ep = env.episodes[0]
    folder_episode = Path(ep["folder"])
    metrics = summarize(folder_episode, "Home", method, task["torque"], task["on"], task["off"], ep["failure"])
    d = pd.read_csv(folder_episode/"physics.csv.gz")
    t, err = d.time_s.to_numpy(), d.orientation_error_deg.to_numpy()
    for stage, start, end in (("load", task["on"], task["off"]), ("unload", task["off"], 6.)):
        for label, band in (("01", .1), ("001", .01)):
            status, elapsed = recovery(t, err, start, end, band, ep["complete"])
            metrics[f"{stage}_band_{label}_status"] = status
            metrics[f"{stage}_band_{label}_s"] = elapsed
    direction = np.array(task["torque"])
    directional_error = d[["rpy_0", "rpy_1"]].to_numpy() @ (direction/np.linalg.norm(direction))
    metrics["unload_reverse_overshoot_deg"] = max(0., -directional_error[t >= task["off"]].min(initial=0.))
    metrics.update(task_id=task["id"], group=group, model_sha256=sha(model_file) if model_file else None,
                   observation_clip_fraction=ep["clip_fraction"], policy_unchanged=model is None or before == parameter_hash(model.policy),
                   accepted_rms_Nm=float(np.sqrt(np.mean(d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()**2))),
                   requested_rms_Nm=float(np.sqrt(np.mean(d[["requested_mx_Nm","requested_my_Nm"]].to_numpy()**2))),
                   projection_fraction=float(d.projected.mean()))
    write_json(folder/"comparison.json", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["preflight","probe","dev","final"])
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--curriculum-seeds", type=int, nargs="*", default=[])
    args = p.parse_args()
    torch.set_num_threads(1)
    c = config("eval")
    group = args.stage
    if group == "probe":
        task = config("ppo")["environment"]["train"][-1]
        method = ["baseline", "xp", "xm", "yp", "ym", "baseline_v1"][args.index]
        trial(task, method, group, v2=method != "baseline_v1")
        return
    if group == "preflight":
        task = c["final"][args.index]
        methods = ["baseline"]
    elif group == "dev":
        task = c["development"][args.index]
        methods = [f"V2_{seed}" for seed in args.seeds]
    else:
        selection=read(RUN/"final_selection.json")
        assert selection["eval_config_sha"]==sha(ROOT/"configs/phase2c6_eval.yaml")
        if args.seeds!=selection["fixed"] or args.curriculum_seeds!=selection["curriculum"]:
            p.error("Supply all seeds from the frozen final selection")
        task = (c["final"]+[c["regression"]])[args.index]
        methods = ["baseline","classical50","V1_42","V1_123","V1_2026"]
        methods += [f"V2_{seed}" for seed in args.seeds]+[f"C2_{seed}" for seed in args.curriculum_seeds]
    for method in methods:
        record = ((task["id"]=="E_mix" and group=="dev" and method=="V2_42")
                  or (args.stage=="final" and args.index in (2,3) and method in ("baseline","classical50","V2_42","C2_42")))
        trial(task, method, group, record=record)


if __name__ == "__main__":
    main()
