"""Independent frozen combinations. Each worker owns its MuJoCo state."""
import _bootstrap
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

from run_phase2c5_attitude_tracking import summarize
from src.model import ROOT, sha, write_json
from src.phase1.plant import platform_state
from src.rl.stewart_attitude_env import SCALE, StewartAttitudeResidualEnv
from src.workspace.attitude_study import RUN

METHODS = ["baseline","classical50","initial_random","PPO42","PPO123","PPO2026"]


def run(task, method, base, gains):
    folder=RUN/"evaluation"/task["id"]/method
    record=task["id"]=="E_mix" and method in ("baseline","classical50","PPO42")
    env=StewartAttitudeResidualEnv(folder,[task],base,[0] if record else [],
                                    source="Independent attitude evaluation")
    obs,_=env.reset(seed=6401)
    model=None
    model_sha=None
    if method.startswith("PPO") or method=="initial_random":
        seed=42 if method=="initial_random" else int(method[3:])
        file=ROOT/"models/phase2c5"/f"seed{seed}"/("initial.zip" if method=="initial_random" else "final.zip")
        model=PPO.load(file,device="cpu")
        model_sha=sha(file)
        torch.manual_seed(6401)
    try:
        for _ in range(300):
            action=np.zeros(2)
            if model is not None:
                with torch.no_grad():
                    distribution=model.policy.get_distribution(torch.as_tensor(obs[None]))
                    tensor=distribution.sample() if method=="initial_random" else distribution.mode()
                action=tensor[0].cpu().numpy()
                env.context=dict(raw=action,version=0 if method=="initial_random" else 16)
            elif method=="classical50":
                owner=env.owner
                _,r,v=platform_state(owner.model,owner.data)
                ref=env.reference.at(owner.data.time)
                error=Rotation.from_matrix(r@ref.r.T).as_rotvec()
                action=(-gains["kp"]*error[:2]-gains["kd"]*(v[3:5]-ref.twist[3:5]))/SCALE
            obs,_,term,trunc,_=env.step(action)
            if term or trunc:
                break
    finally:
        env.close()
    ep=env.episodes[0]
    result=summarize(Path(ep["folder"]),"Home",method,task["torque"],task["on"],task["off"],ep["failure"])
    result.update(task_id=task["id"],model_sha256=model_sha,observation_clip_fraction=ep["clip_fraction"],
                  deterministic=method!="initial_random",controller_base=base)
    write_json(folder/"comparison.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("worker",choices=["0","1","2","3","collect"])
    a=p.parse_args().worker
    summaries=[json.loads((RUN/"training"/f"seed{s}"/"summary.json").read_text()) for s in [42,123,2026]]
    cfg=json.loads((ROOT/"configs/phase2c5_eval.yaml").read_text())
    base=json.loads((ROOT/"configs/phase2c5_attitude_env.yaml").read_text())["method"]
    if a=="collect":
        rows=sum([json.loads((RUN/f"eval_{i}.json").read_text()) for i in range(len(cfg["tasks"]))],[])
        assert len(rows)==6*len(cfg["tasks"])
        pd.DataFrame(rows).to_csv(ROOT/"results/phase2c5_classical_vs_ppo.csv",index=False)
        pd.DataFrame(summaries).to_csv(ROOT/"results/phase2c5_training_summary.csv",index=False)
        return
    if not all(s["decisions"]==8192 and not s["error"] and not s["stop"] for s in summaries):
        raise RuntimeError("Fixed pilot incomplete")
    torch.set_num_threads(1)
    task=cfg["tasks"][int(a)]
    rows=[]
    for method in METHODS:
        rows.append(run(task,method,base,cfg["classical_feedback_gains"]))
        write_json(RUN/f"eval_{a}.json",rows)


if __name__=="__main__":
    main()
