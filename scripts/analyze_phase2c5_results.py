"""Independent numerical audits and fixed, all-seed attitude comparisons."""
import _bootstrap
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.model import ROOT, write_json
from src.workspace.attitude_study import RUN
from validate_phase2c5_env import audit_episode


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def training_audit():
    rows=[]
    reward_rows=[]
    for seed in (42,123,2026):
        folder=RUN/"training"/f"seed{seed}"
        samples=pd.read_csv(folder/"training_samples.csv")
        costs = -samples[["reward_orientation","reward_omega","reward_xyz","reward_action","reward_slew"]].sum()
        for name,value in costs.items():
            reward_rows.append(dict(seed=seed,component=name,total_cost=value,share=value/costs.sum()))
        ep=read(folder/"episodes.json")
        env=SimpleNamespace(episodes=ep,records=samples)
        for i,item in enumerate(ep):
            if item["physical_steps"]:
                rows.append(dict(seed=seed,episode=i,complete=item["complete"],**audit_episode(env,i)))
        for k, f in enumerate(sorted(folder.glob("rollout_*.npz"))):
            with np.load(f) as z:
                actual=samples[["raw_roll","raw_pitch"]].iloc[k*512:(k+1)*512]
                np.testing.assert_allclose(z["actions"].reshape(-1,2),actual,atol=1e-7)
    pd.DataFrame(rows).to_csv(ROOT/"results/phase2c5_training_physical_audit.csv",index=False)
    pd.DataFrame(reward_rows).to_csv(ROOT/"results/phase2c5_training_reward_components.csv",index=False)
    return rows


def compare():
    d=pd.read_csv(ROOT/"results/phase2c5_classical_vs_ppo.csv")
    checks=[]
    for row in d.itertuples():
        p=pd.read_csv(Path(row.folder)/"physics.csv.gz")
        target=Rotation.from_euler("xyz",p[["rpy_ref_0","rpy_ref_1","rpy_ref_2"]],degrees=True)
        actual=Rotation.from_euler("xyz",p[["rpy_0","rpy_1","rpy_2"]],degrees=True)
        error=np.rad2deg((target*actual.inv()).magnitude())
        diff=abs(error-p.orientation_error_deg).max()
        assert diff<1e-8 and abs(error.max()-row.so3_peak_deg)<1e-8
        assert p.generalized_mapping_error.max()<1e-9
        for label,t in [("load",row.on_s),("unload",row.off_s)]:
            assert abs(error[(p.time_s>=t)&(p.time_s<t+1)].sum()*.001-getattr(row,f"{label}_iae_deg_s"))<1e-8
        checks.append(dict(task=row.task_id,method=row.method,so3_difference_deg=diff,steps=len(p),
                           protected=bool(p.failure.notna().any()),weld_m=p.weld_translation_m.max(),
                           weld_rad=p.weld_angle_rad.max(),geometry_m=p.expanded_mesh_gap_m.min(),
                           max_condition=p.actual_jacobian_condition.max(),
                           external_mapping_error=p.generalized_mapping_error.max()))
    comparisons=[]
    for r in d[d.method.str.startswith("PPO")].itertuples():
        b=d[(d.task_id==r.task_id)&(d.method=="classical50")].iloc[0]
        row=dict(task=r.task_id,method=r.method,success=r.success)
        for k in ["so3_peak_deg","omega_peak_deg_s","load_iae_deg_s","unload_iae_deg_s",
                  "xyz_peak_mm","yaw_peak_deg","force_rate_N_s","force_energy_N2_s"]:
            value=getattr(r,k)
            row[k]=value
            row[f"delta_{k}"]=b[k]-value
            row[f"gain_{k}"]=(b[k]-value)/b[k] if b[k] else 0.
        row["promotable"]=bool(r.success and row["delta_load_iae_deg_s"]>0 and row["delta_unload_iae_deg_s"]>0
                            and r.xyz_peak_mm<=1.1*b.xyz_peak_mm and r.force_rate_N_s<=1.1*b.force_rate_N_s)
        comparisons.append(row)
    c=pd.DataFrame(comparisons)
    c.to_csv(ROOT/"results/phase2c5_delta_vs_classical.csv",index=False)
    independent=c[~c.task.str.contains("regression")]
    passed={s:bool(independent[independent.method==f"PPO{s}"].promotable.all()) for s in [42,123,2026]}
    if sum(passed.values())>=2 and d.success.all():
        result="ATTITUDE PPO PILOT SUCCESS"
    elif not d[d.method.str.startswith("PPO")].success.all() or (
            independent.groupby("method").delta_load_iae_deg_s.mean().lt(0).all() and
            independent.groupby("method").delta_unload_iae_deg_s.mean().lt(0).all()):
        result="ATTITUDE PPO PILOT NEGATIVE"
    else:
        result="ATTITUDE PPO PILOT INCONCLUSIVE"
    write_json(ROOT/"results/phase2c5_final_decision.json",dict(decision=result,seed_gates=passed,
               promote_50mm_combination=result=="ATTITUDE PPO PILOT SUCCESS",phase2c5c_executed=False,
               classical_reference="Same 50Hz, same +/-0.05Nm per axis and shared force budget",
               evaluated_trials=len(d),passed_trials=int(d.success.sum())))
    return d,c,checks


def plots(d):
    source=Path(d[(d.task_id=="E_mix")&(d.method=="PPO42")].iloc[0].folder)
    p=pd.read_csv(source/"physics.csv.gz")
    fig,axes=plt.subplots(5,1,sharex=True,figsize=(12,11))
    for i,axis in enumerate(("mx","my")):
        axes[0].plot(p.time_s,p[f"external_world_{i+3}"],label=axis)
        axes[1].plot(p.time_s,p[f"rpy_{i}"],label=["Roll","Pitch"][i])
        axes[2].plot(p.time_s,np.rad2deg(p[f"twist_actual_{i+3}"]),label=["Roll","Pitch"][i])
        axes[3].plot(p.time_s,p[f"requested_{axis}_Nm"],"--",label=axis+" request")
        axes[3].plot(p.time_s,p[f"accepted_{axis}_Nm"],label=axis+" accepted command")
    for j in range(6):
        axes[4].plot(p.time_s,p[f"force_{j}"],label=f"Leg {j+1}")
    for a,name in zip(axes,["External moment (Nm)","Actual attitude (deg)","Angular velocity (deg/s)",
                             "PPO residual moment (Nm)","Actual actuator force (N)"]):
        a.set_ylabel(name)
        a.legend(ncol=3,fontsize=8)
        a.grid(alpha=.25)
    axes[-1].set_xlabel("Physical time (s)")
    fig.suptitle("Independent E_mix | deterministic seed42 | actual 1 ms logs")
    fig.tight_layout()
    fig.savefig(ROOT/"assets/plots/phase2c5_disturbance_explained.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for seed in (42,123,2026):
        folder=RUN/"training"/f"seed{seed}"
        s=pd.read_csv(folder/"training_samples.csv")
        u=pd.DataFrame(read(folder/"updates.json"))
        ep=s[s.truncated]
        axes[0,0].plot(ep.global_decision,ep.episode_return,label=str(seed))
        axes[0,1].plot(u.decisions,u["train/value_loss"],label=str(seed))
        axes[1,0].plot(u.decisions,u.actor_delta,label=str(seed))
        axes[1,1].plot(u.decisions,u["train/approx_kl"],label=str(seed))
    for a,name in zip(axes.flat,["Complete episode return | randomized tasks","Value loss","Actor parameter change","Approx KL"]):
        a.set_title(name)
        a.set_xlabel("Actual PPO decisions")
        a.legend()
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(ROOT/"assets/plots/phase2c5_learning_curves.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","training","evaluation"],default="all")
    stage=parser.parse_args().stage
    if stage!="evaluation":
        train=training_audit()
        print("AUDITED",len(train),"training episodes and partial prefixes",flush=True)
        if stage=="training":
            return
    else:
        train=pd.read_csv(ROOT/"results/phase2c5_training_physical_audit.csv").to_dict("records")
    assert sum(r["steps"] for r in train)==491520
    d,c,checks=compare()
    plots(d)
    write_json(ROOT/"results/phase2c5_safety_audit.json",dict(
        passed=not any(x["protected"] for x in checks),
        training_audit_episodes=len(train),evaluation=checks,
        partial_geometry=True,passive_hard_stops_unknown=30))
    print("AUDITED",len(train),"training episodes and",len(checks),"evaluation trials")


if __name__=="__main__":
    main()
