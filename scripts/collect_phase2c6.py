"""Collect gate evidence and independently verify the archived Phase 2C-6 runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from phase2c6_protocol import OLD, ROOT, RUN, SEEDS, config, read, sha, write_json
from phase2c6_history_audit import check_hashes
from src.rl.ppo_trace import parameter_hash
from validate_phase2c6_reward import physical, columns


def completed_seeds(curriculum=False):
    name="curriculum_training" if curriculum else "training"
    return [seed for seed in SEEDS if (p:=RUN/name/f"seed{seed}"/"summary.json").exists()
            and (s:=read(p))["decisions"]==8192 and not s["stop"] and not s["error"]]


def development():
    c=config("eval")
    rows=[]
    checks={}
    for seed in completed_seeds():
        paired=[]
        for task in c["development"]:
            t=task["id"]
            paths=[OLD/"evaluation"/t/name/"comparison.json" for name in ("baseline","classical50",f"PPO{seed}")]
            paths.append(RUN/"dev"/t/f"V2_{seed}"/"comparison.json")
            if not paths[-1].exists():
                break
            results=[read(path) for path in paths]
            rows.extend(dict(r,matched_seed=seed,phase="historical_V1" if i<3 else "new_V2")
                        for i,r in enumerate(results))
            paired.append(results[-2:])
        if len(paired)!=len(c["development"]):
            continue
        old,new=map(pd.DataFrame,zip(*paired))
        counts, mean_ratio={},{}
        for axis in ("load","unload"):
            key=axis+"_iae_deg_s"
            counts[axis]=int((new[key]<=old[key]*.95).sum())
            mean_ratio[axis]=float(new[key].mean()/old[key].mean())
        xyz=float(new.xyz_peak_mm.mean()/old.xyz_peak_mm.mean())
        slew=float(new.force_rate_N_s.max()/old.force_rate_N_s.max())
        accepted=bool(new.accepted_rms_Nm.min()>.0001)
        safe=bool(new.success.all() and xyz<=1.1 and slew<=1.1)
        improved=any(counts[k]>=2 and mean_ratio[k]<1 for k in counts)
        checks[seed]=dict(passed=bool(safe and accepted and improved),improvement_counts=counts,
                          mean_iae_ratio=mean_ratio,xyz_ratio=xyz,force_rate_ratio=slew,
                          all_tasks_complete=bool(new.success.all()),measurable_action=accepted)
    pd.DataFrame(rows).to_csv(ROOT/"results/phase2c6_reward_ablation.csv",index=False)
    write_json(RUN/"development_gate.json",dict(rule=c["dev_rule"],checks=checks,
               passing_seeds=[seed for seed,g in checks.items() if g["passed"]]))


def admission():
    candidates=config("curriculum")["candidates"]
    rows=[]
    admitted=[]
    for task in candidates:
        path=RUN/"curriculum_preflight"/task["id"]
        pair=[read(path/m/"comparison.json") for m in ("baseline","random")]
        rows.extend(pair)
        if all(r["success"] and r["load_band_01_status"]!="right_censored"
               and r["unload_band_01_status"]!="right_censored" for r in pair):
            admitted.append(task)
    all_levels={round(np.linalg.norm(t["torque"]),2) for t in admitted}
    pd.DataFrame(rows).to_csv(ROOT/"results/phase2c6_curriculum_preflight.csv",index=False)
    write_json(RUN/"curriculum_admission.json",dict(passed={.8,1.,1.2}<=all_levels,
               tested=len(candidates),admitted=admitted,
               rejected=[t for t in candidates if t not in admitted]))


def collect():
    c=config("eval")
    new=completed_seeds()
    curriculum=completed_seeds(True)
    summaries=[dict(read(OLD/"training"/f"seed{seed}"/"summary.json"),group="historical_reward_v1",reused=True)
               for seed in SEEDS]
    tb={}
    for group,folder in (("reward_v2","training"),("curriculum_v2","curriculum_training")):
        for p in sorted((RUN/folder).glob("seed*/summary.json")):
            row=read(p)
            summaries.append(dict(row,reused=False))
            model_file=ROOT/"models/phase2c6"/group/p.parent.name/"final.zip"
            assert sha(model_file)==row["model_sha"]
            model=PPO.load(model_file,device="cpu")
            assert parameter_hash(model.policy)==row["final_hash"]
            tags=set()
            for event in p.parent.glob("tensorboard/**/events.out.tfevents.*"):
                acc=EventAccumulator(str(event)).Reload()
                tags.update(acc.Tags()["scalars"])
            tb[str(p.parent.relative_to(RUN))]=sorted(tags)
    pd.DataFrame(summaries).to_csv(ROOT/"results/phase2c6_training_summary.csv",index=False)
    write_json(RUN/"tensorboard_scalar_tags.json",tb)
    methods=["baseline","classical50","V1_42","V1_123","V1_2026"]+[f"V2_{s}" for s in new]+[f"C2_{s}" for s in curriculum]
    selected=read(RUN/"final_selection.json")
    rows=[]
    for task in c["final"]+[c["regression"]]:
        for method in methods:
            f=RUN/"final"/task["id"]/method/"comparison.json"
            row=read(f)
            if method in selected["models"]:
                assert row["model_sha256"]==selected["models"][method]["sha256"] and row["policy_unchanged"]
            d=physical(row["folder"])
            active=d.failure.isna().to_numpy()
            force=columns(d,"force")
            add=(columns(d,"F_command")-columns(d,"F_C0"))[active]
            slew=np.diff(np.vstack([np.zeros((1,6)),add]),axis=0)*1000
            w=[]
            for edge in ("on","off"):
                step=round(task[edge]*1000)
                mask=(d.input_step>=step)&(d.input_step<step+1000)
                w.append(float(d.orientation_error_deg[mask].sum()*.001))
            assert abs(abs(force).max()-row["force_peak_N"])<1e-9
            assert abs(abs(add).max(initial=0.)-row["add_peak_N"])<1e-9
            assert abs(abs(slew).max(initial=0.)-row["add_rate_N_s"])<1e-7
            row["load_iae_recomputed_deg_s"],row["unload_iae_recomputed_deg_s"]=w
            # Primary windows are defined by the input step applying the physical wrench.
            row["load_iae_deg_s"],row["unload_iae_deg_s"]=w
            row["native_contacts_peak"]=int(d.native_contacts.max())
            row["warning_count_max"]=int(d.warning_count.max())
            row["shared_budget_peak_fraction"]=float(abs(add).max(initial=0.)/5)
            row["roll_peak_deg"]=float(abs(d.rpy_0).max())
            row["pitch_peak_deg"]=float(abs(d.rpy_1).max())
            rows.append(row)
    d=pd.DataFrame(rows)
    d.to_csv(ROOT/"results/phase2c6_classical_vs_ppo.csv",index=False)
    independent=d[d.task_id.isin([t["id"] for t in c["final"]])]
    classic=independent[independent.method=="classical50"].set_index("task_id")
    claims={}
    for m in methods[2:]:
        g=independent[independent.method==m].set_index("task_id")
        ratio={key:float(g[key].mean()/classic[key].mean()) for key in
               ("load_iae_deg_s","unload_iae_deg_s","so3_peak_deg","omega_peak_deg_s","xyz_peak_mm","force_rate_N_s")}
        wins=(g.load_iae_deg_s<classic.load_iae_deg_s)&(g.unload_iae_deg_s<classic.unload_iae_deg_s)
        xyz_safe=bool((g.xyz_peak_mm<=classic.xyz_peak_mm*1.1).all())
        slew_safe=bool((g.force_rate_N_s<=classic.force_rate_N_s*1.1).all())
        claims[m]=dict(ratio_vs_classical=ratio,joint_IAE_wins=int(wins.sum()),all_success=bool(g.success.all()),
                       xyz_gate=xyz_safe,slew_gate=slew_safe,
                       promotion=bool(wins.sum()>=2 and g.success.all() and xyz_safe and slew_safe))
    families={family:[r for m,r in claims.items() if m.startswith(family+"_")] for family in ("V2","C2")}
    latest=families["V2"]+families["C2"]
    go=any(sum(r["promotion"] for r in family)>=2 for family in families.values())
    mixed=any((r["joint_IAE_wins"]>0 or min(r["ratio_vs_classical"][k] for k in
              ("load_iae_deg_s","unload_iae_deg_s"))<1) and r["all_success"] and r["xyz_gate"] and r["slew_gate"]
              for r in latest)
    label=("PHASE 2C-6 SUCCESS — GO TO PHASE 2C-7" if go else
           "PHASE 2C-6 PARTIAL SUCCESS — NEED TARGETED ABLATION" if mixed else
           "PHASE 2C-6 NEGATIVE — STOP SCALING" if latest else "PHASE 2C-6 BLOCKED")
    write_json(ROOT/"results/phase2c6_final_decision.json",dict(decision=label,
               fixed_v2_seeds=new,curriculum_seeds=curriculum,rule=c["promotion_rule"],independent=claims,
               run=str(RUN),phase2c7_started=False))
    gate=read(RUN/"training_gate.json")
    training_safety=pd.read_csv(RUN/"training_safety_by_episode.csv")
    write_json(ROOT/"results/phase2c6_safety_audit.json",dict(training_gate=gate,
               development_gate=read(RUN/"development_gate.json"),preflight=read(RUN/"curriculum_admission.json"),
               evaluation_passed=int(d.success.sum()),evaluation_runs=len(d),
               native_force_peak_N=d.force_peak_N.max(),max_add_N=d.add_peak_N.max(),
               max_add_slew_N_s=d.add_rate_N_s.max(),
               training_samples_physical=int(training_safety.steps.sum()),
               training_protection_episodes=int(training_safety.failure.notna().sum()),
               training_actual_force_peak_N=float(training_safety.force_peak_N.max()),
               training_add_peak_N=float(training_safety.add_peak_N.max()),
               training_add_rate_peak_N_s=float(training_safety.add_rate_peak_N_s.max()),
               training_native_contacts_peak=int(training_safety.native_contacts_peak.max()),
               training_warning_count_peak=int(training_safety.warning_count_peak.max()),
               historical_hashes=read(RUN/"history_after.json"),
               geometry_scope="Frozen partial geometry coverage"))


def history():
    r=check_hashes(ROOT,read(RUN/"history_before.json"))
    write_json(RUN/"history_after.json",r)


def select():
    if (RUN/"final_selection.json").exists():
        raise RuntimeError("Final model selection is already frozen")
    dev=read(RUN/"development_gate.json")
    preflight=read(RUN/"curriculum_admission.json")
    fixed=completed_seeds()
    curriculum=completed_seeds(True)
    if len(dev["passing_seeds"])>=2 and preflight["passed"]:
        assert all((RUN/"curriculum_training"/f"seed{seed}"/"summary.json").is_file() for seed in fixed)
    models={}
    for label, seeds, parent in (("V1",SEEDS,"phase2c5"),("V2",fixed,"phase2c6/reward_v2"),
                                 ("C2",curriculum,"phase2c6/curriculum_v2")):
        for seed in seeds:
            p=ROOT/"models"/parent/f"seed{seed}"/"final.zip"
            models[f"{label}_{seed}"]=dict(path=str(p),sha256=sha(p))
    write_json(RUN/"final_selection.json",dict(fixed=fixed,curriculum=curriculum,models=models,
               eval_config_sha=sha(ROOT/"configs/phase2c6_eval.yaml"),
               selection="Final bounded checkpoint for every completed seed; no search on final cases"))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["development","admission","collect","history","select"])
    globals()[p.parse_args().mode]()
