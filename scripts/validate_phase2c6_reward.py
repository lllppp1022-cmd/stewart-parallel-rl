"""Recompute rewards from physical samples, and gate new optimization."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from phase2c6_protocol import OLD, ROOT, RUN, SEEDS, config, read, sha, write_json
from src.rl.attitude_reward_v2 import NAMES, SCALES, WEIGHTS
from src.rl.ppo_trace import parameter_hash
from src.rl.stewart_attitude_env import CAP, GAMMA, failure_penalty


def columns(d, prefix, axes=range(6)):
    return d[[f"{prefix}_{j}" for j in axes]].to_numpy()


def physical(folder):
    return pd.read_csv(Path(folder)/"physics.csv.gz")


def recompute(d):
    a = d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()
    slew = np.diff(np.vstack([np.zeros((1,2)),a]),axis=0)/.001
    omega = columns(d,"twist_actual",(3,4,5))-columns(d,"twist_ref",(3,4,5))
    x = np.c_[d.orientation_error_deg**2, (omega**2).sum(1), d.position_error_mm**2,
              (a**2).sum(1), (slew**2).sum(1)]
    raw = {
        "V1": x * np.array([1,.2,.1,.01,.005])/np.square([.05,.02,.2,.05,2.5]),
        "slew_only": x * np.array([1,.2,.1,.01,WEIGHTS[-1]])/np.square([.05,.02,.2,.05,2.5]),
        "V2": x * np.asarray(WEIGHTS)/np.square(SCALES),
    }
    return {k:(v*np.minimum(1,CAP/np.maximum(v.sum(1),1e-30))[:,None],v.sum(1)>CAP)
            for k,v in raw.items()}


def audit(root, label, components, actions, totals, safety=None):
    records = pd.read_csv(root/"training_samples.csv")
    for ep in read(root/"episodes.json"):
        rows = records[records.episode==ep["episode"]].reset_index(drop=True)
        if not len(rows):
            continue
        d = physical(ep["folder"])
        if safety is not None:
            assert len(d)==ep["physical_steps"]==int(rows.physical_steps.sum())
            active=d.failure.isna().to_numpy()
            add=(columns(d,"F_command")-columns(d,"F_C0"))[active]
            force=columns(d,"force")
            slew=np.diff(np.vstack([np.zeros((1,6)),add]),axis=0)*1000
            assert np.isfinite(force).all() and np.isfinite(add).all()
            assert abs(force).max()<=400+1e-8 and abs(add).max(initial=0.)<=5+1e-8
            assert abs(slew).max(initial=0.)<=20+1e-6
            safety.append(dict(group=label.split("/")[0],seed=label.split("/")[1],episode=ep["episode"],
                               task=ep["task"]["id"],steps=len(d),failure=ep["failure"],
                               force_peak_N=float(abs(force).max()),add_peak_N=float(abs(add).max(initial=0.)),
                               add_rate_peak_N_s=float(abs(slew).max(initial=0.)),
                               native_contacts_peak=int(d.native_contacts.max()),
                               warning_count_peak=int(d.warning_count.max())))
        versions = recompute(d)
        for version, (p, cap) in versions.items():
            if version=="V1" or (version=="V2" and "v2_cost_orientation" in d):
                prefix = "cost_" if version=="V1" else "v2_cost_"
                np.testing.assert_allclose(p,d[[prefix+n for n in NAMES]],rtol=0,atol=1e-8)
            score = pd.Series(p.sum(1)).groupby(d.decision).sum().to_numpy()/20
            discount = GAMMA**np.arange(len(rows))
            penalty = rows.terminal_penalty.to_numpy()
            live = "V2" if "v2_cost_orientation" in d else "V1"
            if version==live:
                np.testing.assert_allclose(-score+penalty,rows.reward,rtol=0,atol=1e-8)
            totals.append(dict(label=label,episode=ep["episode"],task=ep["task"]["id"],
                               reward=version,discounted_return=float((-score+penalty)@discount),
                               undiscounted_return=float((-score+penalty).sum()),complete=ep["complete"],
                               failure=ep["failure"],physical_steps=len(d)))
            on, off = (ep["task"][k] for k in ("on","off"))
            t = d.input_step.to_numpy()*.001
            windows = {"preload":t<on, "load_edge":(t>=on)&(t<on+.001-.00000001),
                       "load_first_100ms":(t>=on)&(t<on+.1-1e-9),
                       "load_sustained":(t>=on+.1-1e-9)&(t<off),
                       "unload_first_100ms":(t>=off)&(t<off+.1-1e-9),
                       "unload_recovery":t>=off+.1-1e-9}
            if version=="V1":
                requested=d[["requested_mx_Nm","requested_my_Nm"]].to_numpy()
                accepted=d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()
                extra=columns(d,"F_command")-columns(d,"F_C0")
                native_rate=np.abs(np.diff(np.vstack([np.zeros((1,6)),extra]),axis=0))/.001
                raw=rows[["raw_roll","raw_pitch"]].to_numpy()
                boxed=rows[["clipped_roll","clipped_pitch"]].to_numpy()
                np.testing.assert_allclose(boxed,np.clip(raw,-1,1),atol=1e-7)
            for window, mask in windows.items():
                if not mask.any():
                    continue
                row = dict(label=label,episode=ep["episode"],task=ep["task"]["id"],version=version,window=window,
                           n=int(mask.sum()),cap_fraction=float(cap[mask].mean()))
                for k, v in zip(NAMES,p[mask].T):
                    row[f"{k}_sum"] = v.sum()*.001
                    row[f"{k}_min"] = v.min()
                    row[f"{k}_max"] = v.max()
                row["terminal_penalty_episode"]=float(penalty.sum())
                components.append(row)
                if version=="V1":
                    decisions=np.unique(d.decision.to_numpy()[mask])-1
                    rms=lambda v: float(np.sqrt(np.mean(v**2)))
                    actions.append(dict(label=label,episode=ep["episode"],task=ep["task"]["id"],window=window,
                      n=int(mask.sum()), request_rms_Nm=rms(requested[mask]),accepted_rms_Nm=rms(accepted[mask]),
                      request_minus_accepted_rms_Nm=rms((requested-accepted)[mask]),
                      box_clip_fraction=float(rows.action_box_clipped.iloc[decisions].mean()),
                      projection_fraction=float(d.projected[mask].mean()),
                      shared_slew_boundary_fraction=float(np.any(native_rate[mask]>=20-1e-7,axis=1).mean()),
                      raw_action_rms=rms(raw[decisions]),mapped_leg_rms_N=rms(columns(d,"rl_mapped_leg", [f"{j}_N" for j in range(6)])[mask]),
                      command_rms_N=rms(columns(d,"F_command")[mask]),drive_rms_N=rms(columns(d,"F_drive")[mask]),
                      actual_force_rms_N=rms(columns(d,"force")[mask])))


def probes():
    root=RUN/"probe"/"T7"
    names=("baseline","xp","xm","yp","ym","baseline_v1")
    traces={k:physical(root/k/"episode_0000") for k in names}
    baseline=traces["baseline"]
    np.testing.assert_array_equal(
        np.load(root/"baseline"/"episode_0000"/"integration_states.npz")["states"],
        np.load(root/"baseline_v1"/"episode_0000"/"integration_states.npz")["states"])
    frozen = OLD/"validation"/"frozen"/"integration_states.npz"
    np.testing.assert_array_equal(np.load(frozen)["states"],
                                  np.load(root/"baseline"/"episode_0000"/"integration_states.npz")["states"])
    assert all(read(root/k/"comparison.json")["success"] for k in names)
    out=[]
    for name in names[1:5]:
        for ms in (20,40,60,100):
            i=2100+ms-1
            d=traces[name]
            local=d.iloc[2100:i+1]
            line=dict(probe=name,after_load_ms=ms,so3_delta_deg=d.orientation_error_deg[i]-baseline.orientation_error_deg[i])
            for j, axis in enumerate(("x","y")):
                line[f"accepted_m{axis}_Nm"]=local[f"accepted_m{axis}_Nm"].mean()
                line[f"delta_omega_{axis}_rad_s"]=d[f"twist_actual_{j+3}"][i]-baseline[f"twist_actual_{j+3}"][i]
                line[f"delta_{axis}_mm"]=(d[f"p_{j}"][i]-baseline[f"p_{j}"][i])*1000
            for kind in ("F_command","F_drive","force"):
                line[f"delta_{kind}_l2_N"]=np.linalg.norm(columns(d,kind)[i]-columns(baseline,kind)[i])
            out.append(line)
    pd.DataFrame(out).to_csv(ROOT/"results/phase2c6_probe_causality.csv",index=False)
    for line in out:
        if line["after_load_ms"]==100:
            axis, sign=line["probe"]
            assert line[f"accepted_m{axis}_Nm"]*(1 if sign=="p" else -1)>.005
            assert line[f"delta_omega_{axis}_rad_s"]*(1 if sign=="p" else -1)>0
    return names


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["historical","gate","trained"])
    args=p.parse_args()
    components,actions,totals=[],[],[]
    hashes={}
    if args.stage=="historical":
        for seed in SEEDS:
            f=ROOT/f"models/phase2c5/seed{seed}/final.zip"
            m=PPO.load(f,device="cpu")
            assert m.observation_space.shape==(52,) and m.action_space.shape==(2,)
            history=read(OLD/"training"/f"seed{seed}"/"summary.json")
            assert parameter_hash(m.policy)==history["final_hash"] and sha(f)==history["model_sha"]
            hashes[seed]=history["model_sha"]
            audit(OLD/"training"/f"seed{seed}",f"V1_train_{seed}",components,actions,totals)
        for task in config("eval")["development"]+[config("eval")["regression"]]:
            for method in ("baseline","classical50","initial_random","PPO42","PPO123","PPO2026"):
                audit(OLD/"evaluation"/task["id"]/method,f"{task['id']}/{method}",components,actions,totals)
        write_json(RUN/"v1_model_reloads.json",hashes)
        pd.DataFrame(components).to_csv(ROOT/"results/phase2c6_reward_component_audit.csv",index=False)
        pd.DataFrame(actions).to_csv(ROOT/"results/phase2c6_action_authority.csv",index=False)
        pd.DataFrame(totals).to_csv(RUN/"historical_rescores.csv",index=False)
        return
    if args.stage=="trained":
        safety=[]
        for group in ("training","curriculum_training"):
            for root in (RUN/group).glob("seed*"):
                if (root/"summary.json").is_file():
                    audit(root,f"{group}/{root.name}",components,actions,totals,safety)
        pd.DataFrame(components).to_csv(RUN/"v2_training_components.csv",index=False)
        pd.DataFrame(actions).to_csv(RUN/"v2_training_actions.csv",index=False)
        pd.DataFrame(totals).to_csv(RUN/"v2_training_rescores.csv",index=False)
        pd.DataFrame(safety).to_csv(RUN/"training_safety_by_episode.csv",index=False)
        return
    for method in probes():
        audit(RUN/"probe"/"T7"/method,method,components,actions,totals)
    d=pd.concat([pd.read_csv(RUN/"historical_rescores.csv"),pd.DataFrame(totals)],ignore_index=True)
    d.to_csv(RUN/"offline_reward_ordering.csv",index=False)
    rank=d.pivot_table(index="label",columns="reward",values="discounted_return")
    for task in config("eval")["development"]:
        assert rank.loc[f"{task['id']}/classical50","V2"]>rank.loc[f"{task['id']}/baseline","V2"]
    assert rank.loc["E_mix/initial_random","V2"]<rank.loc["E_mix/classical50","V2"]
    for n in (0,1,150,299):
        assert failure_penalty(n)<-sum(CAP*GAMMA**i for i in range(n))
    assert all(read(RUN/"preflight"/t["id"]/"baseline"/"comparison.json")["success"]
               for t in config("eval")["final"])
    pd.DataFrame(components).to_csv(RUN/"probe_reward_components.csv",index=False)
    pd.DataFrame(actions).to_csv(RUN/"probe_action_windows.csv",index=False)
    write_json(RUN/"training_gate.json",dict(passed=True,zero_bitwise_steps=6000,
               signed_action_probe=True,independent_baseline_preflight=True,all_v1_models_reloaded=True,
               offline_classical_over_zero_all_three=True,random_worse_than_classical=True,
               failure_discounted_bound=True,configs={k:sha(ROOT/f"configs/phase2c6_{k}.yaml")
               for k in ("ppo","reward_v2","eval","curriculum")}))


if __name__=="__main__":
    main()
