import _bootstrap
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from run_phase2c5_attitude_tracking import trial
from src.model import ROOT, write_json
from src.rl.stewart_attitude_env import StewartAttitudeResidualEnv, failure_penalty, GAMMA
from src.workspace.attitude_study import RUN, make_attitude_owner


def prepare():
    gate = json.loads((RUN/"torque_gate.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("Torque gate failed")
    method = gate["method"]
    highest = max(gate["four_direction_levels"])
    amplitudes = [.15, .5] if highest >= 2 else [.15]
    train = []
    for magnitude in amplitudes:
        for axis in range(2):
            for sign in [-1,1]:
                torque = [0.,0.]
                torque[axis] = sign*magnitude
                on, off = (1.8,3.2) if sign == -1 else (2.1,3.8)
                train.append(dict(id=f"T{len(train)}", torque=torque, on=on, off=off))
    final = [dict(id="E_roll", torque=[.31 if highest>=2 else .11,0], on=2.4,off=3.9),
             dict(id="E_pitch", torque=[0,-.36 if highest>=2 else -.12],on=1.65,off=3.35),
             dict(id="E_mix", torque=[.24,-.21] if highest>=2 else [.08,-.09],on=2.2,off=3.7)]
    write_json(ROOT/"configs/phase2c5_attitude_env.yaml", dict(method=method, train=train,
               physical_dt=.001, stride=20, moment_scale_Nm=[.05,.05], features=52, tau_s=.005))
    write_json(ROOT/"configs/phase2c5_eval.yaml", dict(tasks=final, recovery_deg=.1, dwell_s=.5,
               classical_feedback_gains=dict(kp=25., kd=1.), frozen_before_training=True))
    write_json(ROOT/"configs/phase2c5_attitude_workspace.yaml", dict(
               source="phase2c3a_large_workspace.json", roll_pitch_guard_deg=.5, yaw_guard_deg=.2,
               admitted_reference_tasks=["Home","R1","R2","R3","Hold"], geometry_partial=True,
               no_wider_position_or_attitude_guard=True))


def config():
    return json.loads((ROOT/"configs/phase2c5_attitude_env.yaml").read_text())


def audit_episode(env, index):
    episode = env.episodes[index]
    fields=["orientation_error_deg","position_error_mm","accepted_mx_Nm","accepted_my_Nm",
            "decision","input_step","native_force_saturated","warning_count"]
    fields += [f"{prefix}_{axis}" for prefix in ("twist_actual","twist_ref") for axis in (3,4,5)]
    fields += [f"cost_{name}" for name in ("orientation","omega","xyz","action","slew")]
    fields += [f"{prefix}_{axis}" for prefix in ("F_command","F_C0","external_world") for axis in range(6)]
    d = pd.read_csv(Path(episode["folder"])/"physics.csv.gz",usecols=fields)
    error = d.orientation_error_deg.to_numpy()
    xyz = d.position_error_mm.to_numpy()
    omega = d[[f"twist_actual_{i}" for i in (3,4,5)]].to_numpy()-d[[f"twist_ref_{i}" for i in (3,4,5)]].to_numpy()
    a = d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()
    slew = np.diff(np.vstack([np.zeros((1,2)),a]),axis=0)*1000
    p = np.c_[(error/.05)**2, .2*((omega/.02)**2).sum(axis=1), .1*(xyz/.2)**2,
              .01*((a/.05)**2).sum(axis=1), .005*((slew/2.5)**2).sum(axis=1)]
    p *= np.minimum(1,100/np.maximum(p.sum(axis=1),1e-30))[:,None]
    actual = d[["cost_orientation","cost_omega","cost_xyz","cost_action","cost_slew"]].to_numpy()
    discrepancy = abs(p-actual).max()
    samples = pd.DataFrame(env.records)
    records = samples[samples.episode==episode["episode"]]
    by_decision = pd.Series(p.sum(axis=1)).groupby(d.decision).sum().to_numpy()
    np.testing.assert_allclose(records.reward.to_numpy(), -by_decision/20+records.terminal_penalty.to_numpy(), atol=1e-8)
    add = d[[f"F_command_{j}" for j in range(6)]].to_numpy()-d[[f"F_C0_{j}" for j in range(6)]].to_numpy()
    rate = abs(np.diff(np.vstack([np.zeros((1,6)),add]),axis=0)).max()*1000
    expected = ((d.input_step>=round(episode["task"]["on"]*1000)) &
                (d.input_step<round(episode["task"]["off"]*1000))).to_numpy()[:,None]*np.array(
                    [0,0,0,*episode["task"]["torque"],0])
    np.testing.assert_allclose(d[[f"external_world_{j}" for j in range(6)]], expected, atol=1e-9)
    assert discrepancy < 1e-8 and abs(add).max() <= 5+1e-8 and rate <= 20+1e-6
    assert not d.native_force_saturated.any() and not d.warning_count.max() and episode["clip_fraction"]<.001
    return dict(task=episode["task"]["id"], reward=episode["episode_return"], reward_error=discrepancy,
                steps=len(d), peak_deg=error.max(), add_N=abs(add).max(), rate_N_s=rate,
                accepted_peak_Nm=abs(a).max(), observation_clip_fraction=episode["clip_fraction"])


def validate_case(index):
    cfg = config()
    task = cfg["train"][index]
    env = StewartAttitudeResidualEnv(RUN/"validation"/f"random_{index}", [task], cfg["method"],
                                    source="Preflight random action")
    env.reset(seed=7401)
    rng = np.random.default_rng(7401+index)
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1,1,2))
        if terminated:
            env.close()
            raise RuntimeError("Random safety trial terminated")
    assert truncated
    row = audit_episode(env,0)
    assert row["accepted_peak_Nm"]>.005
    env.close()
    write_json(RUN/"validation"/f"random_{index}.json",row)


def zero_and_reward():
    cfg = config()
    task = cfg["train"][-1]
    root = RUN/"validation"
    env = StewartAttitudeResidualEnv(root/"zero", [task], cfg["method"], record_episodes=[0], source="Zero equivalence")
    env.reset(seed=42)
    frozen, _ = make_attitude_owner(root/"frozen", method=cfg["method"], torque=task["torque"], on=task["on"], off=task["off"])
    original = env.owner.step
    first = []

    def pair():
        original()
        frozen.step()
        o = env.owner
        for a,b in [(o.recorder.states[-1], frozen.recorder.states[-1]),(o.lag.state,frozen.lag.state),
                    (o.data.actuator_force,frozen.data.actuator_force),
                    (env.port.original.previous, frozen.residual.governor.previous),
                    (o.ref.r,frozen.ref.r)]:
            np.testing.assert_array_equal(a,b)
        if hasattr(o.residual.base,"observer"):
            for name in ("estimate","previous_velocity","previous_drive","steps","clipped"):
                np.testing.assert_equal(getattr(o.residual.base.observer,name),getattr(frozen.residual.base.observer,name))
        assert o.failure==frozen.failure
        if len(first)<20:
            first.append(o.recorder.states[-1].copy())

    env.owner.step = pair
    try:
        for _ in range(300):
            _,_,terminated,truncated,_ = env.step([0.,0.])
        assert truncated and not terminated
        frozen.recorder.close(dict(complete=True))
        rows = [dict(audit_episode(env,0), pattern="zero")]
        for repeat in range(2):
            env.reset(seed=42)
            env.step([0,0])
            np.testing.assert_array_equal(first, env.owner.recorder.states)
        env.close()
    finally:
        if not env.stream.closed:
            env.close()
    for sign in [-1,1]:
        case = StewartAttitudeResidualEnv(root/f"sign_{sign}", [task], cfg["method"], source="Reward development")
        case.reset(seed=42)
        for _ in range(300):
            _,_,term,trunc,_ = case.step([sign*.5,sign*.5])
            assert not term
        assert trunc
        rows.append(dict(audit_episode(case,0),pattern=f"sign_{sign}"))
        case.close()
    probe = StewartAttitudeResidualEnv(root/"gym",cfg["train"],cfg["method"],source="Gymnasium check")
    check_env(probe)
    probe.close()
    for n in (0,1,300):
        assert failure_penalty(n)<-sum(100*GAMMA**i for i in range(n))
    write_json(root/"zero_gate.json",dict(passed=True, bitwise_steps=6000, reset_checks=2,
               rgb_on_vs_off=True, gym_check=True, discounted_failure_bound=True, rows=rows))


def final_preflight():
    cfg=config()
    tasks=json.loads((ROOT/"configs/phase2c5_eval.yaml").read_text())["tasks"]
    rows=[trial(RUN/"validation"/"final"/t["id"],"Home",cfg["method"],t["torque"],t["on"],t["off"]) for t in tasks]
    write_json(RUN/"validation"/"final_gate.json",dict(passed=all(r["success"] and not r["load_right_censored"]
               and not r["unload_right_censored"] for r in rows), rows=rows))


def collect():
    cfg=config()
    root=RUN/"validation"
    zero=json.loads((root/"zero_gate.json").read_text())
    final=json.loads((root/"final_gate.json").read_text())
    boundary=[json.loads((RUN/f"strong_{i}.json").read_text()) for i in range(4)]
    strong=[next((r for r in rows if r["amplitude_Nm"]==2.4 and r["recoverable"]),None) for rows in boundary]
    if all(strong):
        path=ROOT/"configs/phase2c5_eval.yaml"
        cfg_eval=json.loads(path.read_text())
        cfg_eval["tasks"].append(dict(id="E_strong_regression",torque=[2.4,0.],on=2.,off=3.5,
                                     scope="Already preflighted Level 3 regression, not unseen"))
        write_json(path,cfg_eval)
        final["strong_regression_evidence"]=strong[0]
    random=[json.loads((root/f"random_{i}.json").read_text()) for i in range(len(cfg["train"]))]
    traces = [pd.read_csv(root/f"sign_{sign}"/"episode_0000/physics.csv.gz",
                         usecols=["time_s","rpy_0","rpy_1"]) for sign in [-1,1]]
    delta = (traces[1]-traces[0]).loc[(traces[0].time_s>.7)&(traces[0].time_s<1.5),["rpy_0","rpy_1"]].mean()
    assert np.all(delta.to_numpy()>.0001), f"No verified residual authority: {delta}"
    pd.DataFrame(zero["rows"]+[dict(r,pattern="random") for r in random]).to_csv(
        ROOT/"results/phase2c5_reward_audit.csv",index=False)
    write_json(RUN/"training_gate.json",dict(passed=zero["passed"] and final["passed"],
               zero=zero,final=final,random=random,measured_signed_action_difference_deg=delta.to_dict(),
               promotion_rule="At least two seeds beat classical50 in both window IAEs on each of 3 unseen tasks, with no protection and no >10% increase in XYZ or native force slew. Strong regression is reported separately."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode")
    a=p.parse_args().mode
    if a.isdigit():
        validate_case(int(a))
    else:
        dict(prepare=prepare,zero=zero_and_reward,final=final_preflight,collect=collect)[a]()
