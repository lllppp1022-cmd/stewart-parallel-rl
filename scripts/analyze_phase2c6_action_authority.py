"""Analyze action authority using archived true maps and matched 1 ms runs."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from phase2c6_protocol import OLD, ROOT, RUN, SEEDS, config, read
from validate_phase2c6_reward import columns, physical


def true_maps(d):
    return columns(d,"true_kinematic_map",range(36)).reshape(-1,6,6)


def audit_root(root, label, out):
    for ep in read(root/"episodes.json"):
        if ep["physical_steps"]==0:
            continue
        d=physical(ep["folder"])
        requested=d[["requested_mx_Nm","requested_my_Nm"]].to_numpy()
        accepted=d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()
        c0=columns(d,"F_C0")
        command=columns(d,"F_command")
        base=c0+columns(d,"same_state_base_add",[f"{j}_N" for j in range(6)])
        leg=columns(d,"rl_mapped_leg",[f"{j}_N" for j in range(6)])
        unprojected=columns(d,"F_C3")+columns(d,"F_residual_requested")+leg
        previous=np.vstack([np.zeros((1,6)),(command-c0)[:-1]])
        slew_intervened=np.any(abs((unprojected-c0)-previous)>.02+1e-8,axis=1)
        assert np.max(abs(command))<400 and np.max(abs(base))<400
        a=-np.expm1(-.001/.005)
        # Exact ZOH incremental drive in an action-off shadow at the logged state.
        delta=lfilter([a],[1,-(1-a)],command-base,axis=0)
        drive=columns(d,"F_drive")
        force=columns(d,"force")
        np.testing.assert_allclose(force,drive,atol=1e-9)
        assert np.max(abs(drive-delta))<400
        physical_effect=np.einsum("nji,nj->ni",true_maps(d),force-(drive-delta))[:,3:5]
        t=d.input_step.to_numpy()*.001
        on,off=(ep["task"][k] for k in ("on","off"))
        sign=np.asarray(ep["task"]["torque"])
        direction=sign/np.linalg.norm(sign)
        windows={"preload": t<on, "sustained": (t>=on+.1-1e-9)&(t<off), "recovery":t>=off+.1-1e-9}
        for ms in (20,40,60,100):
            windows[f"load_{ms}ms"]=(t>=on)&(t<on+ms*.001-1e-9)
            windows[f"unload_{ms}ms"]=(t>=off)&(t<off+ms*.001-1e-9)
        for name, mask in windows.items():
            if not mask.any():
                continue
            rms=lambda x:float(np.sqrt(np.mean(x[mask]**2)))
            out.append(dict(label=label,episode=ep["episode"],task=ep["task"]["id"],window=name,
                       n=int(mask.sum()),request_RMS_Nm=rms(requested),accepted_RMS_Nm=rms(accepted),
                       lagged_same_state_RMS_Nm=rms(physical_effect),
                       request_minus_lagged_RMS_Nm=rms(requested-physical_effect),
                       requested_along_load_Nm=(requested[mask]@direction).mean(),
                       accepted_along_load_Nm=(accepted[mask]@direction).mean(),
                       lagged_along_load_Nm=(physical_effect[mask]@direction).mean(),
                       raw_combined_leg_RMS_N=rms(unprojected),accepted_total_add_RMS_N=rms(command-c0),
                       shared_slew_intervention_fraction=float(slew_intervened[mask].mean()),
                       final_command_RMS_N=rms(command),drive_RMS_N=rms(drive),measured_force_RMS_N=rms(force)))


def main():
    out=[]
    for seed in SEEDS:
        audit_root(OLD/"training"/f"seed{seed}",f"V1_train_{seed}",out)
    for t in config("eval")["development"]+[config("eval")["regression"]]:
        for m in ("baseline","classical50","initial_random","PPO42","PPO123","PPO2026"):
            audit_root(OLD/"evaluation"/t["id"]/m,f"{t['id']}/{m}",out)
    for m in ("baseline","xp","xm","yp","ym"):
        audit_root(RUN/"probe/T7"/m,f"T7/{m}",out)
    pd.DataFrame(out).to_csv(ROOT/"results/phase2c6_action_lag_windows.csv",index=False)
    path=ROOT/"results/phase2c6_probe_causality.csv"
    d=pd.read_csv(path)
    root=RUN/"probe/T7"
    base=physical(root/"baseline/episode_0000")
    baseline_map=true_maps(base)
    f0=columns(base,"force")
    for method in d.probe.unique():
        p=physical(root/method/"episode_0000")
        f=columns(p,"force")
        delta=np.einsum("nji,nj->ni",true_maps(p),f)-np.einsum("nji,nj->ni",baseline_map,f0)
        for ms in (20,40,60,100):
            mask=(d.probe==method)&(d.after_load_ms==ms)
            for j, axis in enumerate(("x","y")):
                d.loc[mask,f"delta_real_actuator_wrench_m{axis}_Nm"]=delta[2100+ms-1,3+j]
    d.to_csv(path,index=False)


if __name__=="__main__":
    main()
