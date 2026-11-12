"""Finite geometry and baseline/random joint preflight."""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

from phase2c7_protocol import ROOT,RUN,config,read,write_json
from src.model import sha
from src.workspace.dynamic_attitude_study import DynamicReference,reference_key
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv
from src.rl.stewart_attitude_env import StewartAttitudeResidualEnv


def metrics(folder,ep):
    d=pd.read_csv(folder/"physics.csv.gz")
    e=d[[f"p_{i}" for i in range(3)]].to_numpy()-d[[f"p_ref_{i}" for i in range(3)]].to_numpy()
    f=d[[f"force_{i}" for i in range(6)]].to_numpy()
    add=d[[f"F_command_{i}" for i in range(6)]].to_numpy()-d[[f"F_C0_{i}" for i in range(6)]].to_numpy()
    active=d.failure.isna().to_numpy()
    rate=abs(np.diff(np.vstack([np.zeros((1,6)),add[active]]),axis=0)).max(initial=0)*1000
    tail=d.tail(500)
    expected=((d.input_step>=round(ep["task"]["on"]*1000))&(d.input_step<round(ep["task"]["off"]*1000))).to_numpy()[:,None]*np.array([0,0,0,*ep["task"]["torque"],0])
    ext=abs(d[[f"external_world_{i}" for i in range(6)]].to_numpy()-expected).max()
    result=dict(task=ep["task"],folder=str(folder),complete=ep["complete"],failure=ep["failure"],steps=len(d),
        success=bool(ep["complete"] and tail.position_error_mm.max()<.75 and tail.orientation_error_deg.max()<.1
                     and abs(add[active]).max(initial=0)<=5+1e-8 and rate<=20+1e-6 and ext<1e-9),
        so3_peak_deg=d.orientation_error_deg.max(),omega_peak_rad_s=np.linalg.norm(d[[f"twist_actual_{i}" for i in (3,4,5)]],axis=1).max(),
        z_rmse_mm=float(np.sqrt(np.mean(e[:,2]**2))*1000),xyz_rmse_mm=float(np.sqrt(np.mean(np.sum(e**2,axis=1)))*1000),
        xy_peak_mm=float(abs(e[:,:2]).max()*1000),z_peak_m=float(d.actual_z_from_home_m.max()),
        force_peak_N=abs(f).max(),force_rate_N_s=abs(np.diff(f,axis=0)).max()*1000,
        add_peak_N=abs(add[active]).max(initial=0),add_rate_N_s=rate,cap_fraction=float((d.reward_cap_factor<1).mean()),
        observation_clip_fraction=ep["clip_fraction"],external_error=ext,
        accepted_rms_Nm=float(np.sqrt(np.mean(d[["accepted_mx_Nm","accepted_my_Nm"]].to_numpy()**2))),
        condition_peak=d.actual_jacobian_condition.max(),weld_translation_m=d.weld_translation_m.max(),
        weld_angle_rad=d.weld_angle_rad.max(),clearance_m=d.expanded_mesh_gap_m.min())
    for name,edge in (("load","on"),("unload","off")):
        mask=(d.input_step>=round(ep["task"][edge]*1000))&(d.input_step<round(ep["task"][edge]*1000)+1000)
        result[name+"_iae_deg_s"]=float(d.orientation_error_deg[mask].sum()*.001)
    return result


def trial(task,method,group="preflight",model=None,record=False):
    root=RUN/group/task["id"]/method
    if (root/"metrics.json").exists():
        return read(root/"metrics.json")
    env=StewartDynamicAttitudeEnv(root,[task],[0] if record else [],source=group+"/"+method)
    obs,_=env.reset(seed=6401)
    rng=np.random.default_rng(6401)
    policy=PPO.load(model,device="cpu") if model else None
    try:
        for _ in range(round(env.reference.duration*50)):
            a=np.zeros(2)
            if method=="random":a=rng.uniform(-1,1,2)
            elif method=="classical50":
                o=env.owner
                from src.phase1.plant import platform_state
                _,r,v=platform_state(o.model,o.data)
                ref=env.reference.at(o.data.time)
                a=(-25*Rotation.from_matrix(r@ref.r.T).as_rotvec()[:2]-(v[3:5]-ref.twist[3:5]))/.05
            elif policy:
                x=StewartAttitudeResidualEnv.observe(env) if policy.observation_space.shape==(52,) else obs
                a=policy.predict(x,deterministic=True)[0]
                env.context['version']=policy.num_timesteps//512
            elif method in ("positive","negative"):
                a=np.ones(2)*(.5 if method=="positive" else -.5)
            env.context["raw"]=a
            obs,_,term,trunc,_=env.step(a)
            if term or trunc:break
    finally:env.close()
    ep=env.episodes[0]
    result=metrics(__import__("pathlib").Path(ep["folder"]),ep)
    result["method"]=method
    result['model_sha256']=sha(model) if model else None
    write_json(root/"metrics.json",result)
    print(task["id"],method,result["success"],result["failure"],result["so3_peak_deg"],flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["geometry","preflight","collect"])
    p.add_argument("--worker",type=int,default=0)
    a=p.parse_args()
    curriculum=config("curriculum")
    tasks=sum([curriculum[k] for k in ("A","B","C")],[])+config("eval")["final"]+config("eval")["development"]
    torch.set_num_threads(1)
    if a.stage=="geometry":
        specs={reference_key(t["reference"]):t["reference"] for t in tasks}
        for i,(key,spec) in enumerate(specs.items()):
            if i%3==a.worker:
                DynamicReference(spec,build=True)
                print(key,"GEOMETRY PASS",flush=True)
    elif a.stage=="preflight":
        for i,t in enumerate(tasks):
            if i%3==a.worker:
                for m in ("baseline","random"):
                    trial(t,m,record=t["id"]=="C2" and m=="baseline")
    else:
        rows=[read(RUN/"preflight"/t["id"]/m/"metrics.json") for t in tasks for m in ("baseline","random")]
        accepted=[t for t in tasks if all(read(RUN/"preflight"/t["id"]/m/"metrics.json")["success"] for m in ("baseline","random"))]
        pd.DataFrame(rows).to_csv(ROOT/"results/phase2c7_joint_preflight.csv",index=False)
        write_json(RUN/"admission.json",dict(admitted=accepted,rejected=[t for t in tasks if t not in accepted],
                   passed=all(any(t["stage"]==stage for t in accepted) for stage in ("A","B","C"))))


if __name__=="__main__":main()
