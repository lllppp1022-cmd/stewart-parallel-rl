"""Montages from the original live RGB captures and matching physical rows."""
import _bootstrap
import argparse
import hashlib
import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.model import ROOT, sha, write_json
from src.workspace.attitude_study import RUN

SOURCES={
    "attitude_motion": [f"tracking/C4/{name}" for name in ("R1","R2","R3")],
    "ppo_training": [f"training/seed42/episode_{i:04d}" for i in (0,12,25)],
    "ppo_evaluation": [f"evaluation/E_mix/{name}/episode_0000" for name in ("baseline","classical50","PPO42")],
}
TITLES={"attitude_motion":"ACTIVE ROLL / PITCH TRACKING | LIVE PHYSICS",
        "ppo_training":"ACTUAL STOCHASTIC ATTITUDE PPO TRAINING",
        "ppo_evaluation":"INDEPENDENT ATTITUDE DISTURBANCE EVALUATION"}


def render(kind):
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    big=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",31)
    out=ROOT/f"assets/videos/stewart_phase2c5_{kind}.mp4"
    proofs=[]
    count=0
    with imageio.get_writer(out,fps=25,codec="libx264",macro_block_size=8) as writer:
        for relative in SOURCES[kind]:
            folder=RUN/relative
            d=pd.read_csv(folder/"physics.csv.gz")
            states=np.load(folder/"integration_states.npz")["states"]
            frames=[json.loads(s) for s in (folder/"frames.jsonl").read_text().splitlines()]
            previous=None
            changes=[]
            decoded=0
            trace_columns=["rpy_0","rpy_ref_0","rpy_1","rpy_ref_1"]
            trace=d[trace_columns].to_numpy()
            scale=max(.001,float(abs(trace).max()))*1.08
            reward_columns=[f"cost_{k}" for k in ("orientation","omega","xyz","action","slew")]
            cumulative=-d[reward_columns].sum(axis=1).cumsum().to_numpy()/20 if kind=="ppo_training" else None
            with imageio.get_reader(folder/"actual_rgb.mp4") as reader:
                for j,rgb in enumerate(reader):
                    frame=frames[j]
                    i=frame["state_index"]
                    assert hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==frame["qpos_sha256"]
                    row=d.iloc[i]
                    assert abs(row.time_s-frame["time_s"])<1e-9
                    if previous is not None:
                        changes.append(float(abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float)
                    decoded+=1
                    canvas=Image.new("RGB",(1600,896),(11,20,32))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((18,12),TITLES[kind],font=big,fill="white")
                    draw.text((18,57),relative,font=font,fill="#81cfff")
                    canvas.paste(Image.fromarray(rgb).resize((1040,780)),(0,90))
                    y=95
                    lines=[f"Physics time {row.time_s:.3f} s | policy v{int(row.get('policy_version',-1))}",
                           f"Roll actual / ref {row.rpy_0:+.5f} / {row.rpy_ref_0:+.5f} deg",
                           f"Pitch actual / ref {row.rpy_1:+.5f} / {row.rpy_ref_1:+.5f} deg",
                           f"Yaw {row.rpy_2:+.5f} deg | SO3 error {row.orientation_error_deg:.5f} deg",
                           "Omega deg/s: "+" / ".join(f"{np.rad2deg(row[f'twist_actual_{k}']):+.4f}" for k in (3,4,5)),
                           f"External Mx/My: {row.external_world_3:+.3f} / {row.external_world_4:+.3f} Nm",
                           f"Requested RL Mx/My: {row.get('requested_mx_Nm',0):+.5f} / {row.get('requested_my_Nm',0):+.5f}",
                           f"Accepted cmd Mx/My: {row.get('accepted_mx_Nm',0):+.5f} / {row.get('accepted_my_Nm',0):+.5f}",
                           f"Physical reward so far: {cumulative[i]:+.5f}" if cumulative is not None else
                           f"XYZ error: {row.position_error_mm:.5f} mm",
                           "Six slide strokes / measured actuator forces"]
                    lines += [f"Leg {k+1}: {row[f'q_{k}']*1000:+.3f} mm / {row[f'force_{k}']:+.3f} N" for k in range(6)]
                    lines += [f"Protection: {row.failure if isinstance(row.failure,str) else 'none'}",
                              "Physical envelope Roll/Pitch +/-0.5 deg",
                              "Partial geometry coverage | original render orientation"]
                    for line in lines:
                        draw.text((1050,y),line,font=font,fill="white")
                        y+=29
                    draw.text((1050,669),f"Roll/Pitch actual & reference | ±{scale:.4f} deg",font=font,fill="white")
                    draw.rectangle((1050,704,1585,867),outline="#6c879f")
                    selected=np.unique(np.r_[np.arange(0,i+1,30),i])
                    xs=1050+d.time_s.to_numpy()[selected]*535/6
                    for k,color in enumerate(("#51d9aa","#226a53","#ffd172","#85682d")):
                        ys=785-trace[selected,k]*76/scale
                        if len(xs)>1:
                            draw.line(list(zip(xs,ys)),fill=color,width=2)
                    writer.append_data(np.asarray(canvas))
                    count+=1
            assert decoded==len(frames) and max(changes)>0
            proofs.append(dict(source=relative,frames=decoded,frame_change_peak=max(changes),
                               source_sha256=sha(folder/"actual_rgb.mp4"),state_and_time_verified=True))
    with imageio.get_reader(out) as reader:
        assert sum(1 for _ in reader)==count
    write_json(RUN/f"video_{kind}.json",dict(passed=True,frames=count,seconds=count/25,
               sha256=sha(out),proof=proofs))
    print(out,count)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=list(SOURCES))
    render(p.parse_args().kind)
