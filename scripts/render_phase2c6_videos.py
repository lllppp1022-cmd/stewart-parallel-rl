"""Build annotated videos exclusively from archived live RGB frames."""
import argparse
import hashlib
import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from phase2c6_protocol import OLD, ROOT, RUN, read, sha, write_json


def sources(kind):
    if kind=="reward_ablation":
        base=OLD/"evaluation/E_mix"
        return [(label,base/m/"episode_0000") for label,m in (
            ("fourDlead","baseline"),("Classical 50 Hz","classical50"),("PPO-V1 seed42","PPO42"))]+[
            ("PPO-V2 seed42",RUN/"dev/E_mix/V2_42/episode_0000")]
    if kind=="attitude_evaluation":
        methods=(("fourDlead","baseline"),("Classical 50 Hz","classical50"),("PPO-V2 seed42","V2_42"),
                 ("Curriculum-V2 seed42","C2_42"))
        cases=(("Unseen combined load","F_mix"),("2.4 Nm regression","E_strong_regression"))
        return [(f"{label} | {case}",RUN/"final"/task/m/"episode_0000") for case,task in cases
                for label,m in methods if (RUN/"final"/task/m/"comparison.json").is_file()]
    group="training" if kind=="actual_v2_training" else "curriculum_training"
    root=RUN/group/"seed42"
    episodes=read(root/"episodes.json")
    return [(f"Episode {ep['episode']} | {ep['task']['id']} | load {np.linalg.norm(ep['task']['torque']):.2f} Nm",
             root/f"episode_{ep['episode']:04d}")
            for ep in episodes if (root/f"episode_{ep['episode']:04d}/actual_rgb.mp4").exists()]


def render(kind):
    training=kind in ("actual_v2_training","curriculum_training")
    title={"reward_ablation":"REWARD ABLATION | DETERMINISTIC EVALUATION",
           "attitude_evaluation":"UNSEEN TEST AND 2.4 Nm REGRESSION | DETERMINISTIC EVALUATION",
           "actual_v2_training":"ACTUAL STOCHASTIC PPO | FIXED REWARD-V2 ABLATION",
           "curriculum_training":"ACTUAL STOCHASTIC PPO | REWARD-V2 CURRICULUM"}[kind]
    regular=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    large=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",30)
    path=ROOT/f"assets/videos/stewart_phase2c6_{kind}.mp4"
    clips=sources(kind)
    assert clips and all((folder/"actual_rgb.mp4").exists() for _,folder in clips)
    proofs=[]
    n=0
    with imageio.get_writer(path,fps=25,codec="libx264",macro_block_size=8) as writer:
        for label,folder in clips:
            d=pd.read_csv(folder/"physics.csv.gz")
            states=np.load(folder/"integration_states.npz")["states"]
            frames=[json.loads(line) for line in (folder/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
            if training:
                samples=pd.read_csv(folder.parent/"training_samples.csv")
                samples=samples[samples.episode==int(folder.name.split("_")[-1])].set_index("decision")
            metrics=read(folder.parent/"comparison.json") if (folder.parent/"comparison.json").exists() else None
            if metrics is None and not training:
                metrics=read(folder/"metrics.json")
            summary=read(folder/"summary.json")
            assert not summary["recording_errors"]
            if training:
                assert summary["source"]=="ACTUAL STOCHASTIC ATTITUDE PPO TRAINING"
            time=d.time_s.to_numpy()
            traces=d[["rpy_0","rpy_1"]].to_numpy()
            scale=max(.0001,float(abs(traces).max()))*1.10
            count,change=0,0.
            previous=None
            with imageio.get_reader(folder/"actual_rgb.mp4") as reader:
                for j, rgb in enumerate(reader):
                    frame=frames[j]
                    i=frame["state_index"]
                    assert abs(time[i]-frame["time_s"])<1e-9
                    assert hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==frame["qpos_sha256"]
                    row=d.iloc[i]
                    if previous is not None:
                        change=max(change,float(np.abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float)
                    canvas=Image.new("RGB",(1600,896),(12,20,31))
                    canvas.paste(Image.fromarray(rgb).resize((1000,750)),(0,92))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((20,10),title,font=large,fill="white")
                    draw.text((20,55),label,font=regular,fill="#7fd0ff")
                    lines=[f"Physical time {row.time_s:.3f} s | decision {int(row.decision)}",
                           f"Roll/Pitch actual: {row.rpy_0:+.6f} / {row.rpy_1:+.6f} deg",
                           f"SO(3) error {row.orientation_error_deg:.6f} deg | XYZ {row.position_error_mm:.5f} mm",
                           f"Omega roll/pitch {row.twist_actual_3:+.6f} / {row.twist_actual_4:+.6f} rad/s",
                           f"External Mx/My {row.external_world_3:+.3f} / {row.external_world_4:+.3f} Nm",
                           f"Requested Mx/My {row.requested_mx_Nm:+.5f} / {row.requested_my_Nm:+.5f} Nm",
                           f"Accepted Mx/My {row.accepted_mx_Nm:+.5f} / {row.accepted_my_Nm:+.5f} Nm",
                           f"Protection: {row.failure if isinstance(row.failure,str) else 'none'}"]
                    if training:
                        s=samples.loc[row.decision]
                        last=int(row.decision)-(0 if (i+1)%20==0 else 1)
                        r=float(samples.loc[last,"reward"]) if last in samples.index else 0.
                        lines += [f"Policy v{int(row.policy_version)} | global {int(row.global_decision)}",
                                  f"Gaussian draw {s.raw_roll:+.4f} / {s.raw_pitch:+.4f}",
                                  f"Last completed decision reward {r:+.6f}"]
                    else:
                        lines += [f"Peak SO(3) {metrics['so3_peak_deg']:.6f} deg | XYZ {metrics['xyz_peak_mm']:.5f} mm",
                                  f"Load/unload IAE {metrics['load_iae_deg_s']:.6f} / {metrics['unload_iae_deg_s']:.6f} deg s",
                                  f"Peak angular speed {metrics['omega_peak_deg_s']:.6f} deg/s"]
                    lines += ["Actual six leg forces (N)"]+[
                        " / ".join(f"L{k+1} {row[f'force_{k}']:+.3f}" for k in group)
                        for group in ((0,1,2),(3,4,5))]
                    for k,line in enumerate(lines):
                        draw.text((1014,102+k*29),line,font=regular,fill="white")
                    draw.text((1014,550),f"Angle chart: ±{scale:.6f} deg | geometry at true scale",font=regular,fill="#7fd0ff")
                    draw.rectangle((1014,586,1580,850),outline="#6c879f")
                    take=np.unique(np.r_[np.arange(0,i+1,20),i])
                    if len(take)>1:
                        xs=1014+time[take]*566/6
                        for k,color in enumerate(("#4fe0ae","#ffd172")):
                            draw.line(list(zip(xs,718-traces[take,k]*124/scale)),fill=color,width=2)
                    writer.append_data(np.asarray(canvas))
                    n+=1
                    count+=1
            assert count==len(frames) and change>0 and np.ptp(traces,axis=0).max()>0
            proofs.append(dict(label=label,source=str(folder),frames=count,source_sha256=sha(folder/"actual_rgb.mp4"),
                               frame_change_peak=change,angular_range_deg=np.ptp(traces,axis=0).tolist(),
                               all_frame_state_hashes_and_timestamps_match=True))
    with imageio.get_reader(path) as reader:
        assert sum(1 for _ in reader)==n
    write_json(RUN/f"video_{kind}.json",dict(path=str(path),sha256=sha(path),decoded_frames=n,seconds=n/25,
                                           verified=True,clips=proofs))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=["reward_ablation","attitude_evaluation","actual_v2_training","curriculum_training"])
    render(p.parse_args().kind)
