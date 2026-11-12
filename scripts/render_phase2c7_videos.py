"""Annotated live-RGB evidence; never resample a policy or teleport a renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from phase2c7_protocol import ROOT,RUN,read,write_json
from src.model import sha
from src.visualization.enhanced.style import LEG_COLORS

COLORS=list(LEG_COLORS)


def series(d):
    cols=lambda names:d[names].to_numpy()
    q=cols([f'q_{j}' for j in range(6)])
    return [
        ('Z ref / actual (mm)',cols(['reference_z_from_home_m','actual_z_from_home_m'])*1000),
        ('Roll ref / actual (deg)',cols(['rpy_ref_0','rpy_0'])),
        ('Pitch ref / actual (deg)',cols(['rpy_ref_1','rpy_1'])),
        ('SO(3) tracking error (deg)',cols(['orientation_error_deg'])),
        ('External Mx / My (Nm)',cols(['external_world_3','external_world_4'])),
        ('Mx/My requested / accepted (Nm)',cols(['requested_mx_Nm','requested_my_Nm','accepted_mx_Nm','accepted_my_Nm'])),
        ('Six relative slide strokes (mm)',(q-q[0])*1000),
        ('Six actual actuator forces (N)',cols([f'force_{j}' for j in range(6)]))]


def sources(kind):
    if kind=='dynamic_ppo_training':
        root=RUN/'training/seed42'
        return [(f'{phase} | episode {ep["episode"]} | {ep["task"]["id"]}',Path(ep['folder']))
                for phase in ('pilot','continue') if (root/phase/'episodes.json').exists()
                for ep in read(root/phase/'episodes.json') if (Path(ep['folder'])/'actual_rgb.mp4').exists()]
    if kind=='classical_vs_ppo':
        return [(method,RUN/'final/F_mix'/method/'episode_0000') for method in ('baseline','classical50','PPO_42')]
    return [('Frozen combined controller | C2',RUN/'preflight/C2/baseline/episode_0000')]


def render(kind):
    training=kind=='dynamic_ppo_training'
    title='ACTUAL STOCHASTIC PPO TRAINING' if training else 'DETERMINISTIC POLICY EVALUATION' if kind=='classical_vs_ppo' else 'DYNAMIC ACTUATOR SIMULATION | FROZEN CONTROLLER'
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29)
    target=ROOT/f'assets/videos/stewart_phase2c7_{kind}.mp4'
    proofs=[];count=0
    with imageio.get_writer(target,fps=25,codec='libx264',macro_block_size=8) as writer:
        for label,folder in sources(kind):
            d=pd.read_csv(folder/'physics.csv.gz')
            frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
            states=np.load(folder/'integration_states.npz')['states']
            summary=read(folder/'summary.json')
            assert not summary['recording_errors']
            if training:assert 'TRAINING' in summary['source']
            charts=series(d);times=d.time_s.to_numpy();previous=None;change=0.;n=0
            samples=pd.read_csv(folder.parent/'training_samples.csv')
            samples=samples[samples.episode==int(folder.name.split('_')[-1])].set_index('decision')
            with imageio.get_reader(folder/'actual_rgb.mp4') as reader:
                for j,rgb in enumerate(reader):
                    frame=frames[j];i=frame['state_index'];row=d.iloc[i]
                    assert abs(times[i]-frame['time_s'])<1e-9
                    assert hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==frame['qpos_sha256']
                    if previous is not None:change=max(change,float(abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float)
                    canvas=Image.new('RGB',(1920,1080),(12,20,31));draw=ImageDraw.Draw(canvas)
                    draw.text((18,8),title,font=big,fill='white')
                    draw.text((18,49),label+' | TRUE GEOMETRIC SCALE',font=font,fill='#7fd0ff')
                    canvas.paste(Image.fromarray(rgb).resize((1160,870)),(0,88))
                    draw=ImageDraw.Draw(canvas)
                    policy_label=f'policy v{int(row.policy_version)}' if row.policy_version>=0 else 'fixed controller'
                    info=f't={times[i]:.3f}s | decision {int(row.global_decision)} | {policy_label} | stage {row.stage}'
                    draw.text((18,974),info,font=font,fill='white')
                    completed=int(row.decision)-(1 if (i+1)%20 else 0)
                    reward=float(samples.loc[completed,'reward']) if completed in samples.index else 0.
                    status=row.failure if isinstance(row.failure,str) else 'RUNNING' if i<len(d)-1 else ('COMPLETE' if summary.get('complete') else 'BUDGET/INTERRUPTED')
                    draw.text((18,1007),f'Last completed reward {reward:+.6f} | {status}',font=font,fill='white')
                    draw.text((18,1040),'Force-controlled live RGB; 1 ms physics; 50 Hz decisions; curves use actual units.',font=font,fill='#7fd0ff')
                    take=np.unique(np.r_[np.arange(0,i+1,20),i])
                    for k,(name,values) in enumerate(charts):
                        y=70+k*125;lo=float(values.min());hi=float(values.max());span=max(hi-lo,1e-7)
                        lo-=span*.08;hi+=span*.08
                        draw.text((1180,y),name+f' [{lo:.4g}, {hi:.4g}]',font=font,fill='white')
                        draw.rectangle((1180,y+27,1895,y+113),outline='#506b84')
                        if len(take)>1:
                            xs=1180+times[take]/times[-1]*715
                            for c in range(values.shape[1]):
                                ys=y+110-(values[take,c]-lo)/(hi-lo)*80
                                draw.line(list(zip(xs,ys)),fill=COLORS[c%6],width=2)
                    writer.append_data(np.asarray(canvas));count+=1;n+=1
            assert n==len(frames) and change>0
            proofs.append(dict(source=str(folder),frames=n,source_sha=sha(folder/'actual_rgb.mp4'),
                matched_state_hashes=True,frame_change_peak=change,z_motion_mm=float(d.actual_z_from_home_m.max()*1000),
                roll_range_deg=float(np.ptp(d.rpy_0)),pitch_range_deg=float(np.ptp(d.rpy_1))))
    with imageio.get_reader(target) as reader:assert sum(1 for _ in reader)==count
    write_json(RUN/f'video_{kind}.json',dict(path=str(target),sha=sha(target),frames=count,seconds=count/25,clips=proofs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['dynamic_ppo_training','classical_vs_ppo','dynamic_disturbance_demo'])
    render(p.parse_args().kind)
