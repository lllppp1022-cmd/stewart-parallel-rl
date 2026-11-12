"""Synchronized actual training RGB and frozen-policy dynamics, no resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from phase2c9b_protocol import ROOT,RUN,cfg,read,write_json,sha
from render_phase2c7v_training_analysis import chart,FONT,BIG


def sources(kind):
    if kind=='unknown_mass':
        from evaluate_phase2c9b import trial
        for task in cfg('tasks')['development']:trial(task,'baseline','media',record=True)
        return [(read(p),False) for p in sorted((RUN/'media').rglob('metrics.json'))]
    if kind=='single_vs_history':
        return [(read(p),False) for p in sorted((RUN/'final/Final_m21.5_6').rglob('metrics.json'))]
    episodes=[]
    for folder in sorted((RUN/'training/history_5frame/seed42').glob('from_*')):
        selected=(0,7) if folder.name=='from_000000' else (0,20)
        for episode in read(folder/'episodes.json'):
            if episode['episode'] in selected and (Path(episode['folder'])/'actual_rgb.mp4').exists():
                episodes.append((dict(folder=episode['folder'],method='history_5frame',seed=42,mass_kg=episode['task']['mass_kg'],task_id=episode['task']['id']),True))
    return episodes


def main(kind):
    target=ROOT/f'assets/videos/stewart_phase2c9b_{kind}.mp4'
    if target.exists():raise FileExistsError(target)
    clips=sources(kind);proofs=[];count=0
    with imageio.get_writer(target,fps=25,codec='libx264',quality=7,macro_block_size=8) as output:
        for metric,training in clips:
            folder=Path(metric['folder']);d=pd.read_csv(folder/'physics.csv.gz');t=d.time_s.to_numpy()
            frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
            states=np.load(folder/'integration_states.npz')['states'];summary=read(folder/'summary.json')
            assert not summary['recording_errors'] and frames
            sample=pd.read_csv(folder.parent/'training_samples.csv');episode=int(folder.name.split('_')[-1]);sample=sample[sample.episode==episode].set_index('decision')
            panels=[('Roll ref/actual (deg)',d[['rpy_ref_0','rpy_0']].to_numpy()),
                ('Pitch ref/actual (deg)',d[['rpy_ref_1','rpy_1']].to_numpy()),('SO3 error (deg)',d.orientation_error_deg.to_numpy()),
                ('Angular velocity xyz (rad/s)',d[[f'twist_actual_{j}' for j in (3,4,5)]].to_numpy()),
                ('External Mx/My (Nm)',d[['external_world_3','external_world_4']].to_numpy()),
                ('Z ref/actual (mm)',d[['reference_z_from_home_m','actual_z_from_home_m']].to_numpy()*1000),
                ('Requested / marginal accepted Mx/My (Nm)',d[['requested_mx_Nm','requested_my_Nm','accepted_mx_Nm','accepted_my_Nm']].to_numpy()),
                ('Slide displacement (mm)',(d[[f'q_{j}' for j in range(6)]].to_numpy()-d[[f'q_{j}' for j in range(6)]].iloc[0].to_numpy())*1000),
                ('Actual six actuator forces (N)',d[[f'force_{j}' for j in range(6)]].to_numpy()),
                ('V2 costs: orientation / omega / XYZ / action / slew',d[[f'dyn_cost_{j}' for j in ('orientation','omega','xyz','action','slew')]].to_numpy())]
            previous=None;change=0.
            with imageio.get_reader(folder/'actual_rgb.mp4') as source:
                for j,rgb in enumerate(source):
                    frame=frames[j];i=frame['state_index'];r=d.iloc[i]
                    assert abs(t[i]-frame['time_s'])<1e-9
                    assert hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==frame['qpos_sha256']
                    if previous is not None:change=max(change,float(abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float);canvas=Image.new('RGB',(1920,1080),'#0c141f');draw=ImageDraw.Draw(canvas)
                    draw.text((15,7),'ACTUAL STOCHASTIC TRAINING | POST-PROCESSED' if training else 'FROZEN DETERMINISTIC EVALUATION | ACTUAL DYNAMICS',font=BIG,fill='white')
                    draw.text((15,45),f"{metric['method']} | seed{metric['seed']} | {metric['task_id']}",font=FONT,fill='#7fd0ff')
                    canvas.paste(Image.fromarray(rgb).resize((600,450)),(12,82));draw=ImageDraw.Draw(canvas)
                    completed_decision=int(r.decision)-(0 if (i+1)%20==0 else 1)
                    completed=sample[sample.index<=completed_decision];last_reward=float(completed.reward.iloc[-1]) if len(completed) else 0.
                    lines=[f"Actual plant {metric['mass_kg']:g} kg",'Nominal controller 20 kg',f't={r.time_s:.3f}s',
                        f'Decision {int(r.global_decision)}',f'Actor version {int(r.policy_version)}',f'SO3 {r.orientation_error_deg:.6f} deg',
                        f'XYZ {r.position_error_mm:.5f} mm',f'Latest reward {last_reward:.5f}',
                        f'Episode return so far {completed.reward.sum():.4f}', 'TRUE PHYSICAL SCALE']
                    for k,line in enumerate(lines):draw.text((625,90+34*k),line,font=FONT,fill='white')
                    for k,(name,values) in enumerate(panels):chart(draw,(978,62+98*k,925,91),name,t,values,i)
                    if training:
                        for k,key in enumerate(('reward_orientation','reward_omega','reward_xyz','reward_action','reward_slew')):
                            draw.text((22,585+k*42),f'{key}: {completed[key].sum():.5f}',font=FONT,fill='white')
                        draw.text((22,840),'Captured during stochastic PPO sampling; changing tasks/masses.',font=FONT,fill='#ffca77')
                        if len(sample):chart(draw,(20,890,910,150),'Episode cumulative raw reward',sample.index.to_numpy()/50,sample.reward.cumsum().to_numpy(),max(0,len(completed)-1))
                    else:
                        for k,key in enumerate(('load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','xyz_rmse_mm','force_rate_N_s','success')):
                            draw.text((22,585+k*45),f'{key}: {metric[key]}',font=FONT,fill='white')
                        draw.text((22,920),'Full-episode metrics shown after offline evaluation.',font=FONT,fill='#ffca77')
                    output.append_data(np.asarray(canvas));count+=1
            assert j+1==len(frames) and change>0
            proofs.append(dict(source=str(folder),source_sha=sha(folder/'actual_rgb.mp4'),frames=len(frames),
                state_time_sha_match=True,pixel_change=change,qpos_range=float(np.ptp(states[:,1:44],axis=0).max()),training=training))
            print('Rendered',kind,metric['method'],metric['seed'],flush=True)
    with imageio.get_reader(target) as movie:assert movie.count_frames()==count
    write_json(RUN/f'video_{kind}.json',dict(path=str(target),sha=sha(target),frames=count,seconds=count/25,clips=proofs,no_policy_resampling=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['unknown_mass','history_training','single_vs_history']);main(p.parse_args().kind)
