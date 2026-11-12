"""Post-process genuine RGB and state hashes; never drives or resamples a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from phase2c7v_common import ROOT,RUN,OLD,NAMES,read,write_json,sha
from render_phase2c7_videos import sources,COLORS

FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
BIG=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
def chart(draw,rect,title,t,values,end,current=None):
    x,y,w,h=rect;values=np.asarray(values)
    if values.ndim==1:values=values[:,None]
    low=float(values.min());high=float(values.max());span=max(high-low,1e-7);low-=span*.08;high+=span*.08
    draw.text((x,y),title+f'  [{low:.3g}, {high:.3g}]',font=FONT,fill='white');draw.rectangle((x,y+23,x+w,y+h),outline='#4c6478')
    take=np.unique(np.r_[np.arange(0,end+1,max(1,(end+1)//350)),end])
    if len(take)>1:
        xx=x+t[take]/max(t[-1],1e-9)*w
        for col in range(values.shape[1]):
            yy=y+h-3-(values[take,col]-low)/(high-low)*(h-29)
            draw.line(list(zip(xx,yy)),fill=COLORS[col%6],width=2)
    if current is not None:draw.text((x+4,y+h-20),current,font=FONT,fill='white')

def reward_chart(draw,episodes,n,seed=42):
    x,y,w,h=20,630,910,370;e=episodes[(episodes.seed==seed)&episodes.complete];low=float(e.episode_return.min());high=float(e.episode_return.max());span=max(high-low,1);low-=span*.1;high+=span*.1
    px=lambda a:x+a/32768*w;py=lambda a:y+h-(a-low)/(high-low)*h
    draw.text((20,553),f'Complete episode return | seed {seed} | full recorded history',font=BIG,fill='white')
    draw.text((20,587),'Blue: already completed   Gray: future data   x: partial fragment (not comparable)',font=FONT,fill='white')
    draw.rectangle((x,y,x+w,y+h),outline='#4c6478')
    for k in (4096,8192):draw.line((px(k),y,px(k),y+h),fill='#788694')
    for label,k in [('A',2000),('B',6000),('C',20000)]:draw.text((px(k),y+4),label,font=BIG,fill='#c0ccd5')
    for r in e.itertuples():
        xx=px(r.end_decision);yy=py(r.episode_return);color='#4fc3ff' if r.end_decision<=n else '#626a70';draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
    for stage in ('A','B','C'):
        a=e[e.stage==stage];trend=a.episode_return.rolling(3,min_periods=1).mean()
        for k in range(1,len(a)):
            color='#4fc3ff' if a.iloc[k].end_decision<=n else '#626a70';draw.line((px(a.iloc[k-1].end_decision),py(trend.iloc[k-1]),px(a.iloc[k].end_decision),py(trend.iloc[k])),fill=color,width=2)
    for r in episodes[(episodes.seed==seed)&~episodes.complete].itertuples():
        xx=px(r.end_decision);yy=np.clip(py(r.episode_return),y,y+h);draw.line((xx-4,yy-4,xx+4,yy+4),fill='#bb7272');draw.line((xx-4,yy+4,xx+4,yy-4),fill='#bb7272')
    draw.line((px(n),y,px(n),y+h),fill='#ffca44',width=2)
    draw.text((20,1010),f'Actual PPO decisions: {n} / 32768 | return range {low:.2f} to {high:.2f}',font=FONT,fill='white')
    draw.text((20,1036),'Different curriculum tasks / episode lengths: not a fixed-task learning curve.',font=FONT,fill='#9cabb9')

def render(training=True):
    episodes=pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv');proof=[];count=0
    target=ROOT/('assets/videos/stewart_phase2c7_training_analysis.mp4' if training else 'assets/videos/stewart_phase2c7_policy_comparison.mp4')
    clips=sources('dynamic_ppo_training') if training else [(m,RUN/'comparison/F_roll'/m/'final/episode_0000') for m in ('classical50','seed42','seed123','seed2026')]
    with imageio.get_writer(target,fps=25,codec='libx264',quality=7,macro_block_size=8) as writer:
        for label,folder in clips:
            d=pd.read_csv(folder/'physics.csv.gz');t=d.time_s.to_numpy();frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
            with np.load(folder/'integration_states.npz') as state_file:states=state_file['states']
            meta=read(folder/'summary.json');assert not meta['recording_errors']
            if training:assert 'TRAINING' in meta['source']
            else:assert 'EVALUATION' in meta['source']
            sample=pd.read_csv(folder.parent/'training_samples.csv');sample=sample[sample.episode==int(folder.name.split('_')[-1])].set_index('decision')
            err=(Rotation.from_euler('xyz',d[['rpy_0','rpy_1','rpy_2']].to_numpy(),degrees=True)*Rotation.from_euler('xyz',d[['rpy_ref_0','rpy_ref_1','rpy_ref_2']].to_numpy(),degrees=True).inv()).as_rotvec()
            omega=d[[f'twist_actual_{k}' for k in (3,4,5)]].to_numpy()-d[[f'twist_ref_{k}' for k in (3,4,5)]].to_numpy()
            pd_same=np.clip(-25*err[:,:2]-omega[:,:2],-.05,.05)
            top=[('Roll ref / actual (deg)',d[['rpy_ref_0','rpy_0']].to_numpy()),('Pitch ref / actual (deg)',d[['rpy_ref_1','rpy_1']].to_numpy()),('SO3 error (deg)',d.orientation_error_deg.to_numpy()),('Angular velocity xyz (rad/s)',d[[f'twist_actual_{k}' for k in (3,4,5)]].to_numpy()),('External Mx / My (Nm)',d[['external_world_3','external_world_4']].to_numpy()),('Z ref / actual (mm)',d[['reference_z_from_home_m','actual_z_from_home_m']].to_numpy()*1000)]
            bottom=[('Mx request / accepted / same-state PD* (Nm)',np.column_stack([d.requested_mx_Nm,d.accepted_mx_Nm,pd_same[:,0]])),('My request / accepted / same-state PD* (Nm)',np.column_stack([d.requested_my_Nm,d.accepted_my_Nm,pd_same[:,1]])),('Six actual actuator forces (N)',d[[f'force_{k}' for k in range(6)]].to_numpy()),('Physical cost: orientation / total (per 1ms cost)',np.column_stack([d.dyn_cost_orientation,d[[f'dyn_cost_{k}' for k in NAMES]].sum(axis=1)]))]
            result=None if training else read(folder.parent/'result.json');previous=None;change=0;n=0
            with imageio.get_reader(folder/'actual_rgb.mp4') as reader:
                for j,rgb in enumerate(reader):
                    f=frames[j];i=f['state_index'];r=d.iloc[i];assert abs(t[i]-f['time_s'])<1e-9
                    assert hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==f['qpos_sha256']
                    if previous is not None:change=max(change,float(np.abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float);canvas=Image.new('RGB',(1920,1080),'#0c141f');draw=ImageDraw.Draw(canvas)
                    draw.text((16,7),'POST-PROCESSED TRAINING VISUALIZATION' if training else 'FROZEN DETERMINISTIC POLICY EVALUATION | SAME F_roll TASK',font=BIG,fill='white')
                    draw.text((16,42),'ACTUAL STOCHASTIC TRAINING | '+label if training else label+' | REAL LIVE RGB, data.ctrl + mj_step',font=FONT,fill='#7fd0ff')
                    canvas.paste(Image.fromarray(rgb).resize((600,450)),(12,82));draw=ImageDraw.Draw(canvas)
                    completed=int(r.decision)-(1 if (i+1)%20 else 0);reward=float(sample.loc[completed,'reward']) if completed in sample.index else 0.
                    lines=[f'Physical time {t[i]:.3f} s',f'Global decision {int(r.global_decision)}',f'Policy update {int(r.policy_version)}',f'Current checkpoint:',f'{int(r.policy_version)*512} decisions' if training else '32768 decisions' if label!='classical50' else 'Classical 50 Hz',f'Stage {r.stage}',f'Z actual {r.actual_z_from_home_m*1000:.3f} mm',f'Z ref {r.reference_z_from_home_m*1000:.3f} mm',f'Roll {r.rpy_0:+.4f} deg',f'Pitch {r.rpy_1:+.4f} deg',f'Mx {r.external_world_3:+.3f} Nm',f'My {r.external_world_4:+.3f} Nm',f'Last decision reward {reward:+.5f}','TRUE GEOMETRIC SCALE']
                    for k,line in enumerate(lines):draw.text((625,90+29*k),line,font=FONT,fill='white')
                    for k,(title,values) in enumerate(top):chart(draw,(978,62+79*k,925,72),title,t,values,i)
                    for k,(title,values) in enumerate(bottom):chart(draw,(978,550+117*k,925,110),title,t,values,i)
                    draw.text((978,1027),'* Same-state PD diagnostic, NOT executed on this trajectory.',font=FONT,fill='#c5a971')
                    draw.text((978,1050),'Blue/orange/green/red/purple/cyan: listed order; Nm and N.',font=FONT,fill='#9cabb9')
                    if training:reward_chart(draw,episodes,int(r.global_decision))
                    else:
                        draw.text((20,570),'Full-episode metrics (post-processed)',font=BIG,fill='white')
                        keys=['evaluation_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','z_rmse_mm','force_rate_N_s','success']
                        for k,key in enumerate(keys):draw.text((25,620+40*k),f'{key}: {result[key]}',font=BIG if key=='success' else FONT,fill='white')
                        draw.text((25,983),'All four controllers: identical reference, load, initial state and budgets.',font=FONT,fill='#7fd0ff')
                        draw.text((25,1015),'One task illustrates loading / unloading tradeoffs; not overall success rate.',font=FONT,fill='#c5a971')
                    writer.append_data(np.asarray(canvas));count+=1;n+=1
            assert n==len(frames) and change>0
            proof.append(dict(source=str(folder),source_sha=sha(folder/'actual_rgb.mp4'),frames=n,state_timestamp_hashes_match=True,peak_rgb_change=change,qpos_range=float(np.ptp(states[:,1:44],axis=0).max()),z_range_mm=float(np.ptp(d.actual_z_from_home_m)*1000),source_label=meta['source']))
            print('Rendered',label,n,flush=True)
    with imageio.get_reader(target) as reader:assert sum(1 for _ in reader)==count
    write_json(RUN/('training_video_validation.json' if training else 'comparison_video_validation.json'),dict(path=str(target),sha=sha(target),resolution=[1920,1080],fps=25,frames=count,seconds=count/25,clips=proof,no_resampling=True,future_history_gray=training))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',action='store_true');render(not p.parse_args().comparison)
