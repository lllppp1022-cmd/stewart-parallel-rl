"""Actual captured RGB plus synchronized physical/training records; no policy sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from phase2c8_protocol import ROOT,RUN,read,write_json,sha
from render_phase2c7v_training_analysis import chart,FONT,BIG

def sources(kind):
    if kind=='training':
        result=[]
        for method in ('v2_control','v3','v3_return_norm'):
            for stage in ('A','B','C'):
                folder=RUN/'training'/method/'seed42'/stage
                if not (folder/'episodes.json').exists():continue
                eps=read(folder/'episodes.json')
                for ep in eps:
                    p=Path(ep['folder'])
                    if (p/'actual_rgb.mp4').exists() and (ep['episode']==0 or (stage=='C' and ep['episode']==12)):result.append((f'{method} | seed42 | {stage} | episode {ep["episode"]}',p,method,None))
        return result
    task='T8_mix' if kind=='reward_comparison' else 'T8_combined50';result=[]
    for p in (RUN/'final'/task).rglob('metrics.json'):
        m=read(p)
        if kind=='attitude_stability' and m['method']=='v2_control':continue
        folder=Path(m['folder']);assert (folder/'actual_rgb.mp4').exists();result.append((f'{m["method"]} | seed{m["seed"]} | {task}',folder,m['method'],m))
    return sorted(result,key=lambda x:x[0])

def history(draw,e,n,method,budget):
    g=e[(e.method==method)&(e.seed==42)&e.complete];x,y,w,h=18,690,910,280;lo=float(g.return_raw.min());hi=float(g.return_raw.max());span=max(hi-lo,1);lo-=span*.1;hi+=span*.1
    px=lambda value:x+w*value/max(budget,1);py=lambda value:y+h-(value-lo)/(hi-lo)*h
    draw.text((18,555),'RAW TRAINING RETURN | '+method,font=BIG,fill='white')
    draw.text((18,590),'Different V2/V3 reward formulas: these returns do NOT rank controllers.',font=FONT,fill='#ffca77')
    draw.text((18,620),'Blue = completed; gray = future; vertical marker = actual sampling step.',font=FONT,fill='white');draw.rectangle((x,y,x+w,y+h),outline='#4c6478')
    for stage in g.stage.unique():
        part=g[g.stage==stage];trend=part.return_raw.rolling(3,min_periods=1).mean()
        for k,r in enumerate(part.itertuples()):
            color='#49bfff' if r.end_decision<=n else '#656d75';xx=px(r.end_decision);yy=py(r.return_raw);draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
            if k:draw.line((px(part.iloc[k-1].end_decision),py(trend.iloc[k-1]),xx,py(trend.iloc[k])),fill=color,width=2)
    for at in (4096,8192):
        if at<budget:draw.line((px(at),y,px(at),y+h),fill='#8f969c')
    draw.line((px(n),y,px(n),y+h),fill='#ffcb44',width=2)
    draw.text((18,990),f'Actual decisions {n}/{budget}; return range [{lo:.3g},{hi:.3g}]',font=FONT,fill='white')
    draw.text((18,1020),'Only complete episodes plotted; budget fragments remain in CSV.',font=FONT,fill='#9cabb9')

def render(kind):
    training=kind=='training';ep=pd.read_csv(ROOT/'results/phase2c8_episode_returns.csv');updates=pd.read_csv(ROOT/'results/phase2c8_critic_diagnostics.csv');selection=read(RUN/'final_selection.json');budget=selection['budget'];target=ROOT/f'assets/videos/stewart_phase2c8_{kind}.mp4';count=0;proof=[]
    clips=sources(kind);assert clips
    with imageio.get_writer(target,fps=25,codec='libx264',quality=7,macro_block_size=8) as writer:
        for label,folder,method,metrics in clips:
            d=pd.read_csv(folder/'physics.csv.gz');t=d.time_s.to_numpy();frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()];states=np.load(folder/'integration_states.npz')['states'];summary=read(folder/'summary.json');assert not summary['recording_errors']
            assert ('TRAINING' if training else 'EVALUATION') in summary['source']
            sample=pd.read_csv(folder.parent/'training_samples.csv');sample=sample[sample.episode==int(folder.name.split('_')[-1])].set_index('decision')
            charts=[('Roll ref/actual (deg)',d[['rpy_ref_0','rpy_0']].to_numpy()),('Pitch ref/actual (deg)',d[['rpy_ref_1','rpy_1']].to_numpy()),('SO3 error (deg)',d.orientation_error_deg.to_numpy()),('Angular velocity xyz (rad/s)',d[[f'twist_actual_{k}' for k in (3,4,5)]].to_numpy()),('External Mx/My (Nm)',d[['external_world_3','external_world_4']].to_numpy()),('Z ref/actual (mm)',d[['reference_z_from_home_m','actual_z_from_home_m']].to_numpy()*1000),('Requested / marginal accepted Mx/My (Nm)',d[['requested_mx_Nm','requested_my_Nm','accepted_mx_Nm','accepted_my_Nm']].to_numpy()),('Six slide travel from initial (mm)',(d[[f'q_{k}' for k in range(6)]].to_numpy()-d[[f'q_{k}' for k in range(6)]].iloc[0].to_numpy())*1000),('Six actual actuator forces (N)',d[[f'force_{k}' for k in range(6)]].to_numpy()),('Physical costs: SO3 / omega / XYZ / action / slew',d[[f'dyn_cost_{k}' for k in ('orientation','omega','xyz','action','slew')]].to_numpy())]
            previous=None;change=0;n=0
            with imageio.get_reader(folder/'actual_rgb.mp4') as reader:
                for j,rgb in enumerate(reader):
                    f=frames[j];i=f['state_index'];r=d.iloc[i];assert abs(t[i]-f['time_s'])<1e-9 and hashlib.sha256(states[i,1:44].tobytes()).hexdigest()==f['qpos_sha256']
                    if previous is not None:change=max(change,float(abs(rgb.astype(float)-previous).mean()))
                    previous=rgb.astype(float);canvas=Image.new('RGB',(1920,1080),'#0c141f');draw=ImageDraw.Draw(canvas)
                    draw.text((16,7),'ACTUAL STOCHASTIC TRAINING | POST-PROCESSED' if training else 'FROZEN DETERMINISTIC EVALUATION | ACTUAL DYNAMICS',font=BIG,fill='white');draw.text((16,43),label,font=FONT,fill='#7fd0ff');canvas.paste(Image.fromarray(rgb).resize((600,450)),(12,82));draw=ImageDraw.Draw(canvas)
                    dec=int(r.decision)-(1 if (i+1)%20 else 0);reward=float(sample.loc[dec,'reward']) if dec in sample.index else 0.
                    u=updates[(updates.method==method)&(updates.seed==42)&(updates.decisions<=int(r.policy_version)*512)] if training else pd.DataFrame()
                    lines=[f't = {t[i]:.3f} s',f'Stage {r.stage}',f'Global decision {int(r.global_decision)}',f'Policy version {int(r.policy_version)}',f'Checkpoint {int(r.policy_version)*512} steps',f'Z {r.actual_z_from_home_m*1000:.3f} mm',f'Roll {r.rpy_0:+.4f} deg',f'Pitch {r.rpy_1:+.4f} deg',f'Mx {r.external_world_3:+.3f} Nm',f'My {r.external_world_4:+.3f} Nm',f'Last raw reward {reward:+.5f}',f'Value loss {u.iloc[-1].value_loss:.4g}' if len(u) else 'Value loss: no prior update' if training else 'No optimizer updates',f'EV {u.iloc[-1].explained_variance:.4g}' if len(u) else '', 'TRUE PHYSICAL SCALE']
                    for k,line in enumerate(lines):draw.text((625,90+29*k),line,font=FONT,fill='white')
                    for k,(name,values) in enumerate(charts):chart(draw,(978,62+98*k,925,91),name,t,values,i)
                    draw.text((978,1050),'Blue/orange/green/red/purple/cyan: listed order. 1ms physics; 50Hz policy.',font=FONT,fill='#9cabb9')
                    if training:history(draw,ep,int(r.global_decision),method,budget)
                    else:
                        draw.text((18,562),'Full physical episode metrics (post-processed)',font=BIG,fill='white')
                        keys=['load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','xyz_rmse_mm','force_rate_N_s','common_v2_return','success']
                        for k,key in enumerate(keys):draw.text((22,620+k*40),f'{key}: {metrics[key]}',font=FONT,fill='white')
                        draw.text((22,980),'Same task / reference / load / initial state / controller budgets.',font=FONT,fill='#7fd0ff');draw.text((22,1015),'Single-trial video is not evidence of overall superiority.',font=FONT,fill='#ffca77')
                    writer.append_data(np.asarray(canvas));count+=1;n+=1
            assert n==len(frames) and change>0;proof.append(dict(source=str(folder),source_sha=sha(folder/'actual_rgb.mp4'),source_label=summary['source'],frames=n,timestamp_state_hash_match=True,pixel_change=change,qpos_range=float(np.ptp(states[:,1:44],axis=0).max()),model_versions=sorted(d.policy_version.unique().astype(int).tolist())))
            print('Rendered',kind,label,n,flush=True)
    with imageio.get_reader(target) as reader:assert sum(1 for _ in reader)==count
    write_json(RUN/f'video_{kind}.json',dict(path=str(target),sha=sha(target),frames=count,seconds=count/25,resolution=[1920,1080],fps=25,clips=proof,training=training,no_policy_resampling=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['training','reward_comparison','attitude_stability']);render(p.parse_args().kind)
