"""Compose only actual live RGB with causal curves and state/frame provenance."""
import _bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from src.disturbance.metrics import vec
from src.visualization.enhanced.style import LEG_COLORS

def compose(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    evaluation=json.loads((root/'evaluation_frozen.json').read_text(encoding='utf-8'));candidates=evaluation['candidates']
    if candidates:
        task='eval_'+candidates[0]['kind'];clips=[root/'frozen_evaluation'/task/m for m in ['C4','C4_fourD','C4_fixedLead']]
    else:clips=sorted((root/'visualization_only/zload').glob('*'))
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18);fig=Figure(figsize=(6.4,10.8),dpi=100);canvas=FigureCanvasAgg(fig);axes=fig.subplots(5,1);fig.subplots_adjust(left=.17,right=.98,top=.96,bottom=.05,hspace=.55);writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8);audit=[];sources=[];motion=[]
    try:
        for folder in clips:
            d=pd.read_csv(folder/'physics.csv.gz');c=json.loads((folder/'config.json').read_text(encoding='utf-8'));home=json.loads((folder/'home_baseline.json').read_text(encoding='utf-8'));frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];reader=imageio.get_reader(folder/'actual_rgb.mp4');z=np.load(folder/'integration_states.npz');tau=c['actuator_dynamics']['tau_s'];on=c['disturbance']['start_step']*.001;off=c['disturbance']['stop_step']*.001
            assert frames;states=z['states'][[f['state_index'] for f in frames]];span=np.ptp(states[:,1:44],axis=0);assert span.max()>1e-6;motion.append(dict(folder=str(folder),recorded_qpos_max_range=float(span.max())));sources.append(dict(folder=str(folder),sha256=sha(folder/'actual_rgb.mp4'),frames=len(frames)));source_unique=set()
            try:
                for f in frames:
                    i=f['state_index'];r=d.iloc[i];digest=hashlib.sha256(z['states'][i,1:44].tobytes()).hexdigest();assert digest==f['qpos_sha256'] and abs(r.time_s-f['time_s'])<1e-10;rgb=reader.get_data(f['frame']);source_unique.add(hashlib.sha256(rgb.tobytes()).hexdigest());past=d.iloc[:i+1:20]
                    for ax in axes:ax.clear();ax.grid(alpha=.2);ax.set_xlim(0,6);ax.tick_params(labelsize=7);ax.axvline(on,color='k',ls=':',lw=.7);ax.axvline(off,color='k',ls=':',lw=.7)
                    for j,color in enumerate(LEG_COLORS[:3]):axes[0].plot(past.time_s,(past[f'p_{j}']-past[f'p_ref_{j}'])*1000,color=color,label='XYZ'[j]);axes[1].plot(past.time_s,past[f'rpy_{j}'],color=color,label=['Roll','Pitch','Yaw'][j]);axes[1].plot(past.time_s,past[f'rpy_ref_{j}'],color=color,ls='--')
                    axes[0].set_ylabel('XYZ error mm');axes[0].legend(fontsize=7,ncol=3);axes[1].set_ylabel('Actual/ref deg');axes[1].legend(fontsize=7,ncol=3)
                    for j,color in enumerate(LEG_COLORS):
                        axes[2].plot(past.time_s,(past[f'q_{j}']-home['slide_m'][j])*1000,color=color,label=f'Leg{j+1}')
                        initial=d[f'F_drive_previous_{j}'].iloc[0];axes[3].plot(past.time_s,past[f'F_command_{j}']-initial,color=color,ls='--',lw=.7);axes[3].plot(past.time_s,past[f'F_drive_{j}']-initial,color=color,ls=':',lw=.8);axes[3].plot(past.time_s,past[f'force_{j}']-initial,color=color,lw=1.)
                    axes[2].set_ylabel('Slide delta mm');axes[2].legend(fontsize=6,ncol=3);axes[3].set_ylabel('Force delta N');axes[3].set_title('6 legs: command -- / drive : / actual solid',fontsize=8)
                    axes[4].plot(past.time_s,abs(vec(past,'F_add_total',6)).max(axis=1),label='Shared extra max');axes[4].plot(past.time_s,abs(vec(past,'F_command',6)-vec(past,'F_drive',6)).max(axis=1),label='|command-drive|');axes[4].set_ylabel('Force N');axes[4].set_xlabel('Physical time s');axes[4].legend(fontsize=7);canvas.draw()
                    out=Image.new('RGB',(1920,1080),'#102c3c');out.paste(Image.fromarray(rgb),(32,64));out.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3]),(1280,0));draw=ImageDraw.Draw(out);draw.rectangle((0,0,1279,107),fill='#102c3c');draw.text((18,8),'ACTUATOR DYNAMICS AND DISTURBANCE ROBUSTNESS STUDY',fill='white',font=font);draw.text((18,37),f'{folder.name} | ACTUAL LIVE EXECUTION RGB | tau={tau}s display only',fill='#7ce5ff',font=small);draw.text((18,68),f't={r.time_s:.3f}s | world Fz={r.external_world_2:+.2f}N | Mx={r.external_world_3:+.3f}Nm',fill='#ffdb83',font=font);draw.rectangle((0,912,1279,1079),fill='#102c3c');status='PROTECT / FAILED' if pd.notna(r.failure) else 'RUNNING' if r.time_s<6-1e-8 else 'COMPLETED'
                    for k,label in enumerate([f'step {int(r.step)} | {status} | synthetic plant | no PPO',f'XYZ error mm: '+', '.join(f'{(r[f"p_{j}"]-r[f"p_ref_{j}"])*1000:+.5f}' for j in range(3)),f'SO3 {r.orientation_error_deg:.5f}deg | shared extra {max(abs(r[f"F_add_total_{j}"]) for j in range(6)):.3f}/5 N',f'PARTIAL PROXY {r.proxy_clearance_m*1000:.2f}mm | command/drive/actual separately logged']):draw.text((18,925+34*k),label,fill='white',font=small)
                    writer.append_data(np.asarray(out));audit.append(dict(output_frame=len(audit),folder=str(folder),state_index=i,time_s=float(r.time_s),qpos_sha256=digest,source_frame=f['frame'],input_step=int(r.input_step),load_active=bool(r.load_active),external_world=vec(d.iloc[[i]],'external_world',6)[0].tolist(),last_curve_time_s=float(past.time_s.iloc[-1])))
            finally:reader.close();z.close()
            sources[-1]['unique_actual_RGB_frames']=len(source_unique);assert len(source_unique)>1
            print('VIDEO CLIP',folder.name,'done',flush=True)
    finally:writer.close()
    decoded=imageio.get_reader(output);unique=set();count=0
    try:
        for rgb in decoded:unique.add(hashlib.sha256(rgb.tobytes()).hexdigest());count+=1
    finally:decoded.close()
    assert count==len(audit) and len(unique)>1
    write_json(output.with_suffix('.validation.json'),dict(output=str(output),sha256=sha(output),frames=count,unique_decoded_frames=len(unique),fps=25,duration_s=count/25,resolution=[1920,1080],all_frames_decoded=True,all_state_hashes_match=True,no_future_curves=all(x['last_curve_time_s']<=x['time_s']+1e-10 for x in audit),mj_step_calls=0,source='ACTUAL LIVE RGB, no state reconstruction/reintegration',sources=sources,motion=motion,frame_mapping=audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--output',type=Path,default=ROOT/'assets/videos/stewart_phase2c0_dynamic_robustness.mp4');a=p.parse_args();compose(a.run_root,a.output)
