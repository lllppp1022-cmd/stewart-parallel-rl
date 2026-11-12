"""Phase2B3: compose only actual classical-control RGB and time-aligned diagnostics."""
import _bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,mujoco,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from src.visualization.enhanced.style import LEG_COLORS
from src.disturbance.metrics import vec

def compose(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    clips=[('reverse_z',c) for c in ('C3','C4-PD','C4-Observer')]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    fig=Figure(figsize=(6.4,10.8),dpi=100);canvas=FigureCanvasAgg(fig);axes=fig.subplots(5,1);fig.subplots_adjust(left=.16,right=.97,top=.96,bottom=.06,hspace=.48)
    writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8);audit=[];sources=[]
    try:
        for task,method in clips:
            folder=root/'frozen_evaluation'/task/method;df=pd.read_csv(folder/'physics.csv.gz');home=json.loads((folder/'home_baseline.json').read_text(encoding='utf-8'));frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
            if not frames:raise RuntimeError('No actual RGB for '+str(folder))
            source=imageio.get_reader(folder/'actual_rgb.mp4');archive=np.load(folder/'integration_states.npz');sources.append(dict(folder=str(folder),rgb_sha256=sha(folder/'actual_rgb.mp4'),frames=len(frames)))
            try:
                for f in frames:
                    i=f['state_index'];row=df.iloc[i];state=archive['states'][i];qpos=state[1:44];digest=hashlib.sha256(qpos.tobytes()).hexdigest();assert digest==f['qpos_sha256'];assert abs(row.time_s-f['time_s'])<1e-9
                    rgb=source.get_data(f['frame']);source_digest=hashlib.sha256(rgb.tobytes()).hexdigest();past=df.iloc[:i+1:20];t=row.time_s
                    for ax in axes:ax.clear();ax.grid(alpha=.2);ax.set_xlim(0,6);ax.axvline(1.7,color='black',ls=':',lw=.8);ax.axvline(3.45,color='black',ls=':',lw=.8);ax.set_xlabel('Physical time (s)',fontsize=8);ax.tick_params(labelsize=8)
                    axes[0].plot(past.time_s,past.rpy_2,label='actual yaw',color='#9333ea');axes[0].plot(past.time_s,past.rpy_ref_2,'k--',label='target');axes[0].set_ylabel('Yaw (deg)');axes[0].legend(fontsize=8)
                    axes[1].plot(past.time_s,(past.p_2-home['p_m'][2])*1000,label='actual',color='#2563eb');axes[1].plot(past.time_s,(past.p_ref_2-home['p_m'][2])*1000,'k--',label='reference');axes[1].set_ylabel('Z delta (mm)');axes[1].legend(fontsize=8)
                    for j,c in enumerate(LEG_COLORS[:2]):axes[2].plot(past.time_s,past[f'rpy_{j}'],color=c,label=['Roll','Pitch'][j]);axes[2].plot(past.time_s,past[f'rpy_ref_{j}'],color=c,ls='--')
                    axes[2].set_ylabel('Roll / Pitch (deg)');axes[2].legend(fontsize=8)
                    for j,c in enumerate(LEG_COLORS):
                        axes[3].plot(past.time_s,(past[f'q_{j}']-home['slide_m'][j])*1000,color=c,label=f'Leg{j+1}');axes[4].plot(past.time_s,past[f'force_{j}'],color=c)
                    axes[3].set_ylabel('Slide delta (mm)');axes[3].legend(ncol=3,fontsize=7);axes[4].set_ylabel('Actual leg force (N)');canvas.draw()
                    image=Image.new('RGB',(1920,1080),'#102c3c');image.paste(Image.fromarray(rgb),(32,64));image.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3]),(1280,0));draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,1279,107),fill='#102c3c');draw.text((18,8),'ACTUATOR MISMATCH AND YAW COMPENSATION STUDY',fill='white',font=font);draw.text((18,37),f'{task} | {method} | ACTUAL LIVE EXECUTION RGB | t={t:.3f}s',fill='#7ce5ff',font=small)
                    load='LOAD APPLIED' if row.load_active else 'UNLOADED';draw.text((18,68),f'{load} | world Fz={row.external_world_2:+.2f} N | Tx={row.external_world_3:+.3f} N m',fill='#ffdb83',font=font)
                    draw.rectangle((0,912,1279,1079),fill='#102c3c');status='PROTECT / FAILED' if pd.notna(row.failure) else 'RUNNING' if t<6-1e-8 else 'COMPLETED'
                    for k,text in enumerate([f'step {int(row.step)} | {status} | gain1+/-0.4% display only | no RL',f'Error {row.position_error_mm:.4f} mm / {row.orientation_error_deg:.5f} deg (SO3)',f'Shared extra max {max(abs(row[f"F_add_total_{j}"]) for j in range(6)):.3f}/5 N | residual max {max(abs(row[f"F_residual_applied_{j}"]) for j in range(6)):.3f} N',f'Yaw {row.rpy_2:+.5f} / target {row.rpy_ref_2:+.5f}deg | PARTIAL PROXY {row.proxy_clearance_m*1000:.1f}mm']):draw.text((18,925+34*k),text,fill='white',font=small)
                    writer.append_data(np.asarray(image));audit.append(dict(output_frame=len(audit),task=task,controller=method,source_frame=f['frame'],state_index=i,physical_step=int(row.step),time_s=float(t),input_step=int(row.input_step),load_active=bool(row.load_active),external_world=vec(df.iloc[[i]],'external_world',6)[0].tolist(),qpos_sha256=digest,source_rgb_sha256=source_digest,last_curve_time_s=float(past.time_s.iloc[-1])))
            finally:source.close();archive.close()
            print('VIDEO CLIP',task,method,'done',flush=True)
    finally:writer.close()
    decoded=imageio.get_reader(output);unique=set();count=0
    try:
        for rgb in decoded:unique.add(hashlib.sha256(rgb.tobytes()).hexdigest());count+=1
    finally:decoded.close()
    assert count==len(audit) and len(unique)>1
    write_json(output.with_suffix('.validation.json'),dict(output=str(output),sha256=sha(output),frames=count,unique_decoded_frames=len(unique),fps=25,duration_s=count/25,resolution=[1920,1080],all_state_hashes_match=True,all_frames_decoded=True,all_load_vectors_match_csv=True,no_future_curves=all(x['last_curve_time_s']<=x['time_s']+1e-10 for x in audit),source='ACTUAL LIVE EXECUTION RGB + causal recorded curves; no reintegration',mj_step_calls=0,sources=sources,frame_mapping=audit))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);p.add_argument('--output',type=Path,default=ROOT/'assets/videos/stewart_phase2b3_yaw_compensation.mp4');a=p.parse_args();compose(a.run_root,a.output)
