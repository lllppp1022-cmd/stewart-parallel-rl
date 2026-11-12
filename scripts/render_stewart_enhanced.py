"""Phase 1.5.1: actual RGB + true projected labels + causal Delta curves/events."""
import _bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image,ImageDraw,ImageFont
import mujoco
from src.model import write_json
from src.visualization.enhanced.style import LEG_COLORS,series
from src.visualization.enhanced.events import read_events,visible_events,latest_event
from src.visualization.enhanced.overlay import ProjectionOverlay

def compose(folder,output):
    folder=Path(folder);output=Path(output)
    if output.exists():raise FileExistsError(output)
    home=json.loads((folder/'home_baseline.json').read_text(encoding='utf-8'));df=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()];events=read_events(folder);states=np.load(folder/'integration_states.npz');overlay=ProjectionOverlay();reader=imageio.get_reader(folder/'actual_rgb.mp4');output.parent.mkdir(parents=True,exist_ok=True);writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8)
    fig=Figure(figsize=(7.04,9.12),dpi=100,facecolor='#f7f9fc');canvas=FigureCanvasAgg(fig);axes=fig.subplots(3,1);fig.subplots_adjust(left=.13,right=.97,top=.94,bottom=.065,hspace=.4);lines=[];markers=[]
    groups=[('angles','rpy','rpy_ref',3,'Roll / Pitch / Yaw','deg'),('legs','q','q_ref',6,'Delta slide from fixed episode Home','mm'),('force','force','f_cmd',6,'Actual actuator force (dashed = command)','N')]
    for ax,(key,actual,ref,n,title,unit) in zip(axes,groups):
        ax.set_title(title,fontsize=12);ax.set_ylabel(unit);ax.set_xlabel('Physical simulation time (s)');ax.grid(alpha=.2);pairs=[]
        for j in range(n):
            a,=ax.plot([],[],color=LEG_COLORS[j],lw=1.6,label=['Roll','Pitch','Yaw'][j] if n==3 else f'Leg{j+1}');r,=ax.plot([],[],color=LEG_COLORS[j],lw=.9,ls='--');pairs.append((a,r))
        ax.legend(fontsize=8,ncol=3,loc='upper left');lines.append(pairs)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18);audit=[]
    try:
        for oi,f in enumerate(frames):
            row=df.iloc[f['state_index']];t=float(f['time_s']);assert abs(t-row.time_s)<1e-10;mujoco.mj_setState(overlay.model,overlay.data,states['states'][f['state_index']],mujoco.mjtState.mjSTATE_INTEGRATION);digest=hashlib.sha256(overlay.data.qpos.tobytes()).hexdigest();assert digest==f['qpos_sha256']
            raw=reader.get_data(f['frame']);source_sha=hashlib.sha256(raw.tobytes()).hexdigest();rgb,pixels=overlay.apply(raw,overlay.data.qpos.copy(),overlay.data.qvel.copy(),overlay.data.ctrl.copy(),t)
            past=df[(df.time_s<=t+1e-10)&(df.time_s>=max(0,t-10))].iloc[::10];times=past.time_s.to_numpy()
            for marker in markers:marker.remove()
            markers=[]
            for ax,pairs,(key,actual,ref,n,title,unit) in zip(axes,lines,groups):
                av,rv,_=series(past[[f'{actual}_{j}' for j in range(n)]].to_numpy(),past[[f'{ref}_{j}' for j in range(n)]].to_numpy(),key,'Delta from Home',home);values=[]
                for j,(a,r) in enumerate(pairs):a.set_data(times,av[:,j]);r.set_data(times,rv[:,j]);values.extend([av[:,j],rv[:,j]])
                lo=min(float(v.min()) for v in values);hi=max(float(v.max()) for v in values);pad=max((hi-lo)*.18,.005 if unit=='deg' else .02 if unit=='mm' else .3);ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(max(0,t-10),max(.1,t))
                for e in visible_events(events,t):
                    if e['type'] in ('plan_accepted','hold','protect_hold','target_rejected','motion_completed'):markers.append(ax.axvline(e['time_s'],color='#aab2be',lw=.6,ls=':'))
            canvas.draw();image=Image.new('RGB',(1920,1080),'#102c3c');image.paste(Image.fromarray(rgb),(0,0));image.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3]),(1216,0));draw=ImageDraw.Draw(image);draw.rectangle((0,0,1215,139),fill='#102c3c');draw.text((18,10),'DYNAMIC CONTROL DEMONSTRATION | ACTUAL EXECUTION CAPTURE',font=font,fill='white');draw.text((18,40),'True 3D-projected colour labels; original robot motion unchanged',font=small,fill='#87e4ff')
            e=latest_event(events,t);event_text=f'SIM {e["time_s"]:.3f}s | {e["label"]} | command {e.get("command_id","?")}' if e else 'No event yet';draw.text((18,69),event_text,font=font,fill='#ffd785')
            draw.text((18,101),f'REF/ACT: dZ {(row.p_ref_2-home["p_m"][2])*1000:+.3f}/{(row.p_2-home["p_m"][2])*1000:+.3f} mm | R {row.rpy_ref_0:+.3f}/{row.rpy_0:+.3f} | P {row.rpy_ref_1:+.3f}/{row.rpy_1:+.3f} deg',font=small,fill='white')
            draw.rectangle((0,912,1919,1079),fill='#102c3c');texts=[f'Episode {row.episode} | {row.controller} | sim {t:.3f}s | {row.status} | step {row.step}',f'Pose {row.position_error_mm:.4f} mm / {row.orientation_error_deg:.5f} deg | max leg tracking error {row.max_leg_error_mm:.4f} mm',f'Weld {row.weld_translation_m:.2e} m / {row.weld_angle_rad:.2e} rad | PARTIAL PROXY QUERY {row.proxy_clearance_m*1000:.2f} mm','Fixed episode Home baseline | dashed=reference, solid=actual | scheduled commands, NOT human keyboard']
            for k,text in enumerate(texts):draw.text((18,925+k*34),text,font=font if k<2 else small,fill='white')
            writer.append_data(np.asarray(image));audit.append(dict(output_frame=oi,state_index=f['state_index'],source_frame=f['frame'],time_s=t,qpos_sha256=digest,source_decoded_rgb_sha256=source_sha,latest_curve_time_s=float(past.time_s.max()),event_journal_index=e['journal_index'] if e else None,event_time_s=e['time_s'] if e else None,projected_leg_pixels=pixels))
            if oi%100==0:print(f'{oi}/{len(frames)}',flush=True)
    finally:writer.close();reader.close();states.close()
    result=dict(source=str(folder),output=str(output),resolution=[1920,1080],fps=25,frames=len(audit),duration_s=len(audit)/25,source_kind='ACTUAL EXECUTION CAPTURE + projected visual overlay',home_baseline=home,colors=LEG_COLORS,mj_step_calls=0,frame_mapping=audit,no_future_curves=all(x['latest_curve_time_s']<=x['time_s']+1e-10 for x in audit),no_future_events=all(x['event_time_s'] is None or x['event_time_s']<=x['time_s']+1e-10 for x in audit),all_qpos_hashes_match=True)
    write_json(output.with_suffix('.frame_audit.json'),result);fig.savefig(output.with_suffix('.curves.png'));print('Complete',len(audit),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode');p.add_argument('output');a=p.parse_args();compose(a.episode,a.output)
