"""Composite on-site RGB with strictly causal physical curves; no simulation."""
import _bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image,ImageDraw,ImageFont
from src.model import write_json

COLORS=['#009add','#e99017','#36a94b','#e6547b','#946bdd','#998900']
def compose(folder,output,title='Actual execution',start=0,end=None):
    folder=Path(folder);output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines() if s]
    selected=[f for f in frames if f['time_s']>=start and (end is None or f['time_s']<=end)]
    if not selected:raise ValueError('No actual RGB frames in selected interval')
    archive=np.load(folder/'integration_states.npz');reader=imageio.get_reader(folder/'actual_rgb.mp4');writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8)
    # Exact integration state is audited against the original qpos hashes, never re-integrated.
    import mujoco
    from src.phase1.plant import create_plant
    cfg=json.loads((folder/'config.json').read_text());m=create_plant(cfg.get('condition','nominal'));scratch=mujoco.MjData(m)
    fig=Figure(figsize=(7.04,9.12),dpi=100,facecolor='#f7f9fc');canvas=FigureCanvasAgg(fig);axes=fig.subplots(3,1);fig.subplots_adjust(left=.13,right=.97,top=.94,bottom=.065,hspace=.38)
    titles=['Roll / Pitch / Yaw (dashed = reference)','Six relative slide strokes (dashed = reference)','Six actuator forces (dashed = command)'];groups=[('rpy','rpy_ref',3,1,'deg'),('q','q_ref',6,1000,'mm'),('force','f_cmd',6,1,'N')];lines=[]
    for ax,name,(_,_,count,_,unit) in zip(axes,titles,groups):
        ax.set_title(name,fontsize=12);ax.set_ylabel(unit);ax.set_xlabel('Physical simulation time (s)');ax.grid(alpha=.22);pairs=[]
        for j in range(count):
            a,=ax.plot([],[],color=COLORS[j],lw=1.5,label=['Roll','Pitch','Yaw'][j] if count==3 else f'Leg{j+1}');r,=ax.plot([],[],color=COLORS[j],lw=.9,ls='--');pairs.append((a,r))
        ax.legend(fontsize=8,ncol=3,loc='upper left');lines.append(pairs)
    try:font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',19)
    except OSError:font=ImageFont.load_default();small=font
    audit=[];previous=None;changes=[]
    try:
        for oi,f in enumerate(selected):
            idx=f['state_index'];row=df.iloc[idx];t=float(f['time_s']);assert abs(row.time_s-t)<1e-10
            mujoco.mj_setState(m,scratch,archive['states'][idx],mujoco.mjtState.mjSTATE_INTEGRATION)
            digest=hashlib.sha256(scratch.qpos.tobytes()).hexdigest();assert digest==f['qpos_sha256']
            past=df[(df.time_s<=t+1e-10)&(df.time_s>=max(0,t-10))].iloc[::10];times=past.time_s.to_numpy()
            for ax,pairs,(actual,ref,count,scale,unit) in zip(axes,lines,groups):
                values=[]
                for j,(a,r) in enumerate(pairs):
                    av=past[f'{actual}_{j}'].to_numpy()*scale;rv=past[f'{ref}_{j}'].to_numpy()*scale;a.set_data(times,av);r.set_data(times,rv);values.extend([av,rv])
                ax.set_xlim(max(0,t-10),max(.1,t));lo=min(float(x.min()) for x in values);hi=max(float(x.max()) for x in values);pad=max((hi-lo)*.18, .005 if unit=='deg' else .015 if unit=='mm' else .3);ax.set_ylim(lo-pad,hi+pad)
            canvas.draw();right=np.asarray(canvas.buffer_rgba())[:,:,:3];rgb=reader.get_data(f['frame']);assert rgb.shape==(912,1216,3)
            image=Image.new('RGB',(1920,1080),'#101e2d');image.paste(Image.fromarray(rgb),(0,0));image.paste(Image.fromarray(right),(1216,0));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,1215,76),fill='#102c3c');draw.text((22,12),'DYNAMIC CONTROL DEMONSTRATION',font=font,fill='white');draw.text((22,43),'ACTUAL EXECUTION CAPTURE | '+title,font=small,fill='#8de5ff')
            status=str(row.status);failure=str(row.failure) if pd.notna(row.failure) else 'none';draw.rectangle((0,912,1919,1079),fill='#6e2438' if failure!='none' else '#102c3c')
            text=[f'Episode {int(row.episode)} | {row.controller} | sim {t:.3f} s | physics step {int(row.step)} | {status}',f'Pose error {row.position_error_mm:.4f} mm / {row.orientation_error_deg:.5f} deg | max leg error {row.max_leg_error_mm:.4f} mm',f'Weld {row.weld_translation_m:.2e} m / {row.weld_angle_rad:.2e} rad | PARTIAL PROXY QUERY {row.proxy_clearance_m*1000:.2f} mm',f'Protection: {failure} | +/-400 N | source frame {f["frame"]} | curves contain no future samples']
            for k,s in enumerate(text):draw.text((22,925+k*34),s,font=font if k<2 else small,fill='white')
            out=np.asarray(image);writer.append_data(out)
            sample=rgb[::12,::12].astype(float)
            if previous is not None:changes.append(float(np.mean(abs(sample-previous))))
            previous=sample
            audit.append(dict(output_frame=oi,source_frame=f['frame'],state_index=idx,time_s=t,latest_curve_time_s=float(past.time_s.max()),qpos_sha256=digest,qvel_sha256=hashlib.sha256(scratch.qvel.tobytes()).hexdigest(),ctrl_sha256=hashlib.sha256(scratch.ctrl.tobytes()).hexdigest(),status=status))
            if oi%100==0:print(f'{output.name}: {oi}/{len(selected)}',flush=True)
    finally:writer.close();reader.close();archive.close()
    validation=dict(output=str(output),source=str(folder),resolution=[1920,1080],fps=25,frames=len(audit),duration_s=len(audit)/25,source_kind='ACTUAL EXECUTION CAPTURE',mode='DYNAMIC CONTROL DEMONSTRATION',mj_step_calls=0,all_state_hashes_match=True,no_future_curve_samples=all(x['latest_curve_time_s']<=x['time_s']+1e-10 for x in audit),changed_rgb_frame_pairs=sum(x>0 for x in changes),maximum_rgb_mean_change=max(changes,default=0),qpos_range_max=float(np.ptp(df[[f'qpos_{j}' for j in range(43)]].to_numpy(),axis=0).max()),frame_mapping=audit)
    write_json(output.with_suffix('.frame_audit.json'),validation);fig.savefig(output.with_suffix('.curves.png'));return validation

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode');p.add_argument('output');p.add_argument('--title',default='Actual execution');p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float);a=p.parse_args();v=compose(a.episode,a.output,a.title,a.start,a.end);print({k:v[k] for k in ['frames','duration_s','all_state_hashes_match','no_future_curve_samples']})
