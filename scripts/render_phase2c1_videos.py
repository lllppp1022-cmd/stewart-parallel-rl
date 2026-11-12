"""Compose verified on-site RGB and causal curves; never call policy or physics."""
import _bootstrap,argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from src.visualization.enhanced.style import LEG_COLORS

def compose(root,kind):
    root=Path(root);output=ROOT/f'assets/videos/stewart_phase2c1_ppo_{kind}.mp4'
    if output.exists():raise FileExistsError(output)
    training=kind=='training'
    clips=[root/f'seed42/actual_training/episode_{e:04d}' for e in [2,16,32]] if training else [root/'evaluation/classical/E0'/m for m in ['C4','C4_fourDlead']]+[root/'evaluation/seed42/final_deterministic/E0/episode_0000']
    fonts=[ImageFont.truetype('C:/Windows/Fonts/consola.ttf',n) for n in [20,18,16]];fig=Figure(figsize=(6.4,10.8),dpi=100);canvas=FigureCanvasAgg(fig);axes=fig.subplots(5,1);fig.subplots_adjust(left=.18,right=.98,top=.96,bottom=.05,hspace=.6);mapping=[];sources=[]
    writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8)
    try:
        for folder in clips:
            df=pd.read_csv(folder/'physics.csv.gz');frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];reader=imageio.get_reader(folder/'actual_rgb.mp4');z=np.load(folder/'integration_states.npz');summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));name=folder.name if training else folder.parent.name if folder.name.startswith('episode') else folder.name
            qspan=float(np.ptp(z['states'][:,1:44],axis=0).max());assert qspan>1e-6;unique=set();versions=set()
            try:
                for f in frames:
                    i=f['state_index'];r=df.iloc[i];digest=hashlib.sha256(z['states'][i,1:44].tobytes()).hexdigest();assert digest==f['qpos_sha256'] and abs(r.time_s-f['time_s'])<1e-10
                    rgb=reader.get_data(f['frame']);unique.add(hashlib.sha256(rgb.tobytes()).hexdigest());past=df.iloc[:i+1:20]
                    for ax in axes:ax.clear();ax.grid(alpha=.2);ax.set_xlim(0,6);ax.tick_params(labelsize=7)
                    for j,color in enumerate(LEG_COLORS[:3]):axes[0].plot(past.time_s,(past[f'p_{j}']-past[f'p_ref_{j}'])*1000,color=color,label='XYZ'[j]);axes[1].plot(past.time_s,past[f'rpy_{j}']-past[f'rpy_ref_{j}'],color=color,label=['Roll','Pitch','Yaw'][j])
                    axes[0].set_ylabel('Position error mm');axes[1].set_ylabel('RPY error deg')
                    for j,color in enumerate(LEG_COLORS):
                        for field,ls in [('F_command','--'),('F_drive',':'),('force','-')]:axes[3].plot(past.time_s,past[f'{field}_{j}']-df[f'F_drive_previous_{j}'].iloc[0],color=color,ls=ls,lw=.7)
                    axes[3].set_ylabel('6-leg force delta N');axes[3].set_title('Command -- / hidden drive : / actual solid',fontsize=8)
                    rl='rl_actor_raw' in df
                    if rl:
                        axes[2].plot(past.time_s,past.rl_actor_raw,label='raw Gaussian');axes[2].plot(past.time_s,past.rl_normalized_action,label='Box clipped');axes[2].plot(past.time_s,past.rl_actor_mean,label='Actor mean');axes[2].set_ylabel('Actor action')
                        effect=abs(past[[f'rl_applied_command_{j}' for j in range(6)]]).max(axis=1);axes[4].plot(past.time_s,effect,label='RL leg command effect N');axes[4].plot(past.time_s,past.rl_mapped_target4_0,label='Fz target N');axes[4].plot(past.time_s,-past[[f'reward_cost_{k}' for k in ['z','xy','angle','amplitude','slew','request']]].sum(axis=1),label='Instant reward rate');axes[4].set_ylabel('Reward / force')
                        version=int(r.policy_version);versions.add(version);status=f'decision {int(r.rl_global_decision)} | policy version {version}'
                        details=[f'Actor raw {r.rl_actor_raw:+.4f}, mean {r.rl_actor_mean:+.4f}, std {r.rl_actor_std:.4f}',f'Fz target {r.rl_mapped_target4_0:+.4f}N | max actual RL leg effect {max(abs(r[f"rl_applied_command_{j}"]) for j in range(6)):.4f}N']
                    else:
                        axes[2].plot(past.time_s,abs(past[[f'F_add_total_{j}' for j in range(6)]]).max(axis=1),label='Shared extra max N');axes[2].set_ylabel('Shared budget N');axes[4].plot(past.time_s,past.orientation_error_deg,label='SO3 error deg');axes[4].set_ylabel('Orientation deg');status='Frozen classical baseline';details=['C4 / fourDlead share original +/-5N and 20N/s budget','No policy sampling in this baseline']
                    for ax in axes:ax.legend(fontsize=6,loc='upper right') if ax.lines and ax.get_legend_handles_labels()[1] else None
                    axes[-1].set_xlabel('Physical time s (past only)');canvas.draw();out=Image.new('RGB',(1920,1080),'#102c3c');out.paste(Image.fromarray(rgb),(32,64));out.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3]),(1280,0));draw=ImageDraw.Draw(out);draw.rectangle((0,0,1279,110),fill='#102c3c');draw.text((18,8),'ACTUAL STOCHASTIC TRAINING' if training else 'DETERMINISTIC EVALUATION / BASELINE COMPARISON',font=fonts[0],fill='white');draw.text((18,39),f'{name} | on-site RGB | {status}',font=fonts[1],fill='#7ce5ff');draw.text((18,70),f'Physical time {r.time_s:.3f}s | COLLECTING EXPERIENCE' if training else f'Physical time {r.time_s:.3f}s | CHECKPOINT EVALUATION' if rl else f'Physical time {r.time_s:.3f}s | BASELINE COMPARISON',font=fonts[1],fill='#ffdb83');draw.rectangle((0,912,1279,1079),fill='#102c3c')
                    details += [f'Z error {(r.p_2-r.p_ref_2)*1000:+.5f}mm | SO3 {r.orientation_error_deg:.5f}deg',f'Partial proxy clearance {r.proxy_clearance_m*1000:.2f}mm | protection {r.failure if pd.notna(r.failure) else "none"}']
                    for k,line in enumerate(details):draw.text((18,925+32*k),line,font=fonts[1],fill='white')
                    writer.append_data(np.asarray(out));mapping.append(dict(frame=len(mapping),folder=str(folder),source_frame=f['frame'],state_index=i,time_s=float(r.time_s),qpos_sha256=digest,policy_version=int(r.policy_version) if rl else None,global_decision=int(r.rl_global_decision) if rl else None,last_curve_time_s=float(past.time_s.iloc[-1])))
            finally:reader.close();z.close()
            assert len(unique)>1;sources.append(dict(folder=str(folder),RGB_sha256=sha(folder/'actual_rgb.mp4'),frames=len(frames),unique_actual_RGB=len(unique),qpos_range=qspan,policy_versions=sorted(versions),success=summary.get('failure') is None));print('COMPOSED',kind,folder,flush=True)
    finally:writer.close()
    count=0;unique=set();reader=imageio.get_reader(output)
    try:
        for rgb in reader:count+=1;unique.add(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:reader.close()
    assert count==len(mapping) and len(unique)>1
    write_json(output.with_suffix('.validation.json'),dict(output=str(output),sha256=sha(output),all_frames_decoded=True,frames=count,unique_frames=len(unique),duration_s=count/25,fps=25,resolution=[1920,1080],all_qpos_hashes_match=True,no_future_curves=all(x['last_curve_time_s']<=x['time_s'] for x in mapping),physics_steps_during_render=0,policy_calls_during_render=0,source='Actual on-site RGB, no reconstruction',sources=sources,mapping=mapping))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--kind',choices=['training','evaluation'],required=True);a=p.parse_args();compose(a.run_root,a.kind)
