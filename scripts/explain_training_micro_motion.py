"""Read-only explanation of micro-motion: original RGB plus a calibrated gauge.

The gauge is a schematic, not a re-rendered or magnified physical robot. No policy
or MuJoCo integration is run. Four repeated frames provide explicit 4x slow play.
"""
import _bootstrap,hashlib,json
import numpy as np,pandas as pd,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,sha,write_json

def main():
    folder=ROOT/'results/runs/phase2c1_20261029_v1/seed42/actual_training/episode_0002'
    output=ROOT/'assets/videos/phase2c1_micro_motion_explained.mp4'
    if output.exists():raise FileExistsError(output)
    source_hash=sha(folder/'actual_rgb.mp4');df=pd.read_csv(folder/'physics.csv.gz');frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()];z=np.load(folder/'integration_states.npz');reader=imageio.get_reader(folder/'actual_rgb.mp4');writer=imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=8)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',24);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18);mapping=[];source_unique=set();error_um=(df.p_2-df.p_ref_2).to_numpy()*1e6
    try:
        for f in frames:
            i=f['state_index'];row=df.iloc[i];digest=hashlib.sha256(z['states'][i,1:44].tobytes()).hexdigest();assert digest==f['qpos_sha256'] and abs(row.time_s-f['time_s'])<1e-10
            rgb=reader.get_data(f['frame']);source_unique.add(hashlib.sha256(rgb.tobytes()).hexdigest());im=Image.new('RGB',(1920,1080),'#102c3c');im.paste(Image.fromarray(rgb),(24,105));draw=ImageDraw.Draw(im)
            draw.text((24,15),'ORIGINAL ACTUAL TRAINING RGB | seed42 episode2 | 4x SLOW PLAY',font=font,fill='white');draw.text((24,55),f'Physical time {row.time_s:.3f}s | Decision {int(row.rl_global_decision)} | Policy version {int(row.policy_version)}',font=small,fill='#75ddfa')
            draw.text((1280,25),'MEASURED Z ERROR GAUGE',font=font,fill='#ffdb83');draw.text((1280,65),'SCHEMATIC - NOT ROBOT GEOMETRY',font=small,fill='white');draw.text((1280,95),'2 pixels per micrometre (um)',font=small,fill='white')
            center=360
            for val in [-100,-50,0,50,100]:
                y=center-int(2*val);draw.line((1320,y,1840,y),fill='#3c5667' if val else '#a9e6ff',width=1);draw.text((1850,y-10),f'{val:+d}',font=small,fill='white')
            ypos=center-int(round(2*error_um[i]));draw.rectangle((1400,ypos-9,1760,ypos+9),fill='#ffba55');draw.line((1580,center,1580,ypos),fill='#ffba55',width=4);draw.text((1300,590),f'Actual Z error: {error_um[i]:+.3f} um',font=font,fill='#ffdb83');draw.text((1300,625),'Reference: fixed Home (no tracking move)',font=small,fill='white')
            draw.text((1280,690),'CAUSAL ERROR HISTORY (um)',font=small,fill='white');left,right,top,bottom=1300,1860,740,975;draw.rectangle((left,top,right,bottom),outline='#8baab9');mid=(top+bottom)/2;draw.line((left,mid,right,mid),fill='#8baab9')
            idx=np.arange(0,i+1,4);points=[(left+float(df.time_s.iloc[j])/6*(right-left),mid-float(error_um[j])/100*(bottom-top)/2) for j in idx]
            if len(points)>1:draw.line(points,fill='#7ce5ff',width=2)
            draw.text((left,bottom+9),'0                Physical time s              6',font=small,fill='white');draw.text((25,1030),'Robot RGB unchanged. Gauge uses logged micrometres. No extra simulation or policy calls.',font=small,fill='white')
            for repeat in range(4):writer.append_data(np.asarray(im));mapping.append(dict(output_frame=len(mapping),source_frame=f['frame'],repeat=repeat,state_index=i,time_s=float(row.time_s),z_error_um=float(error_um[i]),qpos_sha256=digest))
            if f['frame']==75:im.save(ROOT/'assets/plots/phase2c1_micro_motion_explained.png')
    finally:reader.close();writer.close();z.close()
    decoder=imageio.get_reader(output);count=0;unique=set()
    try:
        for rgb in decoder:count+=1;unique.add(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:decoder.close()
    assert count==len(mapping)==600 and len(unique)>1 and source_hash==sha(folder/'actual_rgb.mp4')
    stats=[]
    for ep in [2,16,32]:
        d=pd.read_csv(folder.parent/f'episode_{ep:04d}/physics.csv.gz',usecols=['p_2','p_ref_2']+[f'q_{j}' for j in range(6)]);stats.append(dict(episode=ep,z_peak_error_um=float(abs(d.p_2-d.p_ref_2).max()*1e6),z_peak_to_peak_um=float(np.ptp(d.p_2)*1e6),slide_peak_to_peak_um=[float(np.ptp(d[f'q_{j}'])*1e6) for j in range(6)],fixed_reference=bool(d.p_ref_2.nunique()==1)))
    write_json(ROOT/'results/phase2c1_visibility_diagnosis.json',dict(source=str(folder),source_rgb_sha256=source_hash,source_unchanged=True,episode_measurements=stats,output=str(output),output_sha256=sha(output),frames=count,unique_decoded_frames=len(unique),unique_source_frames=len(source_unique),physical_duration_s=6,play_duration_s=24,playback_speed=.25,gauge_px_per_um=2,gauge_is_schematic=True,no_robot_geometry_amplification=True,no_physics_steps=True,no_policy_calls=True,frame_mapping=mapping))
    print('VERIFIED',count,'frames; physical6s / playback24s; original RGB unchanged',flush=True)

if __name__=='__main__':main()
