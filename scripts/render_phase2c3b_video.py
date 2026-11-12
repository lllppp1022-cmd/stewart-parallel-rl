"""Actual LIVE RGB archive montage; no state/policy regeneration."""
import _bootstrap,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from src.workspace.feedforward_study import RUN

def main():
    selected=[('formal','C4_L50_T10_tau0_load0_none'),('formal','FF-VA_L50_T10_tau0_load0_none'),('formal','C4_L50_T4_tau0_load0_none'),('formal','FF-VA_L50_T4_tau0_load0_none'),('load','FF-VA_L50_T6_tau0.005_load-5.5_none')];target=ROOT/'assets/videos/stewart_phase2c3b_dynamic_feedforward.mp4';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32);count=0;audits=[]
    with imageio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=8) as writer:
        for stage,name in selected:
            folder=RUN/stage/name;r=json.loads((folder/'trial_result.json').read_text(encoding='utf-8'));d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];states=np.load(folder/'integration_states.npz')['states'];reader=imageio.get_reader(folder/'actual_rgb.mp4');hashes=[];changes=[];prev=None
            for index,rgb in enumerate(reader):
                f=frames[index];row=d.iloc[f['state_index']];hashes.append(hashlib.sha256(states[f['state_index'],1:44].tobytes()).hexdigest()==f['qpos_sha256'])
                if prev is not None:changes.append(float(np.mean(abs(rgb.astype(float)-prev.astype(float)))))
                prev=rgb.copy()
                if index%2:continue
                canvas=Image.new('RGB',(1920,1080),(10,18,30));draw=ImageDraw.Draw(canvas);draw.text((20,12),'LARGE-MOTION DYNAMIC FEEDFORWARD CONTROL STUDY',font=big,fill='white');draw.text((20,55),f'{name} | ACTUAL LIVE RGB ARCHIVE | 2x playback | no displacement scaling',font=font,fill='#97d5e8');canvas.paste(Image.fromarray(rgb).resize((1216,912)),(0,100));x=1240
                text=[f'Physical t={row.time_s:.3f} s',f'Actual / reference Z: {row.actual_z_from_home_m*1000:.3f} / {row.reference_z_from_home_m*1000:.3f} mm',f'Z error: {(row.p_2-row.p_ref_2)*1000:+.6f} mm',f'Ref v={row.twist_ref_2*1000:+.3f} mm/s',f'Ref a={row.reference_z_acceleration_m_s2*1000:+.3f} mm/s2',f'FF velocity term: {row.ff_velocity_force_N:+.4f} N',f'FF acceleration term: {row.ff_acceleration_force_N:+.4f} N',f'Total-add budget use: {row.ff_budget_peak_fraction*100:.1f}%',f'Command / drive leg1: {row.F_command_0:.3f}/{row.F_drive_0:.3f} N','Relative slide stroke | actual force']
                for k,s in enumerate(text):draw.text((x,115+36*k),s,font=font,fill='white' if k not in [2,7] else '#56dbac')
                for j in range(6):draw.text((x,495+32*j),f'Leg{j+1}: {row[f"q_{j}"]*1000:+.3f}mm | {row[f"force_{j}"]:+.3f}N',font=font,fill='white')
                draw.text((x,715),f'Protection: {row.failure if isinstance(row.failure,str) else "NONE"}',font=font,fill='#efa18b');draw.text((x,755),f'Final archived outcome: {"COMPLETED" if r["success"] else "FAILED"}',font=font,fill='#55d6a4');draw.text((x,800),'NOT PPO TRAINING | geometry remains partial',font=font,fill='#b8c7d0');draw.rectangle((30,1020,1890,1060),outline='#71848c');sub=d.iloc[:f['state_index']+1:40]
                for col,color in [('actual_z_from_home_m','#55d6a4'),('reference_z_from_home_m','#f7c45a')]:
                    points=[(30+float(t)/r['requested_duration_s']*1860,1058-float(z)/.055*36) for t,z in zip(sub.time_s,sub[col])]
                    if len(points)>1:draw.line(points,fill=color,width=2)
                writer.append_data(np.asarray(canvas));count+=1
            reader.close();audits.append(dict(trial=name,stage=stage,source_frames=len(frames),all_hashes_match=all(hashes),frame_change_mean=float(np.mean(changes)),z_range_mm=float(np.ptp(d.actual_z_from_home_m)*1000),source_sha256=sha(folder/'actual_rgb.mp4')))
    reader=imageio.get_reader(target);decoded=sum(1 for _ in reader);reader.close();result=dict(video=str(target),sha256=sha(target),decoded_frames=decoded,expected_frames=count,seconds=decoded/25,playback_speed=2,sources=audits,passed=decoded==count and all(x['all_hashes_match'] and x['frame_change_mean']>0 and x['z_range_mm']>49 for x in audits));write_json(RUN/'video_validation.json',result);assert result['passed']
if __name__=='__main__':main()
