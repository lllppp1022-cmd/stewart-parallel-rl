"""Actual live RGB archives only; no policy call, no kinematic substitutes."""
import _bootstrap,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from analyze_phase2c3a_workspace import RUN

def main():
    candidates=['C0_L50_T10_tau0_load0','C4_L50_T10_tau0_load0','fourDlead_L50_T10_tau0_load0','fourDlead_L50_T4_tau0_load0','fourDlead_L50_T10_tau0.005_load-5.5'];folders=[RUN/'dynamics'/n for n in candidates if (RUN/'dynamics'/n/'actual_rgb.mp4').exists() and (RUN/'dynamics'/n/'summary.json').exists()]
    if not folders:raise RuntimeError('No actual large-motion RGB archives available')
    f=ROOT/'assets/videos/stewart_phase2c3a_large_motion.mp4';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32);checks=[];count=0
    with imageio.get_writer(f,fps=25,codec='libx264',quality=8,macro_block_size=8) as w:
        for folder in folders:
            result=json.loads((folder/'trial_result.json').read_text(encoding='utf-8'));d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');meta=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];states=np.load(folder/'integration_states.npz')['states'];reader=imageio.get_reader(folder/'actual_rgb.mp4');hashes=[];previous=None;diff=[]
            # Whole physical trial, including failures/protection; 2x playback clearly marked.
            for idx,frame in enumerate(reader):
                item=meta[idx];row=d.iloc[item['state_index']];hashes.append(hashlib.sha256(states[item['state_index'],1:44].tobytes()).hexdigest()==item['qpos_sha256'])
                if previous is not None:diff.append(float(np.mean(abs(frame.astype(float)-previous.astype(float)))))
                previous=frame.copy()
                if idx%2:continue
                canvas=Image.new('RGB',(1920,1080),(10,18,30));draw=ImageDraw.Draw(canvas);draw.text((20,12),'LARGE-WORKSPACE DYNAMIC MOTION VALIDATION',font=big,fill='white');draw.text((20,56),f'{folder.name} | ACTUAL LIVE RGB | 2x playback | real displacement, no geometry scaling',font=font,fill='#8ed3e6');rgb=Image.fromarray(frame);canvas.paste(rgb.resize((1216,912)),(0,100))
                x=1240;draw.text((x,110),f'Physical time {row.time_s:.3f} s',font=big,fill='white');draw.text((x,155),f'Actual Z {row.actual_z_from_home_m*1000:.3f} mm',font=font,fill='#4ad8aa');draw.text((x,190),f'Reference Z {row.reference_z_from_home_m*1000:.3f} mm',font=font,fill='#f4c55b');draw.text((x,225),f'XY error {np.hypot(row.p_0-row.p_ref_0,row.p_1-row.p_ref_1)*1000:.4f} mm',font=font,fill='white');draw.text((x,260),f'RPY {row.rpy_0:+.4f}, {row.rpy_1:+.4f}, {row.rpy_2:+.4f} deg',font=font,fill='white');draw.text((x,295),f'Jacobian condition {row.actual_jacobian_condition:.3f}',font=font,fill='white');draw.text((x,330),f'Expanded mesh gap {row.expanded_mesh_gap_m*1000:.2f} mm',font=font,fill='white')
                draw.text((x,380),'Relative slide stroke | Actual force',font=font,fill='#94b9e6')
                for j in range(6):draw.text((x,420+34*j),f'Leg{j+1}: {row[f"q_{j}"]*1000:+.3f} mm | {row[f"force_{j}"]:+.2f} N',font=font,fill='white')
                draw.text((x,655),f'Protection: {row.failure if isinstance(row.failure,str) else "NONE"}',font=font,fill='#ee9b8c');draw.text((x,695),f'Outcome: {"COMPLETED" if result["success"] else "FAILED / INCOMPLETE"}',font=big,fill='#4ad8aa' if result['success'] else '#ee796f');draw.text((x,745),'Connected joint interiors / hard stops unknown.',font=font,fill='#b9c5d0');draw.text((x,780),'NO PPO TRAINING',font=font,fill='#b9c5d0')
                draw.rectangle((35,1016,1880,1060),outline='#658292');sub=d.iloc[:item['state_index']+1:40]
                for col,color in [('actual_z_from_home_m','#4ad8aa'),('reference_z_from_home_m','#f4c55b')]:
                    pts=[(35+float(t)/result['reference']['duration_s']*1845,1058-float(z)/.055*40) for t,z in zip(sub.time_s,sub[col])]
                    if len(pts)>1:draw.line(pts,fill=color,width=2)
                w.append_data(np.asarray(canvas));count+=1
            reader.close();checks.append(dict(trial=folder.name,source_frames=len(meta),all_hashes_match=all(hashes),mean_frame_change=float(np.mean(diff)),actual_z_peak_m=float(d.actual_z_from_home_m.max()),source_sha256=sha(folder/'actual_rgb.mp4'),complete=result['complete'],success=result['success']))
    reader=imageio.get_reader(f);decoded=sum(1 for _ in reader);reader.close();write_json(RUN/'video_validation.json',dict(video=str(f),source='ACTUAL RGB DURING mj_step EXECUTION',playback_speed=2,decoded_frames=decoded,expected_frames=count,seconds=decoded/25,clips=checks,passed=count==decoded and all(x['all_hashes_match'] and x['mean_frame_change']>0 for x in checks)))
if __name__=='__main__':main()
