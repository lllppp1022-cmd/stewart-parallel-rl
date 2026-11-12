"""Compose recorded real RGB only; no environment, predict, stepping or qpos writing."""
import _bootstrap
import json,hashlib
from pathlib import Path
import imageio.v2 as imageio,numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from research_phase2c2 import RUN

def main():
    sources=[RUN/f'seed42/actual_training/episode_{n:04d}' for n in [2,8,16]];sources=[p for p in sources if (p/'actual_rgb.mp4').exists()]
    if not sources:raise RuntimeError('No actual training RGB available')
    output=ROOT/'assets/videos/phase2c2_actual_training_visible_v2.mp4';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);large=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',33);audits=[];expected=0
    with imageio.get_writer(output,fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
        for folder in sources:
            before=sha(folder/'actual_rgb.mp4');frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text(encoding="utf-8").splitlines()];d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');st=np.load(folder/'integration_states.npz')['states'];reader=imageio.get_reader(folder/'actual_rgb.mp4');home=json.loads((folder/'home_baseline.json').read_text(encoding="utf-8"))['p_m'][2];q=d[[f'q_{j}' for j in range(6)]].to_numpy();changed=[];prev=None;matches=[]
            for idx,frame in enumerate(reader):
                meta=frames[idx];row=d.iloc[meta['state_index']];matches.append(hashlib.sha256(st[meta['state_index'],1:44].tobytes()).hexdigest()==meta['qpos_sha256'])
                if prev is not None:changed.append(float(np.mean(abs(frame.astype(float)-prev.astype(float)))))
                prev=frame.copy();img=Image.new('RGB',(1920,1080),(10,18,30));draw=ImageDraw.Draw(img);draw.text((28,18),'ACTUAL STOCHASTIC TRAINING | 3 mm physical Z trajectory',font=large,fill='white');draw.text((28,65),'4x slower playback | left: recorded live RGB | right: 4x pixel crop of the SAME frame',font=font,fill='#9dd5ef')
                draw.text((28,106),f'{folder.name}   physics t={row.time_s:.3f}s   global decision={int(row.rl_global_decision)}   policy version={int(row.policy_version)}',font=font,fill='white')
                raw=Image.fromarray(frame);img.paste(raw.resize((1184,666)),(16,155));img.paste(raw.crop((470,170,640,300)).resize((680,520),Image.Resampling.LANCZOS),(1224,155));draw.text((1224,690),'REAL RGB CROP x4 (camera pixels only)',font=font,fill='#f5c35c');draw.text((1224,728),'No simulated displacement scaling.',font=font,fill='white');draw.text((1224,766),f'Z error: {(row.p_2-row.p_ref_2)*1e6:+.2f} um',font=font,fill='#5cddaf')
                draw.text((28,832),f'Action={row.rl_normalized_action:+.4f}  bounded command delta Fz={row.rl_applied_command_wrench_2:+.4f} N (before lag)  cost={sum(row["reward_cost_"+k] for k in ["z","xy","angle","amplitude","slew","request"]):.4f}',font=font,fill='white')
                # Recorded trace shown causally up to this frame, in physical units.
                draw.rectangle((30,900,1890,1030),outline='#668094');sub=d.iloc[:meta['state_index']+1:20]
                for col,color in [('p_2','#43d69e'),('p_ref_2','#f5be50')]:
                    points=[(30+float(t)/6*1860,1020-(float(z)-home)*1000/3.5*105) for t,z in zip(sub.time_s,sub[col])]
                    if len(points)>1:draw.line(points,fill=color,width=3)
                draw.text((34,1040),'Z from Home: green actual, yellow reference | horizontal scale: 0..6 physical seconds | vertical: 0..3.5 mm',font=font,fill='white')
                for _ in range(4):writer.append_data(np.asarray(img));expected+=1
            reader.close();audits.append(dict(folder=str(folder),frames=len(frames),all_qpos_hashes_match=all(matches),mean_frame_difference=float(np.mean(changed)),maximum_slide_displacement_m=float(np.ptp(q,axis=0).max()),z_displacement_mm=float(np.ptp(d.p_2)*1000),source_sha256=before,source_unchanged=before==sha(folder/'actual_rgb.mp4'),original_physical_seconds=len(frames)/25))
    reader=imageio.get_reader(output);decoded=sum(1 for _ in reader);reader.close();report=dict(video=str(output),sha256=sha(output),decoded_frames=decoded,expected_frames=expected,playback_seconds=decoded/25,slowdown=4,crop_zoom=4,sources=audits,passed=decoded==expected and all(a['all_qpos_hashes_match'] and a['source_unchanged'] and a['mean_frame_difference']>0 and a['z_displacement_mm']>2.9 for a in audits));write_json(ROOT/'results/phase2c2_recording_validation.json',report);assert report['passed']
if __name__=='__main__':main()
