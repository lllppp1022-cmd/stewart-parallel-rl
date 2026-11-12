"""Only recorded live RGB; never predicts actions or writes a physics state."""
import _bootstrap,argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from src.model import ROOT,write_json,sha
from src.rl.stewart_large_motion_env import RUN

def render(kind):
    training=kind=='training';sources=[RUN/f'training/seed42/episode_{i:04d}' for i in [0,6,12]] if training else [RUN/'evaluation/E1_newtiming'/m/('episode_0000' if m=='PPO42' else '') for m in ['C4','fourDlead','FF-VA','PPO42']];target=ROOT/f'assets/videos/stewart_phase2c4_ppo_{kind}.mp4';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',31);n=0;proof=[]
    updates=pd.read_csv(RUN/'training/seed42/ppo_update_events.csv').set_index('update') if training else None
    with imageio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=8) as writer:
        for folder in sources:
            d=pd.read_csv(folder/'physics.csv.gz');frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];states=np.load(folder/'integration_states.npz')['states'];reader=imageio.get_reader(folder/'actual_rgb.mp4');hashes=[];delta=[];prev=None
            cumulative=-d[['reward_cost_'+k for k in ['z','xy','rotation','action','slew']]].sum(axis=1).cumsum()*.001/.05 if training else np.zeros(len(d))
            for j,rgb in enumerate(reader):
                f=frames[j];idx=f['state_index'];r=d.iloc[idx];hashes.append(hashlib.sha256(states[idx,1:44].tobytes()).hexdigest()==f['qpos_sha256'])
                if prev is not None:delta.append(float(np.mean(abs(rgb.astype(float)-prev.astype(float)))))
                prev=rgb.copy()
                if j%2:continue
                canvas=Image.new('RGB',(1920,1080),(10,18,30));draw=ImageDraw.Draw(canvas);label='ACTUAL STOCHASTIC PPO TRAINING' if training else 'DETERMINISTIC CHECKPOINT EVALUATION / BASELINE COMPARISON';draw.text((20,15),label,font=big,fill='white');draw.text((20,60),f'{folder.parent.name}/{folder.name} | 2x recorded live physics | 50mm, no displacement magnification',font=font,fill='#9cd4e5');canvas.paste(Image.fromarray(rgb).resize((1216,912)),(0,100));x=1234
                lines=[f'Physical time: {r.time_s:.3f}s',f'Global decision: {int(r.get("rl_global_decision",-1))} | policy update {int(r.get("policy_version",-1))}',f'Z actual/ref: {r.actual_z_from_home_m*1000:.3f}/{r.reference_z_from_home_m*1000:.3f} mm',f'Z error: {(r.p_2-r.p_ref_2)*1000:+.6f} mm',f'XY error: {np.hypot(r.p_0-r.p_ref_0,r.p_1-r.p_ref_1)*1000:.5f} mm',f'RPY: {r.rpy_0:+.5f}, {r.rpy_1:+.5f}, {r.rpy_2:+.5f} deg',f'Raw/Box action: {r.get("rl_actor_raw",0):+.4f}/{r.get("rl_clipped_action",0):+.4f}',f'RL request/accepted cmd: {r.get("rl_requested_Fz_N",0):+.5f}/{r.get("rl_accepted_Fz_N",0):+.5f} N',f'FF-VA target: {r.ff_requested_wrench_2:+.5f} N',f'Actual load: {r.external_world_2:+.3f} N [offline overlay]',f'Return so far: {float(cumulative.iloc[idx]) if training else 0:.4f} [training only]','Relative slide | actual actuator force']
                for k,s in enumerate(lines):draw.text((x,110+35*k),s,font=font,fill='#55d9a0' if k in [3,7,10] else 'white')
                for k in range(6):draw.text((x,545+32*k),f'Leg{k+1}: {r[f"q_{k}"]*1000:+.3f} mm | {r[f"force_{k}"]:+.3f} N',font=font,fill='white')
                draw.text((x,770),f'Protection: {r.failure if isinstance(r.failure,str) else "NONE"}',font=font,fill='#f4a391');draw.text((x,810),'PARTIAL geometry / no hardware safety claim',font=font,fill='#a8c9d6')
                if training:
                    version=int(r.policy_version);draw.text((x,850),'COLLECTING EXPERIENCE | updates every256 decisions',font=font,fill='white')
                    if version in updates.index:
                        u=updates.loc[version];draw.text((x,890),f'Last update: Actor delta {u.actor_change_norm:.5f}',font=font,fill='#9cd4e5');draw.text((x,925),f'Critic delta {u.critic_change_norm:.5f}, KL {u.approx_kl:.5f}',font=font,fill='#9cd4e5')
                draw.text((20,1020),'RGB captured from live MjData during the original actions; state hashes independently verified.',font=font,fill='white');writer.append_data(np.asarray(canvas));n+=1
            reader.close();proof.append(dict(folder=str(folder),frames=len(frames),all_state_hashes_equal=all(hashes),mean_frame_change=float(np.mean(delta)),actual_z_range_mm=float(np.ptp(d.actual_z_from_home_m)*1000),source_sha256=sha(folder/'actual_rgb.mp4')))
    reader=imageio.get_reader(target);decoded=sum(1 for _ in reader);reader.close();result=dict(passed=decoded==n and all(x['all_state_hashes_equal'] and x['mean_frame_change']>0 and x['actual_z_range_mm']>49 for x in proof),frames=decoded,seconds=decoded/25,playback_speed=2,sources=proof,sha256=sha(target));write_json(RUN/f'video_{kind}_validation.json',result);assert result['passed'];print(kind,'VIDEO',decoded,'frames')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['training','evaluation']);a=p.parse_args();render(a.kind)
