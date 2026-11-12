import _bootstrap
import json,hashlib
import numpy as np,pandas as pd,imageio.v2 as imageio
from src.model import ROOT,write_json,sha
summary=pd.read_csv(ROOT/'results/phase0_dynamics_check.csv');metadata=pd.read_csv(ROOT/'results/phase0_video_frames.csv');matches=0;motion={}
for row in summary.to_dict('records'):
    folder=(ROOT/row['log']).parent;z=np.load(folder/'physical_states.npz')
    subset=metadata[metadata.scenario==row['scenario']]
    for frame in subset.to_dict('records'):
        assert hashlib.sha256(z['qpos'][frame['physics_step']-1].tobytes()).hexdigest()==frame['qpos_sha256'];matches+=1
    motion[row['scenario']]=dict(max_generalized_coordinate_change=float(np.max(np.ptp(z['qpos'],axis=0))),max_slide_change_m=float(np.max(np.ptp(z['qpos'][:,[2,8,14,20,26,32]],axis=0))))
    z.close()
video=ROOT/'assets/videos/phase0_simulation.mp4';reader=imageio.get_reader(video);meta=reader.get_meta_data();n=0;scenes=set()
for frame in reader:n+=1;scenes.add(hashlib.sha256(frame[130:].tobytes()).hexdigest())
reader.close();assert n==len(metadata)==450 and len(scenes)>10
result=dict(video=str(video.relative_to(ROOT)),sha256=sha(video),frames=n,fps=meta['fps'],duration_s=n/meta['fps'],resolution=[960,720],unique_scene_hashes=len(scenes),state_log_frame_matches=matches,mode='Dynamic Actuator Simulation / actual RGB captured from live MjData during original 2ms formal run',not_kinematic_reconstruction=True,formal_physics_steps=int(summary.steps.sum()),motion=motion)
write_json(ROOT/'results/phase0_video_validation.json',result)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
selected=summary.loc[summary.scenario=='z_up'].iloc[0];df=pd.read_csv(ROOT/selected['log'])
fig,axs=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
for j in range(1,7):
    axs[0].plot(df.time_s,df[f'stroke_actual_m_{j}']*1000,label=f'Leg{j} actual');axs[0].plot(df.time_s,df[f'stroke_ref_m_{j}']*1000,'--',alpha=.55)
    axs[1].plot(df.time_s,(df[f'stroke_actual_m_{j}']-df[f'stroke_ref_m_{j}'])*1e6,label=f'Leg{j}')
    axs[2].plot(df.time_s,df[f'actuator_force_N_{j}'],label=f'Leg{j}')
for ax,y in zip(axs,['Slide coordinate / relative stroke (mm)','Actual minus reference stroke (um)','Actual actuator force (N)']):ax.set(xlabel='Physical time (s)',ylabel=y);ax.grid(alpha=.2);ax.legend(ncol=3)
fig.savefig(ROOT/'assets/plots/phase0_leg_lengths_forces.png',dpi=150);plt.close(fig)
step=pd.read_csv(ROOT/'results/phase0_timestep_sensitivity.csv');fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
for label in ['home','z_up','roll_pos','pitch_pos']:
    values=[float(summary.loc[summary.scenario==label,'max_weld_angle_rad'].iloc[0])]+[float(step.loc[(step.scenario==label)&(step.timestep_s==dt),'max_weld_angle_rad'].iloc[0]) for dt in [.001,.0005]];ax.plot([2,1,.5],values,'o-',label=label)
ax.axhline(1e-4,color='red',linestyle='--',label='Frozen angle threshold');ax.set(xlabel='Physics timestep (ms)',ylabel='Max weld orientation gap (rad)',yscale='log');ax.legend();ax.grid(alpha=.2);fig.savefig(ROOT/'assets/plots/phase0_timestep_sensitivity.png',dpi=150);plt.close(fig)
print(json.dumps(result,indent=2))
