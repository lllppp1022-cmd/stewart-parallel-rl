"""Visual framing only: real dynamic repeats, never new algorithm results."""
import _bootstrap
import time,json,shutil
import numpy as np,pandas as pd,imageio.v2 as imageio
from src.model import ROOT,load_model,write_json,sha
from src.phase1.trajectory import Trajectory
from src.phase1.collision import CollisionQuery
from src.phase1.executor import run_trial
from src.phase1.video import LiveVideo
folder=ROOT/'results/runs'/f'phase1_video_recapture_{time.time_ns()}';folder.mkdir(parents=True)
video=ROOT/'assets/videos/phase1_dynamic_control.mp4';original=video.with_name('phase1_dynamic_control_initial_capture.mp4')
if original.exists():raise FileExistsError('Recapture already performed; never replace evidence silently')
video.rename(original)
for filename in ['phase1_video_frames.csv','phase1_video_validation.json']:
 source=ROOT/'results'/filename;shutil.copy2(source,source.with_name('phase1_initial_'+filename.removeprefix('phase1_')))
formal=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');selected=formal[(formal.timestep_s==.001)&(((formal.condition=='nominal')&(formal.trajectory.isin(['z_roll','periodic'])))|(formal.condition=='mass110'))]
assert len(selected)==9
write_json(folder/'frozen_selection.json',dict(trials=selected.trial_id.tolist(),reason='Initial 190px stats overlay occluded upper platform; move camera only, keep original capture and all formal data',maximum_display_only_physics_steps=54000,counts_as_new_scientific_trials=False))
m=load_model();m.opt.timestep=.001;query=CollisionQuery();writer=imageio.get_writer(video,fps=10,codec='libx264',quality=8,macro_block_size=16);frames=[];equivalence=[]
try:
 for row in selected.to_dict('records'):
  tr=Trajectory(m,row['trajectory'],ROOT/'results/phase1_preflight/references');target=folder/row['trial_id'];capture=LiveVideo(writer,str(target.relative_to(ROOT)),frames)
  try:summary,df=run_trial(m,tr,row['controller'],row['condition'],.001,target,query,frame_callback=capture)
  finally:capture.close()
  new=np.load(target/'states.npz');old=np.load((ROOT/row['physics_log']).parent/'states.npz');equal=all(np.array_equal(new[k],old[k]) for k in ['qpos','qvel','ctrl','time']);new.close();old.close();equivalence.append(dict(trial=row['trial_id'],full_state_bitwise_equal=equal,physics_steps=summary['physical_steps'],scientific_result_replaced=False));assert equal
  print('DISPLAY-ONLY REPEAT MATCH',row['trial_id'],flush=True)
finally:writer.close();pd.DataFrame(frames).to_csv(ROOT/'results/phase1_video_frames.csv',index=False)
write_json(ROOT/'results/phase1_video_delivery_manifest.json',dict(folder=str(folder.relative_to(ROOT)),original_capture=str(original.relative_to(ROOT)),original_sha256=sha(original),delivery=str(video.relative_to(ROOT)),delivery_sha256=sha(video),frames=len(frames),equivalence=equivalence,display_only_physics_steps=sum(x['physics_steps'] for x in equivalence),formal_42_trials_unchanged=True))
