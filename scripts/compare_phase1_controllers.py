import _bootstrap
import argparse,json,time
import pandas as pd,numpy as np,mujoco,imageio.v2 as imageio
from src.model import ROOT,load_model,write_json,sha
from src.phase1.trajectory import Trajectory,protocol
from src.phase1.collision import CollisionQuery
from src.phase1.executor import run_trial
from src.phase1.video import LiveVideo

def main():
    p=argparse.ArgumentParser(description='Read Phase1 results or explicitly execute the bounded frozen 42-trial suite; never training')
    p.add_argument('--run-frozen-suite',action='store_true');args=p.parse_args()
    if not args.run_frozen_suite:
        print(pd.read_csv(ROOT/'results/phase1_controller_comparison.csv').to_string(index=False));return
    if (ROOT/'results/phase1_suite_manifest.json').exists():raise FileExistsError('Suite already archived; do not silently rerun/overwrite it')
    cfg=protocol();assert (ROOT/'results/phase1_preflight/complete.json').exists();folder=ROOT/'results/runs'/f'phase1_suite_{time.time_ns()}';folder.mkdir(parents=True);write_json(folder/'frozen_protocol.json',cfg)
    tasks=[(name,method,'nominal',.001) for name in cfg['nominal_trajectories'] for method in cfg['controller_names']]+[(cfg['disturbed_trajectory'],method,cond,.001) for cond in cfg['disturbances'] for method in cfg['controller_names']]+[(name,method,'nominal',.0005) for name in cfg['fine_trajectories'] for method in cfg['controller_names']]
    assert len(tasks)==42;write_json(folder/'frozen_tasks.json',tasks);nominal=load_model();nominal.opt.timestep=.001;refs={n:Trajectory(nominal,n,ROOT/'results/phase1_preflight/references') for n in cfg['nominal_trajectories']};query=CollisionQuery()
    video=ROOT/'assets/videos/phase1_dynamic_control.mp4'
    if video.exists():raise FileExistsError('Existing Phase1 video must not be overwritten')
    writer=imageio.get_writer(video,fps=10,codec='libx264',quality=8,macro_block_size=16);summaries=[];views=[];frames=[]
    try:
        for number,(name,method,cond,dt) in enumerate(tasks):
            trial_id=f'{number:02}_{name}_{method}_{cond}_{dt}';selected=dt==.001 and ((cond=='nominal' and name in ['z_roll','periodic']) or cond=='mass110');capture=LiveVideo(writer,str((folder/trial_id).relative_to(ROOT)),frames) if selected else None
            try:summary,df=run_trial(nominal,refs[name],method,cond,dt,folder/trial_id,query,frame_callback=capture)
            finally:
                if capture:capture.close()
            summary['trial_id']=trial_id;summaries.append(summary);view=df.iloc[::round(.02/dt)].copy();view['trial_id']=trial_id;views.append(view)
            pd.DataFrame(summaries).to_csv(ROOT/'results/phase1_controller_comparison.csv',index=False)
            write_json(folder/'progress.json',dict(completed=len(summaries),planned=len(tasks),tracking_physics_steps=sum(r['physical_steps'] for r in summaries)));print(json.dumps(summary),flush=True)
    finally:
        writer.close();pd.DataFrame(frames).to_csv(ROOT/'results/phase1_video_frames.csv',index=False)
        if views:pd.concat(views,ignore_index=True).to_csv(ROOT/'results/phase1_trajectory_tracking.csv',index=False)
        write_json(ROOT/'results/phase1_suite_manifest.json',dict(folder=str(folder.relative_to(ROOT)),planned_trials=42,actual_trials=len(summaries),actual_physics_steps=sum(r['physical_steps'] for r in summaries),protocol_sha256=sha(ROOT/'configs/phase1_protocol.json'),video_frames=len(frames),video_sha256=sha(video),source='Actual live force-control integration; not kinematic playback',training=False))
    assert len(summaries)==42
    print('FROZEN SUITE COMPLETE',flush=True)
if __name__=='__main__':main()
