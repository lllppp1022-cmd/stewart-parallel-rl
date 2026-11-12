import _bootstrap
import argparse,time,json,faulthandler
from src.model import ROOT,load_model,write_json
from src.phase1.trajectory import Trajectory,protocol
from src.phase1.collision import CollisionQuery
from src.phase1.executor import run_trial
from src.rendering import camera

def main():
    faulthandler.enable(all_threads=True);p=argparse.ArgumentParser(description='Actual Phase1 force control, no training and no live qpos playback')
    p.add_argument('--controller',choices=['C0','C1','C2'],default='C0');p.add_argument('--trajectory',choices=protocol()['nominal_trajectories'],default='z_roll');p.add_argument('--condition',choices=['nominal',*protocol()['disturbances']],default='nominal');p.add_argument('--viewer',action='store_true');p.add_argument('--seconds',type=float,default=6);p.add_argument('--timestep',type=float,choices=[.001,.0005],default=.001);args=p.parse_args()
    if not .1<=args.seconds<=120:raise ValueError('Seconds must be explicitly bounded in [.1,120]')
    nominal=load_model();nominal.opt.timestep=.001;tr=Trajectory(nominal,args.trajectory,ROOT/'results/phase1_preflight/references');folder=ROOT/'results/runs'/f'phase1_gui_{time.time_ns()}';query=CollisionQuery()
    def factory(m,d):
        from mujoco import viewer as mv
        handle=mv.launch_passive(m,d)
        with handle.lock():
            cam=camera();handle.cam.lookat[:]=cam.lookat;handle.cam.distance=cam.distance;handle.cam.azimuth=cam.azimuth;handle.cam.elevation=cam.elevation
        handle.set_texts((None,None,'DYNAMIC ACTUATOR SIMULATION',f'{args.controller} / {args.trajectory} / {args.condition}\ndt={args.timestep}s; six FORCE actuators\nReal owner integration; no per-step viewer lock\nFull metrics in gzip CSV; partial collision monitor only'))
        return handle
    summary,_=run_trial(nominal,tr,args.controller,args.condition,args.timestep,folder,query,realtime=args.viewer,viewer_factory=factory if args.viewer else None,time_budget_s=args.seconds)
    write_json(folder/'demo_exit.json',dict(normal_python_exit=True,user_closed=summary['viewer_closed'],formal_suite_trial=False,reference_motion_ends_at_s=6,extra_time_is_static_hold=True));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
