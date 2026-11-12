import _bootstrap
import argparse,time,json,contextlib,csv,faulthandler
import mujoco,numpy as np,pandas as pd
from src.model import ROOT,load_model,write_json
from src.audit_control import AuditController,advance
from src.rendering import camera

@contextlib.contextmanager
def safe_native_viewer(model,data):
    from mujoco import viewer as native_viewer
    handle=native_viewer.launch_passive(model,data)
    try:yield handle
    finally:
        # Handle.__exit__ calls close() even after native GUI destruction.
        # Avoid calling the native exit method twice on a manually closed viewer.
        if handle.is_running():handle.close()

def demo(default_axis='home',static=False):
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser(description='Bounded Phase0 audit demonstration; never training')
    p.add_argument('--viewer',action='store_true');p.add_argument('--seconds',type=float,default=10 if static else 5);p.add_argument('--axis',choices=['z','roll','pitch','home'],default=default_axis);p.add_argument('--sign',type=int,choices=[-1,1],default=1);p.add_argument('--mode',choices=['kinematic','dynamic'],default='kinematic' if static else 'dynamic');p.add_argument('--timestep',type=float,choices=[.002,.001,.0005],default=.001);a=p.parse_args()
    if not 0<a.seconds<=300:raise ValueError('Duration must be bounded in (0,300] seconds')
    if a.mode=='kinematic' and not a.viewer:raise ValueError('Static kinematic inspection needs --viewer')
    m=load_model();m.opt.timestep=a.timestep;d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=AuditController(m,a.axis,a.sign);rows=[];folder=ROOT/'results/runs'/f'{a.mode}_demo_{time.time_ns()}';folder.mkdir(parents=True)
    context=contextlib.nullcontext(None)
    if a.viewer:
        context=safe_native_viewer(m,d)
    reason='budget_complete';wall=time.perf_counter();k=0
    write_json(folder/'started.json',dict(mode=a.mode,axis=a.axis,timestep_s=a.timestep,duration_budget_s=a.seconds,status='running',description='Missing exit.json means interrupted/native-process failure, not a completed trial'))
    journal=(folder/'physics_journal.csv').open('w',newline='',encoding='utf-8');journal_writer=None
    with context as viewer:
        if viewer:
            with viewer.lock():
                cam=camera();viewer.cam.lookat[:]=cam.lookat;viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
            # Set the classification once. Avoid racing native window shutdown with frequent overlay replacement.
            text='KINEMATIC VISUALIZATION / STATIC GEOMETRY' if a.mode=='kinematic' else 'DYNAMIC ACTUATOR SIMULATION'
            viewer.set_texts((None,None,text,f'axis={a.axis}; dt={a.timestep}s\nSix FORCE inputs; platform wrench = 0\nPhysical state and metrics archived in CSV'))
        while k<int(a.seconds/m.opt.timestep):
            if viewer is not None and not viewer.is_running():reason='viewer_closed';break
            if a.mode=='dynamic':
                # Follow the official passive-viewer loop: owner steps, then sync.
                # sync locks internally. Acquiring viewer.lock during native X-close
                # raced mutex destruction on this Windows build (recorded failure).
                row=advance(m,d,c,k+1)
                rows.append(row)
                if journal_writer is None:journal_writer=csv.DictWriter(journal,fieldnames=list(row));journal_writer.writeheader()
                journal_writer.writerow(row)
                if k%500==0:journal.flush()
            k+=1
            if viewer and k%16==0:
                viewer.sync()
            if viewer:
                delay=wall+k*m.opt.timestep-time.perf_counter()
                if delay>0:time.sleep(min(delay,.02))
    journal.close()
    if rows:pd.DataFrame(rows).to_csv(folder/'physics.csv',index=False)
    write_json(folder/'exit.json',dict(mode=a.mode,axis=a.axis,reason=reason,physical_steps=len(rows),simulation_time_s=float(d.time),timestep_s=float(m.opt.timestep),wall_seconds=time.perf_counter()-wall,source_model_unmodified=True))
    print('Saved',folder,reason,flush=True)
