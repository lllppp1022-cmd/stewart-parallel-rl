"""Coloured independent Viewer, no physical model mutation or mj_step."""
import time,json
from collections import deque
from pathlib import Path
from queue import Empty
import numpy as np,mujoco
from ...phase1.plant import create_plant
from ...rendering import camera
from ...model import write_json
from .style import color_display_model,rgba

def anchors(model,data,mapping):
    return [np.mean(data.geom_xpos[row['geoms']],axis=0) for row in mapping]

def mirror_main(config,queue,stop,closed):
    handle=None;rows=deque(maxlen=256) if config.get('stream_logs') else [];labels=True;last_camera=None;mapping=[];samples=0;maximum=0.;journal=None
    try:
        snap=None;deadline=time.perf_counter()+60
        while snap is None and not stop.is_set() and time.perf_counter()<deadline:
            try:snap=queue.get(timeout=.1)
            except Empty:pass
        if snap is None:return
        if config.get('stream_logs'):journal=(Path(config['folder'])/'mirror_samples.jsonl').open('w',encoding='utf-8')
        m=create_plant(config.get('condition','nominal'));mapping=color_display_model(m);d=mujoco.MjData(m)
        from mujoco import viewer
        def keys(key):
            nonlocal labels
            if key==76:labels=not labels
            if key in (81,256):closed.set()
        handle=viewer.launch_passive(m,d,show_left_ui=False,show_right_ui=False,key_callback=keys)
        prior_label_flag=True
        while not stop.is_set() and handle.is_running():
            try:
                for _ in range(8):snap=queue.get_nowait()
            except Empty:pass
            if snap.status=='STOPPED':break
            mujoco.mj_setState(m,d,np.asarray(snap.data['integration_state']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
            difference=max(float(np.max(abs(d.qpos-snap.data['qpos']))),float(np.max(abs(d.qvel-snap.data['qvel']))),float(np.max(abs(d.ctrl-snap.data['ctrl']))));assert difference==0
            if snap.data.get('visual_labels',True)!=prior_label_flag:labels=snap.data['visual_labels'];prior_label_flag=labels
            with handle.lock():
                if last_camera!=snap.data['camera_index']:
                    cam=camera();handle.cam.lookat[:]=cam.lookat;handle.cam.distance=cam.distance;handle.cam.azimuth=[135,45,210][snap.data['camera_index']];handle.cam.elevation=[-19,-10,-35][snap.data['camera_index']];last_camera=snap.data['camera_index']
                handle.user_scn.ngeom=0
                if labels:
                    for i,p in enumerate(anchors(m,d,mapping)):
                        g=handle.user_scn.geoms[i];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),p,np.eye(3).ravel(),rgba(i));g.label=f'Leg{i+1}'
                    handle.user_scn.ngeom=6
            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,None,f'DYNAMIC CONTROL DEMONSTRATION\nLIVE STATE MIRROR / {snap.status}',f'Actuator-order colours | L labels | Q quit\nPhysical time {snap.time_s:.3f} s | PARTIAL PROXY QUERY'))
            handle.sync(state_only=True);row=dict(step=snap.step,time_s=snap.time_s,max_state_difference=difference);rows.append(row);samples+=1;maximum=max(maximum,difference)
            if journal:journal.write(json.dumps(row)+'\n');journal.flush()
            time.sleep(.04)
    finally:
        if handle is not None and handle.is_running():handle.close()
        if journal:journal.close()
        closed.set();write_json(Path(config['folder'])/'enhanced_mirror_validation.json',dict(samples=samples,maximum_state_difference=maximum,mj_step_calls=0,mapping=mapping,rows=list(rows),rows_are_tail=bool(config.get('stream_logs')),independent_display_model=True))
