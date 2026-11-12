"""Read-only native WINDOW A, a separate process. Never calls mj_step.
All displayed integration states originate from the actual physics owner.
Native GUI inputs can affect only this disposable display MjData.
"""
import time,json,hashlib,faulthandler
from pathlib import Path
from queue import Empty
import numpy as np,mujoco
from ..model import write_json
from ..phase1.plant import create_plant
from ..rendering import camera

def mirror_main(config,queue,stop,closed):
    faulthandler.enable(all_threads=True);handle=None;rows=[];last_camera=None;last_label=None;last_label_wall=0.;ignored=0;quit_keys=[]
    try:
        first=None;start=time.perf_counter()
        while first is None and not stop.is_set() and time.perf_counter()-start<60:
            try:first=queue.get(timeout=.1)
            except Empty:pass
        if first is None:return
        m=create_plant(config.get('condition','nominal'));display=mujoco.MjData(m)
        from mujoco import viewer
        def key_callback(key):
            if key in (81,256):quit_keys.append(int(key));closed.set()
        handle=viewer.launch_passive(m,display,show_left_ui=False,show_right_ui=False,key_callback=key_callback);snap=first
        while not stop.is_set() and handle.is_running():
            try:
                for _ in range(8):snap=queue.get_nowait()
            except Empty:pass
            if snap.status=='STOPPED':break
            mujoco.mj_setState(m,display,np.asarray(snap.data['integration_state']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,display)
            desired_q=np.asarray(snap.data['qpos']);difference=float(np.max(abs(display.qpos-desired_q)));assert difference<1e-12
            velocity_difference=float(np.max(abs(display.qvel-np.asarray(snap.data['qvel']))));ctrl_difference=float(np.max(abs(display.ctrl-np.asarray(snap.data['ctrl']))));pose_difference=float(np.max(abs(display.site('upper_center').xpos-np.asarray(snap.data['p']))))
            assert max(velocity_difference,ctrl_difference,pose_difference)<1e-12
            if snap.data['camera_index']!=last_camera:
                with handle.lock():
                    cam=camera();handle.cam.lookat[:]=cam.lookat;handle.cam.distance=cam.distance;handle.cam.azimuth=[135,45,210][snap.data['camera_index']];handle.cam.elevation=[-19,-10,-35][snap.data['camera_index']]
                last_camera=snap.data['camera_index']
            label=(snap.status,snap.data['controller'],snap.data['failure'],int(snap.time_s*5))
            if label!=last_label:
                text=f'{snap.mode}\nLIVE STATE MIRROR / {snap.status}'
                d=snap.data
                detail=f'{d["controller"]} | sim {snap.time_s:.3f}s | separate physical owner\nZ ref/actual {d["p_ref"][2]*1000:.3f}/{d["p"][2]*1000:.3f} mm\nRoll ref/actual {d["rpy_ref"][0]:.3f}/{d["rpy"][0]:.3f} deg\nPitch ref/actual {d["rpy_ref"][1]:.3f}/{d["rpy"][1]:.3f} deg\nPARTIAL PROXY QUERY\n'+(str(d['failure']) if d['failure'] else 'Control keys only in Control Panel; mouse = camera')
                if handle.is_running():handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,None,text,detail))
                last_label=label
            displayed_hash=hashlib.sha256(display.qpos.copy().tobytes()).hexdigest()
            handle.sync(state_only=True)
            # User reset/perturb UI is never transmitted to the physics process.
            if not np.array_equal(display.qpos,desired_q) or not np.array_equal(display.ctrl,np.asarray(snap.data['ctrl'])):ignored+=1
            rows.append(dict(step=snap.step,time_s=snap.time_s,episode=snap.episode,qpos_max_difference=difference,qvel_max_difference=velocity_difference,ctrl_max_difference=ctrl_difference,platform_position_max_difference=pose_difference,owner_qpos_sha256=hashlib.sha256(desired_q.tobytes()).hexdigest(),display_qpos_sha256=displayed_hash))
            time.sleep(.025)
    finally:
        if handle is not None and handle.is_running():handle.close()
        closed.set();write_json(Path(config['folder'])/'mirror_validation.json',dict(source='LIVE ACTUAL OWNER STATE MIRROR',mj_step_calls=0,samples=len(rows),max_qpos_difference=max((r['qpos_max_difference'] for r in rows),default=0),native_quit_keys=quit_keys,native_input_mutations_ignored=ignored,rows=rows,physics_model_not_shared_with_gui=True))
