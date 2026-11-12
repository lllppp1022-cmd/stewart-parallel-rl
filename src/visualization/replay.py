"""Historical display only. This module never integrates or calls a controller."""
from pathlib import Path
import json
import numpy as np,pandas as pd,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from .telemetry import Snapshot
from ..model import ROOT

class ReplaySource:
    def __init__(self,folder):
        self.folder=Path(folder);self.df=pd.read_csv(self.folder/'physics.csv.gz',float_precision='round_trip');self.frame_rows=[];self.reader=None;self.renderer=None;self.display=None;self.model=None;self.last_frame=-1;self.last_image=None
        metadata=self.folder/'frames.jsonl'
        if metadata.exists() and (self.folder/'actual_rgb.mp4').exists():
            self.frame_rows=[json.loads(x) for x in metadata.read_text(encoding='utf-8').splitlines() if x];self.reader=imageio.get_reader(self.folder/'actual_rgb.mp4')
        self.origin='ACTUAL EXECUTION CAPTURE' if self.reader else 'RECORDED STATE REPLAY'
        self.times=np.array([x['time_s'] for x in self.frame_rows]) if self.frame_rows else self.df.time_s.to_numpy()[::40]
        if not len(self.times):raise ValueError('Episode contains no physical samples; inspect startup_failure.json / summary.json')
        self.index=0;self.state_archive=None;self.snapshot_cache={};self.sample_indices=np.arange(0,len(self.df),10);self.sample_times=self.df.time_s.to_numpy()[self.sample_indices]
    def close(self):
        if self.reader:self.reader.close()
        if self.renderer:self.renderer.close()
        if self.state_archive:self.state_archive.close()
    def at(self,index):
        index=int(np.clip(index,0,len(self.times)-1));self.index=index;t=float(self.times[index]);i=min(int(np.searchsorted(self.df.time_s.to_numpy(),t+1e-10,side='right')-1),len(self.df)-1);row=self.df.iloc[max(i,0)]
        return self.snapshot(row)
    def snapshot(self,row):
        if 'p_0' in row.index:
            vec=lambda key,n:tuple(float(row[f'{key}_{j}']) for j in range(n))
            data=dict(p_ref=vec('p_ref',3),p=vec('p',3),rpy_ref=vec('rpy_ref',3),rpy=vec('rpy',3),q_ref=vec('q_ref',6),q=vec('q',6),dq=vec('dq',6),actuatorpos=vec('actuatorpos',6),f_cmd=vec('f_cmd',6),force=vec('force',6),qpos=vec('qpos',43),qvel=vec('qvel',42),ctrl=vec('ctrl',12),passive=tuple(tuple(float(row[f'passive_{leg}_{j}']) for j in range(5)) for leg in range(6)),pose_error_mm=float(row.position_error_mm),angle_error_deg=float(row.orientation_error_deg),leg_error_mm=float(row.max_leg_error_mm),force_peak_N=max(abs(x) for x in vec('force',6)),force_saturated=max(abs(x) for x in vec('force',6))>=400-1e-9,weld_t=float(row.weld_translation_m),weld_r=float(row.weld_angle_rad),proxy_gap=float(row.proxy_clearance_m),controller=str(row.controller),trajectory='Recorded interaction',condition='Recorded episode',message=self.origin,failure=str(row.failure) if pd.notna(row.failure) else None,recording=False,collision_label='PARTIAL PROXY QUERY')
            return Snapshot(int(row.episode),int(row.step),float(row.time_s),'RECORDED STATE REPLAY',str(row.status),data)
        vec=lambda key,n:tuple(float(row[f'{key}_{j}']) for j in range(1,n+1));qpos=vec('qpos',43);qref=vec('slide_ref_m',6);q=vec('slide_actual_m',6)
        qr=np.array(vec('quat_ref_wxyz',4));qa=np.array(vec('quat_actual_wxyz',4));joint_meta=json.loads((ROOT/'results/phase0_model_inspection.json').read_text(encoding='utf-8'))['joints'];lookup={x['name']:x['qpos_address'] for x in joint_meta};passive=[]
        for leg in range(1,7):passive.append(tuple(qpos[lookup[n]] for n in [f'joint_c2_{leg}',f'joint_lb_{leg}',f'joint_c1_{leg+6}',f'joint_c2_{leg+6}',f'joint_c1_{leg+12}']))
        data=dict(p_ref=vec('p_ref_m',3),p=vec('p_actual_m',3),rpy_ref=tuple(Rotation.from_quat(qr[[1,2,3,0]]).as_euler('xyz',degrees=True)),rpy=tuple(Rotation.from_quat(qa[[1,2,3,0]]).as_euler('xyz',degrees=True)),q_ref=qref,q=q,dq=vec('dslide_actual_mps',6),actuatorpos=q,f_cmd=vec('ctrl_leg_N',6),force=vec('F_actual_N',6),qpos=qpos,qvel=vec('qvel',42),ctrl=vec('ctrl_leg_N',6)+vec('external_ctrl',6),passive=tuple(passive),pose_error_mm=float(row.position_error_m*1000),angle_error_deg=float(np.rad2deg(row.orientation_error_rad)),leg_error_mm=max(abs(a-b) for a,b in zip(qref,q))*1000,force_peak_N=max(abs(x) for x in vec('F_actual_N',6)),force_saturated=bool(row.native_force_saturated),weld_t=float(row.max_weld_translation_m),weld_r=float(row.max_weld_angle_rad),proxy_gap=float(row.minimum_proxy_clearance_m),controller=str(row.controller),trajectory=str(row.trajectory),condition=str(row.condition),message='RECORDED STATE REPLAY: no on-site RGB for this individual run',failure='Recorded failure' if row.failed else None,recording=False,collision_label='PARTIAL PROXY QUERY')
        return Snapshot(0,int(row.step),float(row.time_s),'RECORDED STATE REPLAY','FAILED' if row.failed else 'RUNNING',data)
    def history(self,seconds=20):
        t=float(self.times[self.index]);lo=np.searchsorted(self.sample_times,t-seconds);hi=np.searchsorted(self.sample_times,t+1e-10,side='right');indices=self.sample_indices[lo:hi]
        result=[]
        for i in indices:
            if i not in self.snapshot_cache:self.snapshot_cache[i]=self.snapshot(self.df.iloc[i])
            result.append(self.snapshot_cache[i])
        return result
    def image(self,snapshot):
        if self.reader:
            if self.last_frame!=self.index:self.last_image=self.reader.get_data(int(self.frame_rows[self.index]['frame']));self.last_frame=self.index
            return self.last_image
        import mujoco
        if self.model is None:
            from ..model import load_model
            self.model=load_model();self.model.vis.global_.offwidth=960;self.model.vis.global_.offheight=720;self.display=mujoco.MjData(self.model);self.renderer=mujoco.Renderer(self.model,width=960,height=720)
        self.display.qpos[:]=snapshot.data['qpos'];self.display.qvel[:]=snapshot.data['qvel'];self.display.ctrl[:]=snapshot.data['ctrl'];self.display.time=snapshot.time_s;mujoco.mj_forward(self.model,self.display)
        from ..rendering import camera
        self.renderer.update_scene(self.display,camera=camera());return self.renderer.render().copy()
