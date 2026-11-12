"""Bounded Phase2A trial: original controller/safety; separate external-load input only."""
import time,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ..model import ROOT,write_json,ids,site_gap,equality_residual
from ..phase1.plant import platform_state,plant_metadata
from ..phase1.trajectory import Trajectory
from ..phase1.executor import check_safety
from ..visualization.enhanced.owner import EnhancedOwner
from .disturbance_profiles import DisturbanceProfile,generalized_external

class DisturbanceOwner(EnhancedOwner):
    def __init__(self,config,*channels):
        self.profile=DisturbanceProfile(**config['disturbance']);self.external_audit=None
        super().__init__(config,*channels)
        self.site_id=self.model.site('upper_center').id
        self.active=Trajectory(self.nominal,config.get('trajectory','home'),ROOT/'results/phase1_preflight/references');self.active_start=0.;self.status='RUNNING'
        self.initial_qpos=self.data.qpos.copy();self.initial_qvel=self.data.qvel.copy()
        write_json(Path(config['folder'])/'disturbance_profile.json',self.profile.description())
        write_json(Path(config['folder'])/'initial_state.json',dict(qpos=self.initial_qpos.tolist(),qvel=self.initial_qvel.tolist(),integration_state=self.snapshot().data['integration_state'].tolist()))
    def run(self):
        from .runtime import run_disturbance
        return run_disturbance(self)
    def snapshot(self):
        from dataclasses import replace
        s=super().snapshot()
        return replace(s,data=dict(s.data,external_world=self.external_audit['world'].tolist() if self.external_audit else [0.]*6))
    def step(self):
        m,d=self.model,self.data;next_time=float(d.time+.001);self.ref=self.reference(next_time);p,r,v=platform_state(m,d)
        if self.failure:
            raw=self.ref.ff+5000*(self.ref.q-d.qpos[self.qadr])-200*d.qvel[self.dof];force=np.clip(raw,-400,400);self.controller.reset();self.last_info.update(raw=raw,integral=np.zeros(6),clipped=bool(np.any(raw!=force)))
        else:
            try:force,self.last_info=self.controller.command(d.qpos,d.qvel,p,r,v,self.ref,next_time)
            except Exception as exc:
                self.hold('controller_rejected:'+str(exc),failed=True)
                raw=self.ref.ff+5000*(self.ref.q-d.qpos[self.qadr])-200*d.qvel[self.dof];force=np.clip(raw,-400,400);self.last_info.update(raw=raw,integral=np.zeros(6),clipped=bool(np.any(raw!=force)))
        self.last_force=force;d.ctrl[:6]=force;d.ctrl[6:]=self.profile.command(self.n,d.site_xmat[self.site_id].reshape(3,3));self.external_audit=generalized_external(m,d,d.ctrl[6:]);self.command_rotation=d.site_xmat[self.site_id].copy();self.load_position=d.site_xpos[self.site_id].copy()
        if self.n in (self.profile.start_step,self.profile.stop_step):self.event('load_on' if self.n==self.profile.start_step else 'load_off',input_step=self.n,command_local=d.ctrl[6:].tolist(),wrench_world=self.external_audit['world'].tolist(),trajectory=self.config.get('trajectory','home'),reference_twist=self.ref.twist.tolist(),reference_is_moving=bool(np.linalg.norm(self.ref.twist)>1e-8))
        previous=float(d.time);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self.n+=1
        if not np.isclose(d.time,previous+.001,atol=1e-10):raise RuntimeError('Unexpected live time mutation')
        if self.n%20==0:self.proxy=self.query.evaluate(d.qpos)
        snap=self.snapshot();p,r,_=platform_state(m,d);reason=check_safety(m,d,p,r,self.geometry,snap.data['weld_t'],snap.data['weld_r'],snap.data['jacobian_condition'],self.proxy,self.controller.config)
        if reason and not self.failure:self.hold(reason,failed=True);snap=self.snapshot()
        row=dict(episode=self.episode,step=self.n,time_s=float(d.time),status=snap.status,controller=self.controller.name,source_mode=snap.mode,position_error_mm=snap.data['pose_error_mm'],orientation_error_deg=snap.data['angle_error_deg'],max_leg_error_mm=snap.data['leg_error_mm'],weld_translation_m=snap.data['weld_t'],weld_angle_rad=snap.data['weld_r'],proxy_clearance_m=snap.data['proxy_gap'],failure=self.failure or '',recording=self.recorder.enabled)
        for key in ['p_ref','p','rpy_ref','rpy','q_ref','q','dq','actuatorpos','f_cmd','force','qpos','qvel','ctrl']:
            for j,value in enumerate(snap.data[key]):row[f'{key}_{j}']=float(value)
        for leg in range(6):
            for j,value in enumerate(snap.data['passive'][leg]):row[f'passive_{leg}_{j}']=float(value)
        row.update(input_step=self.n-1,load_active=self.profile.active(self.n-1),generalized_mapping_error=self.external_audit['maximum_error'],command_clipped=self.last_info.get('clipped',False),native_force_saturated=bool(max(abs(d.actuator_force[:6]))>=400-1e-9),warning_count=int(sum(w.number for w in d.warning)),native_contacts=int(d.ncon),jacobian_condition=float(self.last_info['condition']),proxy_distance_lower_bound=self.proxy['distance_is_lower_bound'])
        for key,values in [('external_ctrl',d.ctrl[6:]),('external_world',self.external_audit['world']),('external_generalized',self.external_audit['native']),('external_rotation',self.command_rotation),('load_position',self.load_position),('raw_force',self.last_info.get('raw',self.last_force)),('integral',self.last_info.get('integral',np.zeros(6))),('twist_actual',platform_state(m,d)[2]),('twist_ref',self.ref.twist)]:
            for j,value in enumerate(values):row[f'{key}_{j}']=float(value)
        self.recorder.append(row,self.state)
        if self.n%40==0:self.recorder.capture(m,d,snap,self.n-1)
        if self.n%10==0:self.publish()
