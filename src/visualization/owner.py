"""The only live physics owner. Qt and native mirror never receive this MjData."""
import json,time,traceback,faulthandler
from pathlib import Path
from queue import Empty
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ..model import ROOT,write_json,ids,site_gap,equality_residual
from ..phase1.plant import create_plant,plant_metadata,platform_state
from ..phase1.controllers import Controller
from ..phase1.trajectory import Trajectory,Reference,protocol
from ..phase1.executor import check_safety
from .source import ReadOnlyProxy
from .motion import SmoothPlan,target_spec,validate_target
from .telemetry import Snapshot,publish_latest
from .recording import Recorder

def interface_config():return json.loads((ROOT/'configs/phase1_5_interface.json').read_text(encoding='utf-8'))

class PhysicsOwner:
    def __init__(self,config,commands=None,telemetry=None,mirror=None,stop=None,mirror_closed=None):
        self.config=config;self.options=interface_config();self.commands=commands;self.telemetry=telemetry;self.mirror=mirror;self.stop=stop;self.mirror_closed=mirror_closed
        self.nominal=create_plant('nominal');self.model=create_plant(config.get('condition','nominal'));self.data=mujoco.MjData(self.model);mujoco.mj_forward(self.model,self.data)
        self.controller=Controller(self.nominal,config.get('controller','C0'),.001);self.geometry=self.controller.geometry;self.qadr,self.dof=ids(self.model);self.query=ReadOnlyProxy();self.proxy=self.query.evaluate(self.data.qpos)
        self.home=Trajectory(self.nominal,'home',ROOT/'results/phase1_preflight/references');self.ref=self.home.at(0);self.hold_ref=self.ref;self.accepted=target_spec();self.pending=None;self.pending_preset=None;self.active=None;self.active_start=0.;self.paused=False;self.status='HOLDING';self.failure=None;self.protect_end=None;self.camera_index=0;self.last_message='Nominal control ready';self.n=0;self.episode=config.get('episode',0)
        self.locked=config.get('condition','nominal')!='nominal'
        if self.locked and not config.get('diagnostic_asymmetric'):raise ValueError('Mismatch teleoperation is locked; explicit diagnostic preflight only')
        self.recorder=Recorder(config['folder'],config.get('record',False));write_json(Path(config['folder'])/'config.json',config);write_json(Path(config['folder'])/'plant.json',plant_metadata(self.model,config.get('condition','nominal')))
        self.state=np.zeros(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION));self.last_info=dict(condition=self.controller.condition,integral=np.zeros(6),clipped=False);self.last_force=np.zeros(6);self.last_snapshot=None;self.auto_resume=None;self.script=sorted(config.get('events',[]),key=lambda x:x['at_step']);self.script_index=0;self.reason='budget_complete'
        self.recorder.event(dict(type='episode_started',episode=self.episode,time_s=0,source_mode='DYNAMIC CONTROL DEMONSTRATION',keyboard_scope='Control panel only',collision='PARTIAL PROXY QUERY'))
    def event(self,event_type,**data):self.recorder.event(dict(type=event_type,episode=self.episode,step=self.n,time_s=float(self.data.time),**data))
    def hold(self,reason='user_hold',failed=False):
        cancelled=dict(self.accepted);p,r,_=platform_state(self.model,self.data);angles=Rotation.from_matrix(r@self.geometry.home_rotation.T).as_euler('xyz',degrees=True);self.accepted=target_spec((p[2]-self.geometry.home_position[2])*1000,angles[0],angles[1]);self.hold_ref=Reference(self.data.qpos[self.qadr].copy(),np.zeros(6),p,r,np.zeros(6),self.ref.ff.copy());self.ref=self.hold_ref;self.active=None;self.pending=None;self.pending_preset=None;self.controller.reset();self.status='FAILED' if failed else 'HOLDING'
        if failed:self.failure=reason;self.protect_end=self.data.time+.2
        self.last_message=reason;self.event('protect_hold' if failed else 'hold',reason=reason,actual_qpos=self.data.qpos.tolist(),cancelled_target=cancelled)
    def command(self,cmd):
        kind=cmd['kind'];payload=cmd.get('payload',{});self.event('input',kind=kind,payload=payload,source=cmd.get('source','unknown'))
        if kind in ('quit','reset'):self.reason='reset_requested' if kind=='reset' else 'user_quit';return False
        if kind=='record':self.recorder.enabled=bool(payload.get('enabled',not self.recorder.enabled));self.last_message='Recording '+str(self.recorder.enabled);return True
        if kind=='camera':self.camera_index=(self.camera_index+1)%3;return True
        if kind=='pause':
            self.paused=not self.paused
            if payload.get('auto_resume_after_s'):self.auto_resume=time.perf_counter()+payload['auto_resume_after_s']
            return True
        if kind=='controller':self.event('rejected',reason='Controller changes only with explicit Reset/new episode');self.last_message='Select next controller, then Reset';return True
        if self.failure:self.event('rejected',reason='Failure latched; Reset required');return True
        if self.locked:self.event('rejected',reason='Diagnostic mismatch: free teleoperation disabled');return True
        if kind=='hold':self.hold();return True
        if kind=='preset':
            if payload['name'] not in protocol()['nominal_trajectories']:raise ValueError('Invalid preset')
            self.pending_preset=payload['name'];self.pending=target_spec();return True
        if kind=='home':self.pending=target_spec();self.pending_preset=None;return True
        if kind in ('target','increment'):
            spec=dict(self.pending or self.accepted)
            if kind=='target':spec.update(payload)
            else:
                for key,value in payload.items():spec[key]+=value
            try:self.pending=validate_target(spec);self.last_message='Target queued; current validated segment will finish'
            except ValueError as exc:self.event('target_rejected',reason=str(exc));self.last_message=str(exc)
        return True
    def maybe_plan(self):
        if self.active is not None or self.pending is None or self.paused or self.failure:return
        spec=self.pending;self.pending=None;self.last_message='Validating target; simulation clock temporarily frozen';self.publish()
        try:
            if max(abs(np.array(list(spec.values()))-np.array(list(self.accepted.values()))))>1e-12 or max(abs(self.ref.twist))>1e-12:
                plan=SmoothPlan(self.nominal,self.geometry,self.ref,spec,self.query);self.active=plan;self.active_start=float(self.data.time);self.accepted=spec;self.status='RUNNING';self.event('plan_accepted',goal=spec,audit=plan.audit)
            elif self.pending_preset is not None:
                name=self.pending_preset;self.pending_preset=None;self.active=Trajectory(self.nominal,name,ROOT/'results/phase1_preflight/references');self.active_start=float(self.data.time);self.status='RUNNING';self.event('preset_started',name=name)
            self.last_message='Validated target active'
        except Exception as exc:self.hold('reference_preflight:'+str(exc),failed=True)
    def reference(self,t):
        if self.active is None:return self.hold_ref
        local=t-self.active_start;ref=self.active.at(local)
        if local>=self.active.duration-1e-10:
            self.ref=ref;self.hold_ref=Reference(ref.q,np.zeros(6),ref.p,ref.r,np.zeros(6),ref.ff);self.active=None;self.status='HOLDING'
            angles=Rotation.from_matrix(ref.r@self.geometry.home_rotation.T).as_euler('xyz',degrees=True);self.accepted=target_spec((ref.p[2]-self.geometry.home_position[2])*1000,angles[0],angles[1])
            if self.pending_preset is not None:self.pending=target_spec()
        return ref
    def snapshot(self):
        m,d=self.model,self.data;p,r,v=platform_state(m,d);g=site_gap(m,d);a=np.array([np.linalg.norm(Rotation.from_matrix(d.site(f'site_c1_{i+12}').xmat.reshape(3,3)@d.site(f'site_upper_{i}').xmat.reshape(3,3).T).as_rotvec()) for i in range(1,7)])
        mujoco.mj_getState(m,d,self.state,mujoco.mjtState.mjSTATE_INTEGRATION)
        passive=d.qpos[self.controller.chain_addresses[:,[0,1,3,4,5]]].copy()
        data=dict(controller=self.controller.name,trajectory=self.active.name if isinstance(self.active,Trajectory) else 'manual smooth target' if self.active else 'restricted hold',condition=self.config.get('condition','nominal'),p_ref=tuple(self.ref.p),p=tuple(p),rpy_ref=tuple(Rotation.from_matrix(self.ref.r).as_euler('xyz',degrees=True)),rpy=tuple(Rotation.from_matrix(r).as_euler('xyz',degrees=True)),q_ref=tuple(self.ref.q),q=tuple(d.qpos[self.qadr]),dq=tuple(d.qvel[self.dof]),actuatorpos=tuple(d.actuator_length[:6]),f_cmd=tuple(self.last_force),force=tuple(d.actuator_force[:6]),passive=tuple(tuple(row) for row in passive),qpos=tuple(d.qpos),qvel=tuple(d.qvel),ctrl=tuple(d.ctrl),integration_state=self.state.copy(),weld_t=float(max(g)),weld_r=float(max(a)),proxy_gap=self.proxy['minimum_clearance_m'],pose_error_mm=float(np.linalg.norm(p-self.ref.p)*1000),angle_error_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(self.ref.r@r.T).as_rotvec()))),leg_error_mm=float(max(abs(d.qpos[self.qadr]-self.ref.q))*1000),force_peak_N=float(max(abs(d.actuator_force[:6]))),force_saturated=bool(max(abs(d.actuator_force[:6]))>=400-1e-9),jacobian_condition=float(self.last_info['condition']),accepted_target=dict(self.accepted),pending_target=self.pending,message=self.last_message,failure=self.failure,recording=self.recorder.enabled,camera_index=self.camera_index,collision_label='PARTIAL PROXY QUERY')
        return Snapshot(self.episode,self.n,float(d.time),'DYNAMIC CONTROL DEMONSTRATION','PAUSED' if self.paused and not self.failure else self.status,data)
    def publish(self):
        snap=self.snapshot();self.last_snapshot=snap;publish_latest(self.telemetry,snap);publish_latest(self.mirror,snap);return snap
    def step(self):
        m,d=self.model,self.data;next_time=float(d.time+.001);self.ref=self.reference(next_time);p,r,v=platform_state(m,d)
        if self.failure:
            force=np.clip(self.ref.ff+5000*(self.ref.q-d.qpos[self.qadr])-200*d.qvel[self.dof],-400,400);self.controller.reset()
        else:force,self.last_info=self.controller.command(d.qpos,d.qvel,p,r,v,self.ref,next_time)
        self.last_force=force;d.ctrl[:6]=force;d.ctrl[6:]=0;previous=float(d.time);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self.n+=1
        if not np.isclose(d.time,previous+.001,atol=1e-10):raise RuntimeError('Unexpected live time mutation')
        if self.n%20==0:self.proxy=self.query.evaluate(d.qpos)
        snap=self.snapshot();p,r,_=platform_state(m,d);reason=check_safety(m,d,p,r,self.geometry,snap.data['weld_t'],snap.data['weld_r'],snap.data['jacobian_condition'],self.proxy,self.controller.config)
        if reason and not self.failure:self.hold(reason,failed=True);snap=self.snapshot()
        row=dict(episode=self.episode,step=self.n,time_s=float(d.time),status=snap.status,controller=self.controller.name,source_mode=snap.mode,position_error_mm=snap.data['pose_error_mm'],orientation_error_deg=snap.data['angle_error_deg'],max_leg_error_mm=snap.data['leg_error_mm'],weld_translation_m=snap.data['weld_t'],weld_angle_rad=snap.data['weld_r'],proxy_clearance_m=snap.data['proxy_gap'],failure=self.failure or '',recording=self.recorder.enabled)
        for key in ['p_ref','p','rpy_ref','rpy','q_ref','q','dq','actuatorpos','f_cmd','force','qpos','qvel','ctrl']:
            for j,value in enumerate(snap.data[key]):row[f'{key}_{j}']=float(value)
        for leg in range(6):
            for j,value in enumerate(snap.data['passive'][leg]):row[f'passive_{leg}_{j}']=float(value)
        self.recorder.append(row,self.state)
        if self.n%40==0:self.recorder.capture(m,d,snap,self.n-1)
        if self.n%10==0:self.publish()
    def run(self):
        wall=time.perf_counter();self.publish();run=True;last_wall_publish=wall
        try:
            while run and self.n<round(self.config.get('max_sim_seconds',60)*1000) and time.perf_counter()-wall<self.config.get('max_wall_seconds',600):
                if self.stop is not None and self.stop.is_set():self.reason='user_quit';break
                if self.mirror_closed is not None and self.mirror_closed.is_set():self.reason='native_viewer_closed';break
                while self.script_index<len(self.script) and self.script[self.script_index]['at_step']<=self.n:
                    event=self.script[self.script_index];self.script_index+=1;run=self.command(dict(event,source=event.get('source','fixed-script')))
                    if not run:break
                if self.commands is not None:
                    for _ in range(64):
                        try:cmd=self.commands.get_nowait()
                        except Empty:break
                        run=self.command(cmd)
                        if not run:break
                if not run:break
                if self.auto_resume is not None and time.perf_counter()>=self.auto_resume:self.paused=False;self.auto_resume=None;self.event('auto_resume')
                if self.paused or (self.failure and self.data.time>=self.protect_end-1e-10):
                    if self.failure and self.config.get('exit_on_failure',False):self.reason='protected_failure';break
                    if time.perf_counter()-last_wall_publish>.05:self.publish();last_wall_publish=time.perf_counter()
                    time.sleep(.005);continue
                self.maybe_plan();self.step()
                if self.config.get('pace',True):
                    delay=wall+self.n*.001-time.perf_counter()
                    if delay>0:time.sleep(min(delay,.001))
            if self.reason=='budget_complete' and self.failure:self.reason='protected_failure'
            elif self.reason=='budget_complete' and self.n<round(self.config.get('max_sim_seconds',60)*1000):self.reason='wall_budget_complete'
        except Exception as exc:self.failure=self.failure or ('owner_exception:'+str(exc));self.reason='owner_exception';self.event('exception',traceback=traceback.format_exc())
        finally:
            self.status='FAILED' if self.failure else 'STOPPED'
            try:self.publish()
            except Exception as exc:self.event('final_telemetry_error',reason=repr(exc))
            self.recorder.close(dict(episode=self.episode,physics_steps=self.n,simulation_time_s=float(self.data.time),wall_seconds=time.perf_counter()-wall,terminal_reason=self.reason,failure=self.failure,controller=self.controller.name,condition=self.config.get('condition','nominal'),mujoco_warnings=int(sum(w.number for w in self.data.warning)),no_live_qpos_assignment=True));
        return self.reason

def owner_main(config,commands,telemetry,mirror,stop,mirror_closed):
    faulthandler.enable(all_threads=True)
    for q in [telemetry,mirror]:
        if q is not None and hasattr(q,'cancel_join_thread'):q.cancel_join_thread()
    try:PhysicsOwner(config,commands,telemetry,mirror,stop,mirror_closed).run()
    except Exception:
        Path(config['folder']).mkdir(parents=True,exist_ok=True);write_json(Path(config['folder'])/'startup_failure.json',dict(traceback=traceback.format_exc()))
        raise
