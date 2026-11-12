"""Telemetry/event wrapper only: delegates every force calculation and step unchanged."""
import time,traceback
from pathlib import Path
from dataclasses import replace
from ..owner import PhysicsOwner
from ...phase1.plant import platform_state
from ...model import write_json
from .style import leg_mapping

class EnhancedOwner(PhysicsOwner):
    def __init__(self,*a,**kw):
        self.event_id=0;self.command_id=0;self.pending_id=None;self.active_id=None;self.started=False;self.completed=False;self.labels=True;self.enhancement_ready=False
        super().__init__(*a,**kw)
        if self.config.get('stream_logs'):
            from .streaming import StreamingRecorder
            self.recorder=StreamingRecorder.adopt(self.recorder,self.config.get('state_chunk_steps',2000),self.config.get('video_segment_seconds',60))
        p,r,_=platform_state(self.model,self.data)
        self.home_baseline=dict(source='Explicit episode initial Home, before first mj_step',episode=self.episode,time_s=0.,p_m=p.tolist(),rpy_deg=list(self.snapshot().data['rpy']),slide_m=self.data.qpos[self.qadr].tolist(),home_reference_slide_m=self.home.at(0).q.tolist(),qpos=self.data.qpos.tolist())
        write_json(Path(self.config['folder'])/'home_baseline.json',self.home_baseline);write_json(Path(self.config['folder'])/'leg_mapping.json',leg_mapping(self.model));self.enhancement_ready=True
    def event(self,event_type,**data):
        self.event_id+=1
        if event_type in ('plan_accepted','preset_started'):
            data.update(command_id=self.pending_id,preflight='PASSED',target_after=dict(self.accepted));self.active_id=self.pending_id;self.pending_id=None;self.started=True
        elif event_type=='input':data.update(command_id=self.command_id,target_before=dict(self.pending or self.accepted),preflight='NOT YET CHECKED',execution='SUBMITTED')
        else:data.setdefault('command_id',self.command_id)
        super().event(event_type,event_seq=self.event_id,**data)
    def command(self,cmd):
        if cmd['kind']=='visual_labels':self.labels=bool(cmd.get('payload',{}).get('enabled',not self.labels));self.event('display_labels',enabled=self.labels);return True
        self.command_id+=1;before=dict(self.pending or self.accepted);previous_pending=self.pending_id;kind=cmd['kind'];result=super().command(cmd)
        if kind in ('target','increment','home','preset') and self.pending is not None and not self.failure:
            # A rejected new target can leave an older pending target unchanged.
            last=self.last_message
            if kind not in ('target','increment') or last.startswith('Target queued'):
                if previous_pending is not None and previous_pending!=self.command_id:self.event('target_superseded',superseded_command_id=previous_pending)
                self.pending_id=self.command_id;self.event('target_queued',target_before=before,target_after=dict(self.pending),preflight='NOT YET CHECKED',execution='QUEUED')
        if kind=='record' and not self.recorder.enabled and hasattr(self.recorder,'finish_video'):self.recorder.finish_video()
        if kind=='pause':self.event('pause_state',paused=self.paused)
        return result
    def maybe_plan(self):
        pending=self.pending is not None;candidate_id=self.pending_id;super().maybe_plan()
        if pending and self.failure and self.failure.startswith('reference_preflight:'):self.event('target_rejected',command_id=candidate_id,preflight='FAILED',execution='REJECTED',reason=self.failure)
        if pending and self.pending is None and self.active is None and not self.failure:self.event('target_no_motion',command_id=self.pending_id);self.pending_id=None
    def reference(self,t):
        old=self.active;ref=super().reference(t)
        if old is not None and self.active is None:self.completed=True
        return ref
    def hold(self,*a,**kw):
        interrupted=self.active is not None;super().hold(*a,**kw)
        if interrupted:self.event('motion_aborted',command_id=self.active_id,reason=self.failure or 'user_hold')
        self.pending_id=None;self.started=False
    def step(self):
        super().step()
        if self.started and not self.failure:self.event('motion_started',command_id=self.active_id,execution='FIRST ACTUAL PHYSICS STEP COMPLETE');self.started=False
        if self.completed:
            if self.failure:self.event('motion_aborted',command_id=self.active_id,reason=self.failure)
            else:self.event('motion_completed',command_id=self.active_id,execution='FINAL ACTUAL PHYSICS STEP COMPLETE')
            self.completed=False
    def run(self):
        if self.config.get('stream_logs'):
            from .runtime import run_continuous
            return run_continuous(self)
        return super().run()
    def snapshot(self):
        snap=super().snapshot()
        if self.enhancement_ready:return replace(snap,data=dict(snap.data,home_baseline=self.home_baseline,visual_labels=self.labels))
        return snap

def owner_main(config,*channels):
    for q in channels[1:3]:
        if q is not None and hasattr(q,'cancel_join_thread'):q.cancel_join_thread()
    try:EnhancedOwner(config,*channels).run()
    except Exception:
        Path(config['folder']).mkdir(parents=True,exist_ok=True);write_json(Path(config['folder'])/'startup_failure.json',dict(traceback=traceback.format_exc()));raise
