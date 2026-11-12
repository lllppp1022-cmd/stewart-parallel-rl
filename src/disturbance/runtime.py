"""Continuous scheduling; the original control/physics step stays untouched."""
import time,traceback
from queue import Empty

def budget_reason(config,steps,elapsed):
    sim=config.get('max_sim_seconds');wall=config.get('max_wall_seconds')
    if sim is not None and steps>=round(sim*1000):return 'budget_complete'
    if wall is not None and elapsed>=wall:return 'wall_budget_complete'
    return None

def run_disturbance(self):
    wall=time.perf_counter();self.publish();run=True;last_wall_publish=wall
    try:
        while run:
            budget=budget_reason(self.config,self.n,time.perf_counter()-wall)
            extending_hold=(budget=='budget_complete' and self.failure and self.protect_end is not None and self.data.time<self.protect_end-1e-10)
            if budget and not extending_hold:self.reason='protected_failure' if self.failure else budget;break
            tick=time.perf_counter()
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
                delay=.001-(time.perf_counter()-tick)
                if delay>0:time.sleep(min(delay,.001))
        if self.reason=='budget_complete' and self.failure:self.reason='protected_failure'

    except Exception as exc:self.failure=self.failure or ('owner_exception:'+str(exc));self.reason='owner_exception';self.event('exception',traceback=traceback.format_exc())
    finally:
        self.status='FAILED' if self.failure else 'STOPPED'
        try:self.publish()
        except Exception as exc:self.event('final_telemetry_error',reason=repr(exc))
        self.recorder.close(dict(episode=self.episode,physics_steps=self.n,simulation_time_s=float(self.data.time),wall_seconds=time.perf_counter()-wall,terminal_reason=self.reason,failure=self.failure,controller=self.controller.name,condition=self.config.get('condition','nominal'),mujoco_warnings=int(sum(w.number for w in self.data.warning)),no_live_qpos_assignment=True));
    return self.reason
