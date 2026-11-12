"""Bounded 2048-step fixed-load equivalence with Qt/native display and actual recording."""
import _bootstrap
import time,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from PySide6 import QtCore,QtWidgets
from src.model import ROOT,write_json
from src.control.phase2c0_experiment import PhysicalOwner as DisturbanceOwner
from src.disturbance.recording import install_colored_capture
from src.visualization.enhanced.mirror import mirror_main
from src.visualization.enhanced.dashboard import EnhancedDashboard

class ObserverSession:
    def __init__(self,config,telemetry,commands):self.base=self.current=config;self.telemetry=telemetry;self.commands=commands;self.stopped=False;self.dropped_commands=0
    def poll(self):return not self.stopped
    def request_stop(self):self.stopped=True
    def send(self,kind,payload=None,source='observer-validation'):
        # Fixed-input validation: only user quit can interrupt; no task commands admitted.
        if kind=='quit':self.request_stop()
        return False

def main():
    root=ROOT/'results/runs'/f'phase2c0_observer_{time.time_ns()}';root.mkdir(parents=True)
    parent=Path(json.loads((ROOT/'results/phase2b1_latest_run.json').read_text(encoding='utf-8'))['root']);chosen=json.loads((parent/'controller_parameters_frozen.json').read_text(encoding='utf-8'));task_model=json.loads((parent/'nominal_task_model.json').read_text(encoding='utf-8'))
    cfg=dict(yaw_policy=dict(kind='observer',gain=1.),actuator_dynamics=dict(enabled=True,tau_s=.002),stage='gui_validation',action=dict(kind='lead',space='task',frequency=20),controller='C3',task_model=task_model,bandwidth_rad_s=chosen['bandwidth_rad_s'],compensation_gain=chosen['compensation_gain'],trajectory='home',condition='nominal',record=False,pace=False,max_sim_seconds=2.048,max_wall_seconds=60,exit_on_failure=True,disturbance=dict(start_step=512,stop_step=1024,wrench=[0,0,-2,0,0,0]))
    baseline=DisturbanceOwner(dict(cfg,folder=str(root/'off')));baseline.run()
    context=mp.get_context('spawn');mirror=context.Queue(8);telemetry=context.Queue(64);commands=context.Queue(64);stop=context.Event();closed=context.Event()
    for q in (mirror,telemetry,commands):q.cancel_join_thread()
    config=dict(cfg,folder=str(root/'on'),record=True,close_on_complete=True);owner=DisturbanceOwner(config,commands,telemetry,mirror,stop,closed);install_colored_capture(owner.recorder,owner.model)
    viewer=context.Process(target=mirror_main,args=(config,mirror,stop,closed));viewer.start()
    app=QtWidgets.QApplication([]);session=ObserverSession(config,telemetry,commands);w=EnhancedDashboard(session);w.setWindowTitle('Phase2C0 Fixed-input Observer Validation');w.show();timer=QtCore.QTimer();wall=time.perf_counter();saved=False;interrupted=False
    def finish():
        nonlocal interrupted
        timer.stop();stop.set();owner.status='STOPPED';owner.publish();owner.recorder.close(dict(physics_steps=owner.n,simulation_time_s=float(owner.data.time),failure=owner.failure,observer_interrupted=interrupted));w.final_close=True;w.close();app.quit()
    def tick():
        nonlocal saved,interrupted
        if session.stopped or closed.is_set() or time.perf_counter()-wall>90:interrupted=True;finish();return
        for _ in range(min(10,2048-owner.n)):owner.step()
        if owner.n>=1000 and not saved:w.grab().save(str(ROOT/'assets/plots/phase2c0_observer_validation.png'));saved=True
        if owner.n>=2048:finish()
    timer.timeout.connect(tick);timer.start(1);app.exec();stop.set();viewer.join(15)
    if viewer.is_alive():raise RuntimeError('Native display did not close normally')
    with np.load(root/'off/integration_states.npz') as a,np.load(root/'on/integration_states.npz') as b:
        equal=np.array_equal(a['states'],b['states']);delta=float(np.max(abs(a['states']-b['states']))) if a['states'].shape==b['states'].shape else None
    result=dict(root=str(root),physics_steps_each=owner.n,bitwise_equal=equal,maximum_integration_state_difference=delta,gui_qt_and_native=True,actual_recording=True,interrupted=interrupted,mirror_exitcode=viewer.exitcode,recording_errors=owner.recorder.render_failures)
    write_json(ROOT/'results/phase2c0_observer_validation.json',result);print(json.dumps(result),flush=True);assert equal and owner.n==2048 and not interrupted and not result['recording_errors']
if __name__=='__main__':mp.freeze_support();main()
