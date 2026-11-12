"""Gym fixed-action off/on comparison with real WindowsQt/native display."""
import _bootstrap
import json,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from PySide6 import QtCore,QtWidgets
from src.model import ROOT,write_json
from src.rl.stewart_residual_env import StewartResidualEnv,env_config
from src.visualization.enhanced.mirror import mirror_main
from src.visualization.enhanced.dashboard import EnhancedDashboard
from validate_phase2b1_observer import ObserverSession

def main():
    root=ROOT/'results/runs'/f'phase2c1_observer_{time.time_ns()}';root.mkdir(parents=True);task=env_config()['training_tasks'][1];actions=np.random.default_rng(707).uniform(-.6,.6,size=(50,1)).astype(np.float32)
    off=StewartResidualEnv(root/'off',tasks=[task],mode='Environment Observer Validation');observations=[off.reset(seed=99,options=dict(task_id=task['id']))[0]];rewards=[]
    for a in actions:o,r,*_=off.step(a);observations.append(o);rewards.append(r)
    off.close();context=mp.get_context('spawn');mirror=context.Queue(8);telemetry=context.Queue(64);commands=context.Queue(64);stop=context.Event();closed=context.Event()
    for q in [mirror,telemetry,commands]:q.cancel_join_thread()
    on=StewartResidualEnv(root/'on',tasks=[task],record_episodes=[0],mode='Environment Observer Validation');ons=[on.reset(seed=99,options=dict(task_id=task['id']))[0]];on_rewards=[];owner=on.owner;owner.mirror=mirror;owner.telemetry=telemetry;owner.stop=stop;owner.mirror_closed=closed;cfg=dict(owner.config,close_on_complete=True)
    native=context.Process(target=mirror_main,args=(cfg,mirror,stop,closed));native.start();app=QtWidgets.QApplication([]);session=ObserverSession(cfg,telemetry,commands);window=EnhancedDashboard(session);window.setWindowTitle('Phase2C1 Gym Observer Validation - Fixed Actions, NOT PPO');window.show();timer=QtCore.QTimer();count=0;interrupted=False;saved=False;wall=time.perf_counter()
    def finish():
        timer.stop();stop.set();on.close();window.final_close=True;window.close();app.quit()
    def tick():
        nonlocal count,interrupted,saved
        if session.stopped or closed.is_set() or time.perf_counter()-wall>120:interrupted=True;finish();return
        o,r,*_=on.step(actions[count]);ons.append(o);on_rewards.append(r);count+=1
        if count>=20 and not saved:window.grab().save(str(ROOT/'assets/plots/phase2c1_observer_validation.png'));saved=True
        if count>=len(actions):finish()
    timer.timeout.connect(tick);timer.start(1);app.exec();stop.set();native.join(15)
    assert not native.is_alive()
    with np.load(root/'off/episode_0000/integration_states.npz') as a,np.load(root/'on/episode_0000/integration_states.npz') as b:equal=np.array_equal(a['states'],b['states']);difference=float(abs(a['states']-b['states']).max()) if a['states'].shape==b['states'].shape else None
    obs_equal=np.array_equal(observations,ons);reward_equal=np.array_equal(rewards,on_rewards)
    result=dict(root=str(root),bitwise_physics=bool(equal),maximum_difference=difference,observations_equal=bool(obs_equal),rewards_equal=bool(reward_equal),decisions_each=count,physics_steps_each=on.total_physical_steps,gui_qt_native=True,interrupted=interrupted,native_exitcode=native.exitcode,recording_errors=owner.recorder.render_failures,not_PPO_training=True)
    write_json(ROOT/'results/phase2c1_observer_validation.json',result);print(json.dumps(result),flush=True);assert equal and obs_equal and reward_equal and count==50 and not interrupted and not result['recording_errors']
if __name__=='__main__':mp.freeze_support();main()
