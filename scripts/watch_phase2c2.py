"""Read-only replay of actual recorded PPO samples; never trains or advances physics."""
import _bootstrap,argparse
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets,QtCore
from src.model import ROOT
from src.visualization.enhanced.dashboard import EnhancedDashboard

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=ROOT/'results/runs/phase2c2_20261030_v1');p.add_argument('--seed',type=int,default=42);p.add_argument('--episode',type=int,default=2);p.add_argument('--validate-seconds',type=float);a=p.parse_args()
    if a.validate_seconds is not None and not 0<a.validate_seconds<=30:p.error('Bounded GUI check must be <=30 seconds')
    folder=a.run_root/f'seed{a.seed}/actual_training/episode_{a.episode:04d}'
    if not (folder/'summary.json').exists():p.error('Episode not yet archived; select a completed episode')
    app=QtWidgets.QApplication([]);w=EnhancedDashboard(replay_folder=str(folder));w.setWindowTitle('Phase2C2 Actual Training Archive | READ ONLY');w.labels.setChecked(False);label=QtWidgets.QLabel('Actual stochastic training archive; no new sampling');label.setWordWrap(True);w.centralWidget().layout().insertWidget(1,label)
    def update():
        if w.latest is None:return
        d=w.replay.df;i=max(0,int(np.searchsorted(d.time_s.to_numpy(),w.latest.time_s+1e-10,side='right')-1));r=d.iloc[i];source='ON-SITE ACTUAL RGB' if w.replay.reader else 'RECORDED STATE RECONSTRUCTION (no original RGB)'
        label.setText(f'{source} | episode {a.episode}, policy version {int(r.policy_version)}, global decision {int(r.rl_global_decision)} | actual time {r.time_s:.3f}s | raw action {r.rl_actor_raw:+.4f}, clipped {r.rl_normalized_action:+.4f} | applied extra Fz {r.rl_applied_command_wrench_2:+.4f}N | Z error {(r.p_2-r.p_ref_2)*1e6:+.2f}um | RGB REPLAY IS NOT A NEW TRAINING RUN')
        w.message.setText(f'Training archive | reward cost={sum(r["reward_cost_"+k] for k in ["z","xy","angle","amplitude","slew","request"]):.4f} | policy={int(r.policy_version)} | Play/Pause or Frame >')
    timer=QtCore.QTimer(w);timer.timeout.connect(update);timer.start(100);w.show();QtCore.QTimer.singleShot(100,w.toggle_replay)
    if a.validate_seconds is not None:
        def screenshot():
            w.grab().save(str(ROOT/'results/phase2c2_replay_gui_playing.png'));w.replay_robot.grab().save(str(ROOT/'results/phase2c2_replay_robot_playing.png'))
        QtCore.QTimer.singleShot(round(a.validate_seconds*500),screenshot)
        QtCore.QTimer.singleShot(round(a.validate_seconds*1000),w.close)
    print('Actual PPO archive replay ready',flush=True);app.exec();print('Archive replay closed normally',flush=True)
if __name__=='__main__':main()
