"""Phase2B3 residual-control historical RGB/state replay, no new dynamics or tuning."""
import _bootstrap
import argparse
import numpy as np
from PySide6 import QtWidgets,QtCore
from src.visualization.enhanced.dashboard import EnhancedDashboard

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True);p.add_argument('--validate-seconds',type=float);a=p.parse_args()
    if a.validate_seconds is not None and not 0<a.validate_seconds<=30:p.error('validate-seconds must be (0,30]')
    app=QtWidgets.QApplication([]);w=EnhancedDashboard(replay_folder=a.trial);w.setWindowTitle('Phase2B3 Yaw Compensation - Historical Replay');label=QtWidgets.QLabel('Recorded residual diagnostics; no RL or new dynamics');label.setWordWrap(True);w.centralWidget().layout().insertWidget(1,label)
    def update():
        if w.latest is None:return
        df=w.replay.df;i=max(0,int(np.searchsorted(df.time_s.to_numpy(),w.latest.time_s+1e-10,side='right')-1));r=df.iloc[i]
        label.setText(f'Yaw actual {r.rpy_2:+.5f}deg | yaw target {r.rpy_ref_2:+.5f}deg | Mz request {r.task4_target_3:+.4f}Nm | ARCHIVE REPLAY | shared extra max {max(abs(r[f"F_add_total_{j}"]) for j in range(6)):.3f}/5N | {r.controller} | t={r.time_s:.3f}s | world Fz={r.external_world_2:+.2f}N / Mx={r.external_world_3:+.3f}Nm | local equiv Fz={r.diag_equivalent_disturbance_2:+.3f}N / Mx={r.diag_equivalent_disturbance_3:+.4f}Nm | max KiI={max(abs(r[f"integral_{j}"]) for j in range(6))*3000:.4f}N')
    timer=QtCore.QTimer(w);timer.timeout.connect(update);timer.start(100);w.show()
    if a.validate_seconds is not None:QtCore.QTimer.singleShot(round(a.validate_seconds*1000),w.close)
    print('Phase2B3 archive replay ready',flush=True);app.exec();print('Phase2B3 archive replay closed',flush=True)
if __name__=='__main__':main()
