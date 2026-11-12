"""Read actual archived large-motion RGB and physical curves, no new integration."""
import _bootstrap,argparse
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets,QtCore
from src.model import ROOT
from src.visualization.enhanced.dashboard import EnhancedDashboard
from analyze_phase2c3a_workspace import RUN

def main():
    p=argparse.ArgumentParser();p.add_argument('--trial',default='fourDlead_L50_T10_tau0_load0');p.add_argument('--validate-seconds',type=float);a=p.parse_args();folder=RUN/'dynamics'/a.trial
    if not (folder/'summary.json').exists():p.error('Select an archived completed trial')
    if a.validate_seconds is not None and not 0<a.validate_seconds<=30:p.error('Validation must be <=30 seconds')
    app=QtWidgets.QApplication([]);w=EnhancedDashboard(replay_folder=str(folder));w.setWindowTitle('LARGE-WORKSPACE DYNAMIC MOTION VALIDATION | ACTUAL RGB ARCHIVE');w.labels.setChecked(False);label=QtWidgets.QLabel();label.setWordWrap(True);w.centralWidget().layout().insertWidget(1,label)
    def update():
        if w.latest is None:return
        d=w.replay.df;i=max(0,int(np.searchsorted(d.time_s.to_numpy(),w.latest.time_s+1e-10,side='right')-1));r=d.iloc[i]
        label.setText(f'ACTUAL DYNAMIC EXECUTION ARCHIVE | {a.trial} | t={r.time_s:.3f}s | actual Z={r.actual_z_from_home_m*1000:.3f}mm / reference Z={r.reference_z_from_home_m*1000:.3f}mm | mesh gap={r.expanded_mesh_gap_m*1000:.2f}mm | weld={r.weld_translation_m:.2e}m,{r.weld_angle_rad:.2e}rad | {r.failure if isinstance(r.failure,str) else "NO PROTECTION"}')
    timer=QtCore.QTimer(w);timer.timeout.connect(update);timer.start(100);w.show();QtCore.QTimer.singleShot(100,w.toggle_replay)
    if a.validate_seconds:
        QtCore.QTimer.singleShot(round(a.validate_seconds*500),lambda:w.replay_robot.grab().save(str(ROOT/'results/phase2c3a_gui_validation.png')));QtCore.QTimer.singleShot(round(a.validate_seconds*1000),w.close)
    app.exec();print('Large-motion archive GUI closed normally',flush=True)
if __name__=='__main__':main()
