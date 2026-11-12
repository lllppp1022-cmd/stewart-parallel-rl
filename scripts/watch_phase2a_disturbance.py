"""Read-only Phase2A historical display; actual RGB if recorded, otherwise labelled state replay."""
import _bootstrap
import argparse
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.visualization.enhanced.dashboard import EnhancedDashboard
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True);p.add_argument('--validate-seconds',type=float,help='Bounded read-only GUI smoke test, then normal close');a=p.parse_args()
    if a.validate_seconds is not None and not 0<a.validate_seconds<=30:p.error('validate-seconds must be (0,30]')
    print('Loading archived trial',flush=True);app=QApplication([]);w=EnhancedDashboard(replay_folder=a.trial);w.setWindowTitle('Phase2A Disturbance Experiment - Historical Replay');w.show()
    if a.validate_seconds is not None:QTimer.singleShot(round(a.validate_seconds*1000),w.close)
    print('Historical replay GUI ready',flush=True);app.exec();print('Historical replay closed normally',flush=True)
