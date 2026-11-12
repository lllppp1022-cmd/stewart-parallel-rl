import _bootstrap
import argparse
from PySide6.QtWidgets import QApplication
from src.visualization.live_dashboard import Dashboard
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Read-only replay: folder selects episode/controller/condition; no integration')
    p.add_argument('episode');a=p.parse_args();app=QApplication([]);w=Dashboard(replay_folder=a.episode);w.show();app.exec()
