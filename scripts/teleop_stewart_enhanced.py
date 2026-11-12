"""Phase 1.5.1 opt-in entry; frozen original entry remains available."""
import _bootstrap
import argparse,json,time,math,multiprocessing as mp
from pathlib import Path
from src.visualization.enhanced.session import EnhancedSession
def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Continuous interactive simulation; close either window to stop.')
    for flag in ['record','headless','no-native-viewer','unpaced','close-on-complete']:p.add_argument('--'+flag,action='store_true')
    p.add_argument('--events',type=Path);p.add_argument('--replay',type=Path);p.add_argument('--controller',choices=['C0','C1','C2'],default='C0')
    p.add_argument('--max-sim-seconds',type=float,default=None);p.add_argument('--max-wall-seconds',type=float,default=None)
    p.add_argument('--state-chunk-steps',type=int,default=2000);p.add_argument('--video-segment-seconds',type=float,default=60)
    a=p.parse_args(argv)
    for key in ['max_sim_seconds','max_wall_seconds','video_segment_seconds']:
        value=getattr(a,key)
        if value is not None and (not math.isfinite(value) or value<=0):p.error(key+' must be positive and finite')
    if not 1<=a.state_chunk_steps<=10000:p.error('state-chunk-steps must be 1..10000')
    if a.headless and not a.replay and a.max_sim_seconds is None and a.max_wall_seconds is None:p.error('Headless runs require an explicit finite budget')
    return a

def main():
    a=parse_args()
    if a.replay:
        from PySide6.QtWidgets import QApplication
        from src.visualization.enhanced.dashboard import EnhancedDashboard
        app=QApplication([]);w=EnhancedDashboard(replay_folder=a.replay);w.show();app.exec();return
    streaming=a.max_sim_seconds is None or a.max_wall_seconds is None or a.max_sim_seconds>120 or a.max_wall_seconds>1200
    cfg=dict(controller=a.controller,condition='nominal',record=a.record,native_viewer=not a.headless and not a.no_native_viewer,pace=not a.unpaced,max_sim_seconds=a.max_sim_seconds,max_wall_seconds=a.max_wall_seconds,stream_logs=streaming,state_chunk_steps=a.state_chunk_steps,video_segment_seconds=a.video_segment_seconds,close_on_complete=a.close_on_complete or streaming,events=json.loads(a.events.read_text(encoding='utf-8')) if a.events else [])
    s=EnhancedSession(cfg);s.start();print('SESSION='+str(s.folder),flush=True)
    print('Continuous mode: close either window / Quit to stop and archive.' if a.max_sim_seconds is None and a.max_wall_seconds is None else 'Explicit finite budget enabled.',flush=True)
    if a.headless:
        try:
            while s.poll():time.sleep(.05)
        except KeyboardInterrupt:
            s.request_stop()
            while s.poll():time.sleep(.05)
    else:
        from PySide6.QtWidgets import QApplication
        from src.visualization.enhanced.dashboard import EnhancedDashboard
        app=QApplication([]);w=EnhancedDashboard(s);w.show();app.exec()
if __name__=='__main__':mp.freeze_support();main()
