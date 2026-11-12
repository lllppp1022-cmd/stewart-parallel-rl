"""Live controller. Spawn-safe on native Windows."""
import _bootstrap
import argparse,json,time,multiprocessing as mp
from pathlib import Path
from src.visualization.session import Session

def main(default_preset=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',choices=['C0','C1','C2'],default='C0')
    p.add_argument('--condition',choices=['nominal','asymmetric_gain'],default='nominal')
    p.add_argument('--diagnostic-asymmetric',action='store_true')
    p.add_argument('--record',action='store_true');p.add_argument('--headless',action='store_true')
    p.add_argument('--no-native-viewer',action='store_true');p.add_argument('--unpaced',action='store_true')
    p.add_argument('--preset',choices=['home','z','roll','pitch','z_roll','roll_pitch','periodic'],default=default_preset)
    p.add_argument('--events',type=Path);p.add_argument('--max-sim-seconds',type=float,default=60)
    p.add_argument('--max-wall-seconds',type=float,default=600);p.add_argument('--exit-on-failure',action='store_true')
    p.add_argument('--title',default='Stewart Phase1.5 Control Panel');p.add_argument('--close-on-complete',action='store_true')
    a=p.parse_args()
    if a.condition!='nominal' and not a.diagnostic_asymmetric:p.error('Mismatch is locked: use --diagnostic-asymmetric for hold-only diagnostic')
    if not 0<a.max_sim_seconds<=120 or not 0<a.max_wall_seconds<=1200:p.error('Bounded budget required (<=120 sim seconds /1200 wall seconds)')
    c=vars(a).copy();c['native_viewer']=not a.no_native_viewer and not a.headless;c['pace']=not a.unpaced
    c['events']=json.loads(a.events.read_text(encoding='utf-8')) if a.events else []
    if a.preset:c['events']=[dict(at_step=0,kind='preset',payload=dict(name=a.preset))]+c['events']
    c.pop('preset');c.pop('headless');c.pop('no_native_viewer');c.pop('unpaced')
    session=Session(c);session.start();print('SESSION='+str(session.folder),flush=True)
    if a.headless:
        try:
            while session.poll():time.sleep(.05)
        except KeyboardInterrupt:
            session.request_stop()
            while session.poll():time.sleep(.05)
        print((Path(session.current['folder'])/'summary.json').read_text(encoding='utf-8'),flush=True)
    else:
        from PySide6.QtWidgets import QApplication
        from src.visualization.live_dashboard import Dashboard
        app=QApplication([]);window=Dashboard(session);window.setWindowTitle(a.title);window.show();app.exec()
    return session

if __name__=='__main__':mp.freeze_support();main()
