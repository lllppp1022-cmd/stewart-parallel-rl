"""Finite deterministic observer equivalence and preset acceptance, no learning."""
import _bootstrap
import json,time,hashlib,platform,importlib.metadata as md
from pathlib import Path
import numpy as np
from src.model import ROOT,write_json
from src.visualization.owner import PhysicsOwner

EVENTS=[dict(at_step=100,kind='target',payload=dict(z_mm=1.,roll_deg=0.,pitch_deg=0.)),dict(at_step=250,kind='pause',payload=dict(auto_resume_after_s=.03)),dict(at_step=400,kind='target',payload=dict(z_mm=1.,roll_deg=.15,pitch_deg=0.)),dict(at_step=1000,kind='target',payload=dict(z_mm=10000)),dict(at_step=1500,kind='hold'),dict(at_step=2000,kind='home'),dict(at_step=4000,kind='preset',payload=dict(name='periodic'))]
def run(folder,record=False,events=None,preset=None,condition='nominal',seconds=8,controller='C0'):
    events=list(events or [])
    if preset:events.insert(0,dict(at_step=0,kind='preset',payload=dict(name=preset)))
    cfg=dict(folder=str(folder),record=record,events=events,condition=condition,controller=controller,pace=False,max_sim_seconds=seconds,max_wall_seconds=120,exit_on_failure=True,diagnostic_asymmetric=condition!='nominal')
    PhysicsOwner(cfg).run();return json.loads((folder/'summary.json').read_text())
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--existing',type=Path);args=parser.parse_args()
    root=args.existing or ROOT/'results/runs'/f'phase1_5_validation_{time.time_ns()}';root.mkdir(exist_ok=True);items={}
    def cached(name,**kwargs):
        file=root/name/'summary.json'
        return json.loads(file.read_text(encoding='utf-8')) if file.exists() else run(root/name,**kwargs)
    for name,rec in [('observer_off',False),('recorder_on',True)]:items[name]=cached(name,record=rec,events=EVENTS)
    a=np.load(root/'observer_off/integration_states.npz');b=np.load(root/'recorder_on/integration_states.npz');same=np.array_equal(a['states'],b['states']);maxdiff=float(np.max(abs(a['states']-b['states'])));assert same
    for preset in ['home','roll','pitch','z_roll','roll_pitch','periodic']:items[preset]=cached(preset,preset=preset,seconds=6)
    items['asymmetric_gain']=cached('asymmetric_gain',condition='asymmetric_gain',seconds=1)
    report=dict(run_directory=str(root),fixed_events=EVENTS,recording_equivalence=dict(full_integration_states_bitwise_equal=same,max_absolute_difference=maxdiff,physics_steps=items['observer_off']['physics_steps'],same_terminal_reason=items['observer_off']['terminal_reason']==items['recorder_on']['terminal_reason']),runs=items)
    old=json.loads((ROOT/'results/phase1_5_frozen_history.json').read_text(encoding='utf-8'));changes=[]
    for path,meta in old['files'].items():
        p=ROOT/path;h=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'MISSING'
        if h!=meta['sha256']:changes.append(path)
    report['history_freeze']=dict(files=old['count'],changed=changes);write_json(ROOT/'results/phase1_5_backend_validation.json',report)
    pins={}
    for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
        if '==' in line:
            name,version=line.split('==',1);actual=md.version(name);pins[name]=dict(expected=version,actual=actual,unchanged=actual==version)
    write_json(ROOT/'results/phase1_5_environment.json',dict(python=platform.python_version(),windows=platform.platform(),prior_dependency_versions=pins,new={n:md.version(n) for n in ['PySide6-Essentials','shiboken6','pyqtgraph']},all_prior_pins_unchanged=all(v['unchanged'] for v in pins.values())))
    (ROOT/'requirements-phase1_5-lock.txt').write_text((ROOT/'requirements-lock.txt').read_text()+'\nPySide6-Essentials==6.12.0\nshiboken6==6.12.0\npyqtgraph==0.14.0\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
