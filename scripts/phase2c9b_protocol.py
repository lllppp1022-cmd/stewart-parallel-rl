"""Independent bounded Phase9B protocol and immutable experiment configuration."""
import _bootstrap
import copy
import hashlib
import json
from pathlib import Path
from src.model import ROOT, write_json, sha

RUN = ROOT / 'results/runs/phase2c9b_20261112_v1'
SEEDS = (42, 123, 2026)
METHODS = ('single_frame', 'history_5frame')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def cfg(name):
    return read(ROOT / f'configs/phase2c9b_{name}.yaml')


def task(name, mass, reference, torque=(.6, -.8), on=4.4, off=6.2):
    return dict(id=name, reference=copy.deepcopy(reference), torque=list(torque),
                on=on, off=off, stage='phase2c9b', plant=dict(kind='mass', value=mass/20), mass_kg=mass)


def initialize():
    RUN.mkdir(parents=True, exist_ok=False)
    old = read(ROOT/'configs/phase2c8_training.yaml')
    refs = old['curriculum']['C']
    home = old['curriculum']['A'][0]['reference']
    z = old['curriculum']['B'][0]['reference']
    combined = refs[6]['reference']
    train = [task(f'Train_m{mass}_{i}', mass, refs[i]['reference'])
             for mass in (20.,20.6,21.2,22.) for i in (0,6)]
    dev = [task(f'Dev_m{mass}', mass, dict(refs[3]['reference'],rise_s=6.5), (.8,.6),5.2,7.)
           for mass in (20.,22.)]
    final = [task(f'Final_m{mass}_{i}', mass, dict(refs[i]['reference'],rise_s=5.5),(-.8,.6),3.9,6.5)
             for mass in (20.,20.3,21.5,21.8) for i in (1,6)]
    layers = []
    for mass in (20.,20.3,20.6,21.2,21.5,21.6,21.8,22.):
        for i,ref in enumerate((home,z,combined)):
            layers.append(task(f'Layer{i}_m{mass}',mass,ref,(0.,0.)))
    configs = dict(mass=dict(nominal_kg=20.,training_kg=[20.,20.6,21.2,22.],test_kg=[20.3,21.5,21.8],
                            inertia='fixed-shape density scaling',lag_s=.005),
        tasks=dict(training=train,development=dev,final=final,layers=layers),
        ppo=old['ppo'],
        evaluation=dict(seeds=SEEDS,methods=METHODS,pilot=8192,total=32768,
          hard_gate='all admitted tasks complete with zero/random residual and unchanged physical protections; zero/interface/RNG tests pass',
          history_classification='auxiliary evidence; Home ceiling never blocks training',
          pilot_gate='all six finite runs and real actor/critic updates; no protection; per-seed development mean IAE/XYZ <=150% initial; at least one development IAE or common return improves in each method family',
          final_success='>=2 seeds each improve both IAEs by >=1% on >=2 final tasks vs matched single-frame; all safe and peak/omega/XYZ/force slew/integral <=110% comparator',
          adaptive_candidates=[dict(kp=25.,kd=1.,ki=k,leak=1.,imax=.025) for k in (0.,10.,30.)],
          selection='all seeds at common terminal budget; adaptive chosen by development mean load+unload IAE, safety/XYZ guards'))
    for name,value in configs.items():
        path=ROOT/f'configs/phase2c9b_{name}.yaml'
        assert not path.exists()
        write_json(path,value)
    write_json(RUN/'config_freeze.json',{str(ROOT/f'configs/phase2c9b_{n}.yaml'):sha(ROOT/f'configs/phase2c9b_{n}.yaml') for n in configs})
    print('FROZEN',RUN)


def freeze():
    files=[]
    for folder in ('src','scripts','tests','configs','docs','assets','models','results','third_party'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and 'phase2c9b' not in str(p).lower()
                     and '__pycache__' not in p.parts and 'pytest' not in str(p).lower())
    files.append(ROOT/'README.md')
    with (RUN/'history.jsonl').open('x',encoding='utf8') as stream:
        for p in sorted(files):
            with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            stream.write(json.dumps([str(p.relative_to(ROOT)),digest],ensure_ascii=False)+'\n')
    print('FROZEN FILES',len(files),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['initialize','freeze'])
    globals()[parser.parse_args().mode]()
