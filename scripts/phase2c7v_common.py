"""Read-only Phase 7 inputs and independent Phase 7V outputs."""
import _bootstrap
import hashlib
import json
from pathlib import Path
from src.model import ROOT,sha,write_json
OLD=ROOT/'results/runs/phase2c7_20261107_v1'
RUN=ROOT/'results/runs/phase2c7v_20261108_v1'
SEEDS=(42,123,2026)
POINTS=(0,2048,4096,6144,8192,12288,16384,24576,32768)
NAMES=('orientation','omega','xyz','action','slew')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def stage(n):return 'initial' if n==0 else 'A' if n<=4096 else 'B' if n<=8192 else 'C'
def checkpoint(seed,n):
    return ROOT/f'models/phase2c7/seed{seed}'/({0:'initial.zip',8192:'pilot.zip',32768:'final.zip'}.get(n,f'update_{n//512:03d}.zip'))
def prepare():
    RUN.mkdir(parents=True,exist_ok=False)
    paths=set(ROOT/n for n in read(OLD/'history_before.json'))
    for folder in ('src','scripts','tests','configs','docs','assets','models','results'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and
          '__pycache__' not in p.parts and 'pytest_' not in str(p) and '.pytest_cache' not in p.parts and
          'phase2c7v' not in str(p).lower() and 'phase2c7_training_analysis' not in p.name and
          'phase2c7_policy_comparison' not in p.name and not p.name.startswith('~$') and p.suffix!='.pyc')
    hashes={}
    for p in sorted(paths):
        if not p.exists():continue
        with p.open('rb') as f:hashes[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(RUN/'history_before.json',hashes)
    finalize()
    print('Frozen',len(hashes),'historical files',flush=True)

def finalize():
    inventory=[]
    for seed in SEEDS:
        records=read(OLD/f'training/seed{seed}/continue/updates.json')
        for p in sorted((ROOT/f'models/phase2c7/seed{seed}').glob('*.zip')):
            if p.stem.startswith('update_'):
                update=int(p.stem.split('_')[1]);expected=records[update-1]['sha256'];n=update*512
            else:
                n={'initial':0,'pilot':8192,'final':32768}[p.stem]
                expected=read(OLD/f'training/seed{seed}'/('pilot' if n==8192 else 'continue')/'summary.json')['model_sha'] if n else sha(p)
            assert sha(p)==expected
            inventory.append(dict(seed=seed,decisions=n,path=str(p),sha256=expected,selected=p==checkpoint(seed,n) and n in POINTS))
    write_json(RUN/'checkpoint_inventory.json',inventory)
    write_json(RUN/'protocol.json',dict(seeds=SEEDS,points=POINTS,task=read(ROOT/'configs/phase2c7_eval.yaml')['development'][0],
       evaluation='Frozen deterministic 15-second development episodes, no gradients; sampled checkpoint curve, not all 64 updates',
       comparison_video_task='F_roll',comparison_video_rule='Preselected historical loading/unloading tradeoff case; all three final seeds and classical; no model selection',
       original_final_selection_unchanged=sha(OLD/'final_selection.json'),sensitivity='Local central finite differences on recorded rollout observations; not closed-loop causality',
       training_forbidden=True))
    print('Frozen',len(inventory),'checkpoints',flush=True)
def verify():
    bad=[]
    manifest=read(RUN/'history_before.json')
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.exists():bad.append([name,'missing']);continue
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:bad.append([name,'changed'])
    write_json(RUN/'history_after.json',dict(passed=not bad,checked=len(manifest),differences=bad))
    assert not bad,bad
    print('Historical SHA audit passed',len(manifest))
if __name__=='__main__':
    import sys
    (verify if len(sys.argv)>1 and sys.argv[1]=='verify' else prepare)()
