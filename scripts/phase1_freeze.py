import _bootstrap
import hashlib,json,time
from pathlib import Path
from src.model import ROOT,write_json
p=ROOT/'results/phase1_frozen_phase0.json'
if p.exists():raise FileExistsError('Freeze already exists; do not replace it')
files={};excluded=[]
for folder in ['src','scripts','configs','tests','docs','assets','results','third_party']:
    for f in (ROOT/folder).rglob('*'):
        rel=f.relative_to(ROOT)
        if not f.is_file() or any(x in rel.parts for x in ['.git','__pycache__']) or 'phase1' in str(rel).lower():continue
        digest=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(1024*1024),b''):digest.update(b)
        files[rel.as_posix()]=dict(sha256=digest.hexdigest(),bytes=f.stat().st_size)
write_json(p,dict(client_date='2026-10-27',files=files,count=len(files),purpose='Immutable Phase0 evidence/source list before Phase1; no old file is edited',algorithm_or_rl_training=False))
print('Frozen',len(files),'files',sum(x['bytes'] for x in files.values()),'bytes')
