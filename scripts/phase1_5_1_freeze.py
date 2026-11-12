import _bootstrap
from pathlib import Path
import hashlib
from src.model import ROOT,write_json

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    out=ROOT/'results/phase1_5_1_frozen_history.json'
    if out.exists():raise FileExistsError('Freeze already exists; do not replace')
    paths=[]
    for folder in ['src','scripts','tests','docs','configs','models','results','assets','third_party']:
        for p in (ROOT/folder).rglob('*'):
            rel=p.relative_to(ROOT)
            if p.is_file() and not any(x in rel.parts for x in ['__pycache__','.git','enhanced']) and 'phase1_5_1' not in str(rel).lower() and 'stewart_enhanced' not in str(rel).lower():paths.append(p)
    paths.extend(p for p in ROOT.glob('*') if p.is_file())
    files={p.relative_to(ROOT).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size) for p in paths}
    write_json(out,dict(files=files,count=len(files),policy='All pre-existing files remain byte-identical; enhancement is opt-in new modules/entry points'))
    print('Frozen',len(files),'files')
