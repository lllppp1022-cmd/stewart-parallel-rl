import _bootstrap
import hashlib
from src.model import ROOT,write_json
out=ROOT/'results/phase1_5_frozen_history.json'
if out.exists():raise FileExistsError('Do not replace history freeze')
files={}
for name in ['src','scripts','configs','tests','docs','assets','results','models','third_party']:
 for p in (ROOT/name).rglob('*'):
  rel=p.relative_to(ROOT)
  if not p.is_file() or any(x in rel.parts for x in ['.git','__pycache__']) or any(x in str(rel).lower() for x in ['phase1_5','visualization','teleop_stewart','replay_stewart','view_stewart_dashboard','render_stewart_dashboard']):continue
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
  files[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
write_json(out,dict(files=files,count=len(files),scope='Pre-existing Phase0/Phase1 source, model, reports and raw records; caches/.git excluded'))
print('Frozen',len(files),'files')
