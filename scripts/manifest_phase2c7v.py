"""Finalize evidence manifest for the new analysis only."""
import hashlib
from phase2c7v_common import ROOT,RUN,read,write_json
def main():
    paths=set()
    for directory,pattern in [('scripts','*phase2c7v*.py'),('tests','*phase2c7v*.py'),('docs','*Phase2C7V*.md'),('results','phase2c7v_*.csv'),('results','phase2c7v_*.png')]:paths.update((ROOT/directory).glob(pattern))
    for name in ('training_analysis','policy_comparison'):paths.add(ROOT/f'assets/videos/stewart_phase2c7_{name}.mp4')
    paths.update(p for p in RUN.rglob('*') if p.is_file() and 'pytest_regression' not in p.parts and 'draft_videos' not in p.parts and p.name!='release_manifest.json')
    paths.update(p for p in (ROOT/'results/phase2c7v_tensorboard').rglob('*') if p.is_file())
    out={}
    for p in sorted(paths):
        with p.open('rb') as f:out[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    h=read(RUN/'history_after.json');assert h['passed'] and not h['differences']
    write_json(RUN/'release_manifest.json',dict(files=out,files_count=len(out),historical_files_preserved=h['checked'],new_training_decisions=0,new_gradient_updates=0))
    print('New artifact manifest:',len(out),'files; historical preserved:',h['checked'],flush=True)
if __name__=='__main__':main()
