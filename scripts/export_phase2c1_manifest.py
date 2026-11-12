"""Content hashes for new source, frozen configs, models and real Pilot journals."""
import _bootstrap
from src.model import ROOT,sha,write_json

def main():
    run=ROOT/'results/runs/phase2c1_20261029_v1';paths=set(p for p in run.rglob('*') if p.is_file())
    for pattern in ['src/rl/*.py','scripts/*phase2c1*.py','configs/phase2c1*','tests/test_phase2c1*.py','docs/31_*','docs/32_*','models/phase2c1/**/*.zip','assets/plots/phase2c1*.png','assets/videos/stewart_phase2c1*','results/phase2c1*.csv','results/phase2c1*.json','results/phase2c1_final_tests.xml']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and p.name!='phase2c1_artifact_manifest.json')
    paths.add(ROOT/'phase2c1_pytest_policy.py');files={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
    write_json(ROOT/'results/phase2c1_artifact_manifest.json',dict(count=len(files),total_bytes=sum(x['bytes'] for x in files.values()),scope='All files of actual main Pilot run; new source/configs/models/selected plots and videos; no changes to historical manifest',files=files));print('HASHED',len(files),'files',flush=True)

if __name__=='__main__':main()
