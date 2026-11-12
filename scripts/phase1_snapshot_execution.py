import _bootstrap
import shutil,time
from src.model import ROOT,write_json,sha
folder=sorted((ROOT/'results/runs').glob('phase1_suite_*'))[-1]/'executed_sources'
folder.mkdir(exist_ok=False);files=list((ROOT/'src/phase1').glob('*.py'))+[ROOT/'scripts/compare_phase1_controllers.py',ROOT/'configs/phase1_protocol.json'];records=[]
for source in files:
 target=folder/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);records.append(dict(path=source.relative_to(ROOT).as_posix(),sha256=sha(source)))
write_json(folder/'manifest.json',dict(stage='Snapshot during initial formal trial, before adding optional GUI lifetime/budget arguments; mathematical control/plant logic remains unchanged',files=records))
print(folder)
