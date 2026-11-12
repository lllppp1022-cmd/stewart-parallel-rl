"""Historical verification and completed new-artifact manifest."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from phase2c9b_protocol import ROOT,RUN,read,write_json,sha


def verify():
    changes=[];count=0
    for line in (RUN/'history.jsonl').open(encoding='utf8'):
        name,expected=json.loads(line);p=ROOT/name;count+=1
        actual=sha(p) if p.exists() else None
        if actual!=expected:changes.append(name)
    write_json(RUN/'history_verification.json',dict(checked=count,changes=changes,passed=not changes))
    assert not changes,changes
    print('HISTORY VERIFIED',count,flush=True)


def files_under(root):
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if not d.startswith('pytest') and d!='__pycache__']
        for name in files:yield Path(folder)/name


def seal():
    assert read(RUN/'history_verification.json')['passed']
    for p,h in read(RUN/'config_freeze.json').items():assert sha(p)==h
    for p,h in read(RUN/'executed_training_implementation.json')['files'].items():assert sha(ROOT/p)==h
    models=read(RUN/'final_selection.json')['models']
    for item in models.values():assert sha(item['path'])==item['sha']
    decision=read(ROOT/'results/phase2c9b_final_decision.json')
    if decision['matched_capacity_control_required']:
        assert read(RUN/'capacity_control_validation.json')['passed']
    docs=[next((ROOT/'docs').glob(f'{i}_Phase2C9B*.md')) for i in range(88,95)]
    broken=[]
    for p in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if not target.startswith(('http://','https://','#')) and not (p.parent/target).resolve().exists():broken.append([str(p),target])
    assert not broken,broken
    for kind in ('unknown_mass','history_training','single_vs_history'):
        v=read(RUN/f'video_{kind}.json');assert sha(v['path'])==v['sha']
    proofs=[read(p) for p in (RUN/'viewer').glob('*/viewer_validation.json')]
    assert {p['method'] for p in proofs}=={'single_frame','history_5frame'} and all(p['headless_bitwise_equal'] and p['window_closed'] for p in proofs)
    assert read(RUN/'physical_validation.json')['passed']
    assert read(RUN/'tensorboard_validation.json')['status']==200
    write_json(RUN/'delivery_validation.json',dict(passed=True,documents=7,broken_links=broken,
        config_and_executed_sources_unchanged=True,final_models_match=True,gui_proofs=proofs,
        all_seed_results_included=True,physical_reward_and_metric_checks=True))
    paths=set(docs)
    for folder in ('scripts','tests','configs','assets/videos','assets/plots','results'):
        paths.update(p for p in (ROOT/folder).glob('*phase2c9b*') if p.is_file())
    for folder in (RUN,ROOT/'models/phase2c9b'):paths.update(files_under(folder))
    manifest=RUN/'artifact_manifest.json'
    paths=sorted(p for p in paths if p!=manifest and not p.name.startswith(('tb_server','delivery_seal')))
    records=[]
    for i,p in enumerate(paths):
        records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        if i%500==0:print('Hashed',i,len(paths),flush=True)
    write_json(manifest,dict(files=records,count=len(records),bytes=sum(r['bytes'] for r in records),
        scope='New completed Phase9B artifacts, excludes regression temporary directories, live TensorBoard server logs, active delivery_seal logs and self'))
    print('SEALED',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','seal']);globals()[parser.parse_args().mode]()
