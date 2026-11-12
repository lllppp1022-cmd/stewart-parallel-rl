"""Resolve portable metadata paths inside this checkout after cloning."""
import _bootstrap
from pathlib import Path
import json
from src.model import ROOT


def resolve(value):
    if isinstance(value,str):return value.replace('__PROJECT_ROOT__',str(ROOT))
    if isinstance(value,list):return [resolve(x) for x in value]
    if isinstance(value,dict):return {resolve(k):resolve(v) for k,v in value.items()}
    return value


def main():
    count=0
    for folder in ('results','configs'):
        for path in (ROOT/folder).rglob('*.json'):
            text=path.read_text(encoding='utf8')
            if '__PROJECT_ROOT__' in text:
                path.write_text(json.dumps(resolve(json.loads(text)),ensure_ascii=False,indent=2),encoding='utf8');count+=1
        for path in (ROOT/folder).glob('*.csv'):
            text=path.read_text(encoding='utf8')
            if '__PROJECT_ROOT__' in text:path.write_text(text.replace('__PROJECT_ROOT__',ROOT.as_posix()),encoding='utf8');count+=1
    print('Resolved metadata files:',count)


if __name__=='__main__':main()
