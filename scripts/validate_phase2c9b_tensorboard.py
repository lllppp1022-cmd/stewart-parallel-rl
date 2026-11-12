"""Verify the running local server exposes actual completed run scalars."""
import json
import time
from urllib.request import urlopen
from urllib.parse import urlencode
from phase2c9b_protocol import RUN,METHODS,SEEDS,write_json

URL='http://127.0.0.1:6037'

def get(path):
    with urlopen(URL+path,timeout=20) as response:
        assert response.status==200
        return json.load(response)

def main():
    tag='fixed_development/load_iae_deg_s'
    for attempt in range(150):
        tags=get('/data/plugin/scalars/tags')
        derived={key:value for key,value in tags.items() if key.startswith('derived')}
        ready=len(derived)==6 and all(tag in value for value in derived.values())
        if ready:break
        if attempt%10==0:print('WAITING SERVER RELOAD',[(key,len(value)) for key,value in derived.items()],flush=True)
        time.sleep(2)
    assert len(derived)==6
    proofs=[]
    for method in METHODS:
        for seed in SEEDS:
            run=next(key for key in derived if key.endswith(f'{method}_seed{seed}'))
            assert tag in tags[run]
            values=get('/data/plugin/scalars/scalars?'+urlencode(dict(run=run,tag=tag)))
            points=[int(row[1]) for row in values]
            assert points==[0,8192,16384,24576,32768],points
            proofs.append(dict(run=run,development_points=points,scalar_tags=len(tags[run])))
    write_json(RUN/'tensorboard_validation.json',dict(url=URL,status=200,runs=list(tags),derived=proofs))
    print('TENSORBOARD VERIFIED',len(tags),'runs',flush=True)

if __name__=='__main__':main()
