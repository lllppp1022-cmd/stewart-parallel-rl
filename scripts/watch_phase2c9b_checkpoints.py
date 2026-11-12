"""Read-only development evaluation of completed 16k/24k update snapshots."""
import concurrent.futures
import argparse
import subprocess
import sys
import time
import torch
from phase2c9b_protocol import ROOT,RUN,METHODS,SEEDS,cfg,read,write_json
from evaluate_phase2c9b import trial


def evaluate(method,seed,point):
    torch.set_num_threads(1)
    model=ROOT/f'models/phase2c9b/{method}/seed{seed}/update_{point//512:03d}.zip'
    for task in cfg('tasks')['development']:
        trial(task,method,'development',seed,point,model_override=model)


def launch(method,seed,point):
    with (RUN/f'early_development_{method}_{seed}_{point}.log').open('w',encoding='utf8') as output:
        process=subprocess.run([sys.executable,str(ROOT/'scripts/watch_phase2c9b_checkpoints.py'),
            '--method',method,'--seed',str(seed),'--point',str(point)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
    if process.returncode:raise RuntimeError((method,seed,point,process.returncode))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int);parser.add_argument('--point',type=int);args=parser.parse_args()
    if args.method:
        evaluate(args.method,args.seed,args.point)
        return
    torch.set_num_threads(1)
    for point in (16384,24576):
        deadline=time.monotonic()+7200
        while True:
            complete=True
            for method in METHODS:
                for seed in SEEDS:
                    path=RUN/f'training/{method}/seed{seed}/from_008192/updates.json'
                    complete &= path.exists() and read(path)[-1]['decisions']>=point
            if complete:break
            if time.monotonic()>deadline:raise TimeoutError(point)
            time.sleep(20)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(launch,m,s,point) for m in METHODS for s in SEEDS]
            for future in concurrent.futures.as_completed(futures):future.result()
        write_json(RUN/f'development_{point}_complete.json',dict(passed=True,checkpoint_decisions=point))
        print('DEVELOPMENT POINT COMPLETE',point,flush=True)


if __name__=='__main__':main()
