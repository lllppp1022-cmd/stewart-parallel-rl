"""Finite post-run worker. Waits for one bounded pilot, then audits and evaluates it."""
import argparse
import time
import torch
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read
from analyze_phase2c8_critic import analyze
from evaluate_phase2c8 import evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['v2_control','v3','v3_return_norm'],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();torch.set_num_threads(1)
    summary=RUN/f'training/{a.method}/seed{a.seed}/B/summary.json';deadline=time.monotonic()+3600
    while not summary.exists():
        if time.monotonic()>deadline:raise TimeoutError('Pilot did not finish within one hour; no further action')
        time.sleep(10)
    s=read(summary);assert s['error'] is None and s['decisions']==8192 and s['failures']==0,s
    analyze(a.method,a.seed)
    for n in (0,4096,8192):
        for task in cfg('evaluation')['development']:
            evaluate(task,a.method,'development',a.seed,n,ROOT/f'models/phase2c8/{a.method}/seed{a.seed}/checkpoint_{n:06d}.zip')
    if a.method=='v2_control' and a.seed==42:
        for task in cfg('evaluation')['development']:
            for method in ('baseline','classical50'):evaluate(task,method,'development')
    print('PILOT POSTPROCESS COMPLETE',a.method,a.seed,flush=True)
if __name__=='__main__':main()
