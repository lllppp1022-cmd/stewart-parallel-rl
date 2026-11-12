import argparse
import torch
from phase2c8_protocol import RUN,SEEDS,cfg,read,sha
from evaluate_phase2c8 import evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args();torch.set_num_threads(1);selection=read(RUN/'final_selection.json');assert selection['frozen'];work=[]
    for t in cfg('evaluation')['final']:
        work.extend((t,m,0,0,None) for m in ('baseline','classical50'))
        for key,value in selection['models'].items():
            method,seed=key.split('/');assert sha(value['path'])==value['sha'];work.append((t,method,int(seed[4:]),selection['budget'],value['path']))
    for i,(task,method,seed,n,model) in enumerate(work):
        if i%6==a.worker:evaluate(task,method,'final',seed,n,model,record=task['id'] in ('T8_mix','T8_combined50'))
if __name__=='__main__':main()
