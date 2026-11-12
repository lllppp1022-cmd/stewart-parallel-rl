"""Matched complete dynamic trials; final model set is frozen before use."""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from phase2c7_protocol import ROOT,RUN,config,read,write_json,SEEDS
from validate_phase2c7_dynamic_tasks import trial
from evaluate_phase2c6 import recovery
from src.model import sha


def enrich(result):
    from pathlib import Path
    d=pd.read_csv(Path(result['folder'])/'physics.csv.gz')
    task=result['task'];t=d.time_s.to_numpy()
    force=d[[f'force_{i}' for i in range(6)]].to_numpy()
    actual=Rotation.from_euler('xyz',d[[f'rpy_{i}' for i in range(3)]],degrees=True)
    target=Rotation.from_euler('xyz',d[[f'rpy_ref_{i}' for i in range(3)]],degrees=True)
    err=(actual*target.inv()).as_rotvec()*180/np.pi
    direction=np.array(task['torque']);direction/=np.linalg.norm(direction)
    signed=err[:,:2]@direction
    result.update(task_id=task['id'],force_squared_integral_N2_s=float(np.sum(force**2)*.001),
        yaw_error_peak_deg=float(abs(err[:,2]).max()),
        unload_reverse_overshoot_deg=float(max(0,-signed[t>=task['off']].min(initial=0))),
        projection_fraction=float(d.projected.mean()),requested_rms_Nm=float(np.sqrt(np.mean(d[['requested_mx_Nm','requested_my_Nm']].to_numpy()**2))))
    for stage,start,end in [('load',task['on'],task['off']),('unload',task['off'],3+2*task['reference']['rise_s'])]:
        for band in (.1,.01):
            status,elapsed=recovery(t,d.orientation_error_deg.to_numpy(),start,end,band,result['complete'])
            result[f'{stage}_recovery_{band}_status']=status
            result[f'{stage}_recovery_{band}_s']=elapsed
    write_json(__import__('pathlib').Path(result['folder']).parent/'comparison.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['pilot','final'])
    p.add_argument('--worker',type=int,default=0);a=p.parse_args();torch.set_num_threads(1)
    final=a.group=='final'
    tasks=config('eval')['final' if final else 'development']
    models={f'PPO_{seed}':ROOT/'models/phase2c7'/f'seed{seed}'/('final.zip' if final else 'pilot.zip') for seed in SEEDS}
    if final:
        selection=read(RUN/'final_selection.json')
        assert selection['models']=={key:sha(path) for key,path in models.items()}
        assert selection['eval_sha']==sha(ROOT/'configs/phase2c7_eval.yaml')
        models['OldCurriculum42']=ROOT/'models/phase2c6/curriculum_v2/seed42/final.zip'
    admitted={t['id'] for t in read(RUN/'admission.json')['admitted']}
    work=[(task,method,model) for task in tasks if task['id'] in admitted for method,model in [('baseline',None),('classical50',None),*models.items()]]
    for i,(task,method,model) in enumerate(work):
        if i%3!=a.worker:continue
        result=trial(task,method,'final' if final else 'pilot_evaluation',model=model,
                     record=final and task['id']=='F_mix' and method in ('baseline','classical50','PPO_42'))
        enrich(result)


if __name__=='__main__':main()
