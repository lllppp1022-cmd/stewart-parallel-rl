import _bootstrap
import time,json
from src.model import ROOT,load_model,write_json
from src.phase1.trajectory import Trajectory
from src.phase1.collision import CollisionQuery
from src.phase1.executor import run_trial
folder=ROOT/'results/runs'/f'phase1_pilot_{time.time_ns()}';folder.mkdir(parents=True);m=load_model();m.opt.timestep=.001;tr=Trajectory(m,'z_roll',ROOT/'results/phase1_preflight/references');query=CollisionQuery();out=[]
for method in ['C0','C1','C2']:
 summary,_=run_trial(m,tr,method,'nominal',.001,folder/method,query);out.append(summary);print(json.dumps(summary),flush=True)
write_json(folder/'complete.json',out)
