"""Independent Phase 8 experiment protocol. Prior artifacts are read-only."""
import _bootstrap
import copy
import hashlib
import json
import platform
import sys
import importlib.metadata as metadata
from pathlib import Path
from src.model import ROOT,write_json,sha
RUN=ROOT/'results/runs/phase2c8_20261109_v1'
OLD=ROOT/'results/runs/phase2c7_20261107_v1'
SEEDS=(42,123,2026)
METHODS=('v2_control','v3')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def cfg(name):return read(ROOT/f'configs/phase2c8_{name}.yaml')
def task(name,ref,mx,my,on,off,stage):return dict(id=name,reference=copy.deepcopy(ref),torque=[mx,my],on=on,off=off,stage=stage)
def initialize():
    RUN.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/'configs/phase2c7_curriculum.yaml');c=old['C'];b=old['B']
    dev=[task('D8_roll',c[0]['reference'],.7,0,4.1,6.1,'development'),task('D8_pitch',c[1]['reference'],0,.85,8.4,10.3,'development'),task('D8_mix',c[2]['reference'],.48,-.64,5.,7.,'development')]
    final=[task('T8_roll',dict(c[0]['reference'],rise_s=5.5),-.9,0,3.7,5.6,'final'),task('T8_pitch',dict(c[1]['reference'],rise_s=6.5),0,1.05,9.7,11.4,'final'),task('T8_mix',dict(c[3]['reference'],rise_s=6.5),-.66,-.88,7.6,9.3,'final'),task('T8_combined50',c[6]['reference'],.96,-.72,4.9,6.7,'final')]
    old_tasks=sum([old[k] for k in ('A','B','C')],[])+read(ROOT/'configs/phase2c7_eval.yaml')['final']+read(ROOT/'configs/phase2c7_eval.yaml')['development']
    signature=lambda t:json.dumps({k:v for k,v in t.items() if k not in ('id','stage')},sort_keys=True)
    assert not {signature(t) for t in final}&{signature(t) for t in old_tasks+dev}
    ppo=read(ROOT/'configs/phase2c7_ppo.yaml')
    configs=dict(training=dict(ppo=ppo,seeds=SEEDS,methods=METHODS,curriculum={k:old[k] for k in ('A','B','C')},stage_boundaries=[4096,8192,32768],pilot=8192,total=32768,task_rng='PCG64 SeedSequence([training_seed,8008]); independent of Gymnasium/VecEnv seeds',resume_scope='Declared curriculum boundaries only; interrupted mid-stage trials are archived, never silently reset and called equivalent',checkpoint_points=[0,4096,8192,16384,24576,32768],
        pilot_gate=dict(rule='All six runs finite/complete, no protection. >=2/3 V3 seeds improve BOTH development mean load and unload IAE by >=1% vs matched V2; no seed worsens either mean by >10%; peak attitude/omega/Z/XYZ/XY/force slew/force squared integral <=110% matched V2 on every task.',minimum_improvement=.01,noninferiority=1.10),
        final_success='At least two seeds each improve BOTH load and unload IAE by >=1% vs classical50 on >=2 final tasks, all safe/complete; peak/omega/Z/XYZ/XY/force slew/integral <=110% classical on every task.'),
      reward_v3=dict(version='V3_equal_linear_quadratic',alpha=.5,beta=.5,epsilon=.01,orientation_scale_deg=.006,formula='alpha*(sqrt(e_norm**2+epsilon**2)-epsilon)+beta*e_norm**2',other_costs='Unchanged V2 omega/XYZ/accepted/slew; common proportional physical cap 100 and original failure penalty',candidates=[dict(alpha=.5,beta=.5),dict(alpha=1.,beta=.25)],selection='A priori equal mix selected if finite/unit/ranking/failure checks pass; second candidate offline diagnostic only, no seed-dependent weight search',privileged_inputs=False),
      evaluation=dict(development=dev,final=final,freeze_final_before_training=True,final_policy_selection='All terminal checkpoints at the common admitted budget; no best-seed selection',gains=dict(kp=25.,kd=1.),recovery_band_deg=.01,recovery_dwell_s=.5,optional_2p4_Nm='Not admitted: optional, not needed for this bounded study',forbidden_4p8_Nm=True),
      return_norm=dict(enabled_only_after_reward_gate=True,norm_obs=False,norm_reward=True,gamma=ppo['gamma'],clip_reward=1000000.,epsilon=1e-8,training=True,evaluation_training=False,evaluation_norm_reward=False,interpretation='Reward scale treatment affects Actor and Critic, not Critic-only',pilot=8192,total=32768))
    for name,value in configs.items():
        path=ROOT/f'configs/phase2c8_{name}.yaml';assert not path.exists();write_json(path,value)
    write_json(RUN/'config_freeze.json',{str(ROOT/f'configs/phase2c8_{k}.yaml'):sha(ROOT/f'configs/phase2c8_{k}.yaml') for k in configs})
    versions={n:metadata.version(n) for n in ('mujoco','stable-baselines3','torch','numpy','gymnasium','pandas')}
    import torch
    write_json(RUN/'runtime.json',dict(python=sys.executable,version=sys.version,platform=platform.platform(),packages=versions,cuda_available=torch.cuda.is_available(),training_device='cpu',threads=1))
    print('Protocol frozen',RUN,flush=True)
def freeze():
    paths=set()
    for folder in ('src','scripts','tests','configs','docs','assets','models','results','third_party'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and 'phase2c8' not in str(p).lower() and '__pycache__' not in p.parts and 'pytest' not in str(p).lower() and p.suffix!='.pyc' and not p.name.startswith('~$'))
    out={}
    for p in sorted(paths):
        with p.open('rb') as f:out[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(RUN/'history_before.json',out);print('Historical hashes frozen',len(out),flush=True)
def verify():
    changes=[];before=read(RUN/'history_before.json')
    for name,h in before.items():
        p=ROOT/name
        if not p.exists():changes.append(name);continue
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=h:changes.append(name)
    write_json(RUN/'history_after.json',dict(checked=len(before),passed=not changes,changes=changes));assert not changes,changes
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','freeze','verify']);globals()[p.parse_args().mode]()
