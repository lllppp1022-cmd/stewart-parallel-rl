import _bootstrap,json,platform,sys
from src.model import ROOT,sha,write_json
from src.rl.stewart_large_motion_env import RUN,FIXED
from src.rl.large_motion_rewards import CONFIG
from src.rl.large_motion_observations import FIELDS

def main():
    if RUN.exists():raise FileExistsError(RUN)
    RUN.mkdir(parents=True);old={}
    for name in ['phase2c3b_artifact_manifest.json']:
        for k,v in json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))['files'].items():old[k]=v['sha256']
    old.update(json.loads((ROOT/'results/runs/phase2c3b_20261105_v1/history_freeze.json').read_text(encoding='utf-8')))
    for folder in ['src','scripts','tests','configs','models','docs']:
        for f in (ROOT/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and 'phase2c4' not in str(f).lower() and 'large_motion_' not in f.name:old[str(f.relative_to(ROOT))]=sha(f)
    write_json(RUN/'history_freeze.json',old)
    cfg=dict(version='phase2c4_v1',fixed=FIXED,physics_dt_s=.001,decision_physical_steps=50,history_length=1,frame_fields=FIELDS,frame_dimensions=sum(x[1] for x in FIELDS),action_scale_N=.1,reward=CONFIG,normalisation='Fixed physical scales in large_motion_observations.py; no fitted/eval statistics',reward_revision_count=0)
    ppo=dict(n_steps=256,batch_size=64,n_epochs=10,learning_rate=3e-4,gamma=.99,gae_lambda=.95,log_std_init=-1.,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,target_kl=.03,net_arch=dict(pi=[32,32],vf=[32,32]),device='cpu',torch_threads=1,seeds=[42,123,2026],maximum_per_seed=4096,smoke_seed=42,smoke256_included_in_seed42_4096=True,checkpoint_every_rollout=True,selection='Last4096 checkpoint for every seed; no best-test selection',record_seed42_episodes=[0,6,12])
    ev=dict(training_regression=FIXED,final_tasks=[dict(id='E1_newtiming',rise_s=6.,tau_s=.005,Fz_N=-5.5,on_s=5.6,off_s=10.6),dict(id='E2_small_load',rise_s=6.,tau_s=.005,Fz_N=-5.25,on_s=4.8,off_s=10.2)],preflight='Frozen FF-VA full real physics before training for safety only; no parameter/reward/checkpoint selection using final outcomes',windows_s=1.,protocol_frozen_before_training=True,multi_gate=dict(minimum_seeds_passing=2,requirements='Both independent tasks: load and unload IAE improve at least1%, Z-RMSE not worse; XY peak <= baseline*1.05+0.005mm, SO3 <=baseline*1.05+0.0001deg, force slew<=baseline*1.1; no safety failure; actual parameter updates',if_pass='Level1 timings only, two seeds each4096 additional; same53-dim action/reward; history3 only separate equal-budget future ablation',if_fail='Do not run multi-condition training'),stop='Any nonfinite or actual safety termination stops that seed and future stage; preserve partial logs; no unbounded retry')
    for file,obj in [('phase2c4_env.yaml',cfg),('phase2c4_ppo.yaml',ppo),('phase2c4_eval.yaml',ev)]:write_json(ROOT/'configs'/file,obj)
    write_json(RUN/'protocol_frozen.json',dict(environment=cfg,ppo=ppo,evaluation=ev,platform=platform.platform(),python=sys.version,history_files=len(old),known_tradeoff='User explicitly authorises limited Z residual study despite FF-induced XY coupling; do not hide it or weaken baseline'))
    print('Frozen',RUN,len(old),'history entries')
if __name__=='__main__':main()
