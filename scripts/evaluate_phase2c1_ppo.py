"""Open frozen final tasks only after all seed budgets/checkpoints are frozen."""
import _bootstrap,argparse,json
from pathlib import Path
import torch
from src.model import ROOT,write_json,sha
from src.rl.stewart_residual_env import env_config,owner_config
from src.rl.ppo_trace import TracedPPO,parameter_hash
from src.rl.evaluation import evaluate_model
from src.control.phase2c0_experiment import PhysicalOwner
from src.control.phase2b1_experiment import trial_metrics
from src.disturbance.recording import install_colored_capture

def freeze(root):
    h=json.loads((ROOT/'configs/phase2c1_ppo.yaml').read_text(encoding='utf-8'));models=[]
    for seed in h['training_seeds']:
        r=json.loads((root/f'seed{seed}/training_result.json').read_text(encoding='utf-8'));models.append(dict(seed=seed,initial=str(Path(r['final_checkpoint']).parent/'initial.zip'),final=r['final_checkpoint'],final_sha256=r['final_sha256'],decisions=r['actual_decisions'],updates=r['updates'],stop=r['stop_reason']))
    out=root/'final_evaluation_frozen.json'
    if out.exists():raise FileExistsError(out)
    write_json(out,dict(models=models,tasks=env_config()['final_tasks'],evaluation_seeds=h['final_evaluation_seeds'],initial_stochastic_seeds=h['initial_stochastic_evaluation_seeds'],checkpoint_selection='Final actual budget endpoint for every seed, no best-test checkpoint selection',success_screen='At least2/3 seeds improve mean load IAE by>=10% versus C4+fourDlead, both tasks not degraded on aggregate, no new failure or>5% actual force slew/XY peak increase; finite pilot, no statistical significance',VecNormalize=False))

def classical(root):
    cfg=env_config();rows=[]
    for i,task in enumerate(cfg['final_tasks']):
        for method in ['C4','C4_fourDlead']:
            folder=root/'evaluation/classical'/task['id']/method;c=owner_config(folder,task,record=i==0);c.update(stage='final_evaluation',method=method)
            if method=='C4_fourDlead':c['diagnostic']=dict(kind='four_d_lead')
            owner=PhysicalOwner(c)
            if i==0:install_colored_capture(owner.recorder,owner.model)
            owner.run();m=trial_metrics(folder,dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3));m.update(method=method,task_id=task['id'],source='Baseline Comparison');write_json(folder/'evaluation_metrics.json',m);rows.append(m);print('FINAL CLASSICAL',task['id'],method,m['success'],flush=True)
    write_json(root/'classical_evaluation.json',rows)

def learned(root,seed):
    frozen=json.loads((root/'final_evaluation_frozen.json').read_text(encoding='utf-8'));entry=next(x for x in frozen['models'] if x['seed']==seed);rows=[]
    for variant,path,deterministic in [('initial_deterministic',entry['initial'],True),('initial_stochastic',entry['initial'],False),('final_deterministic',entry['final'],True)]:
        model=TracedPPO.load(path,device='cpu');model._phase_update_index=entry['updates'] if variant=='final_deterministic' else 0;before=parameter_hash(model.policy)
        for i,task in enumerate(frozen['tasks']):
            base_seed=(frozen['initial_stochastic_seeds'] if variant=='initial_stochastic' else frozen['evaluation_seeds'])[i];record=seed==42 and variant=='final_deterministic' and i==0;folder=root/'evaluation'/f'seed{seed}'/variant;r=evaluate_model(model,folder,[task],seed=base_seed,deterministic=deterministic,record=record)[0];r.update(method=f'PPO_{variant}',training_seed=seed,checkpoint=path,checkpoint_sha256=sha(path),source='Checkpoint Evaluation');rows.append(r);print('FINAL MODEL',seed,variant,task['id'],r['success'],flush=True)
        assert parameter_hash(model.policy)==before
    write_json(root/f'seed{seed}/final_evaluation.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--part',choices=['freeze','classical','model'],required=True);p.add_argument('--seed',type=int);a=p.parse_args();torch.set_num_threads(1)
    if a.part=='freeze':freeze(a.run_root)
    elif a.part=='classical':classical(a.run_root)
    else:learned(a.run_root,a.seed)
