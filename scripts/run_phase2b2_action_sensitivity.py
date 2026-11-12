"""Bounded residual interface/development/evaluation; frozen controllers, no learning."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.control.phase2b2_experiment import ResidualOwner
from src.control.phase2b1_experiment import trial_metrics
from src.disturbance.recording import install_colored_capture

def defaults():
    old=ROOT/'results/runs/phase2b1_20261028_v1';return json.loads((old/'nominal_task_model.json').read_text(encoding='utf-8'))
def protocol():return json.loads((ROOT/'configs/phase2b2_protocol.json').read_text(encoding='utf-8'))
def task(name='force10',trajectory='home',wrench=None,start_step=2000,stop_step=4000):return dict(name=name,trajectory=trajectory,wrench=wrench or [0,0,-10,0,0,0],start_step=start_step,stop_step=stop_step)
def run_trial(root,t,base='C3',action=None,stage='preflight',variant=None,record=False,**extra):
    p=protocol();folder=Path(root)/stage/t['name']/(variant or base);w=t['wrench'];cfg=dict(folder=str(folder),controller=base,condition='nominal',record=record,pace=False,trajectory=t['trajectory'],max_sim_seconds=6.,max_wall_seconds=240,exit_on_failure=True,stage=stage,split=extra.pop('split',stage),task=t['name'],task_model=defaults(),bandwidth_rad_s=30,compensation_gain=1.,leak_per_s=1.,disturbance=dict(start_step=t['start_step'],stop_step=t['stop_step'],wrench=w,frame=t.get('frame','world'),kind='zero' if not any(w) else 'composite' if sum(x!=0 for x in w)>1 else 'step'),action=action or dict(kind='zero',space='task',frequency=20),protocol_sha256=sha(ROOT/'configs/phase2b2_protocol.json'),**extra)
    x=ResidualOwner(cfg)
    if record:install_colored_capture(x.recorder,x.model)
    x.run();r=trial_metrics(folder,p);r.update(variant=folder.name,action_kind=cfg['action']['kind'],action_space=cfg['action'].get('space','leg'),frequency=cfg['action'].get('frequency',1000),privileged=stage=='oracle_diagnostic',actual_perturbation=cfg.get('actual_perturbation',{}),measurement=cfg.get('measurement',{}));write_json(folder/'metrics.json',r)
    print(stage,t['name'],folder.name,'PASS' if r['success'] else 'FAIL',r['failure_reason'],flush=True);return r

def zero_checks(root):
    old=ROOT/'results/runs/phase2b1_20261028_v1/final/benchmark_force10';rows=[]
    for base in ('C0','C3'):
        r=run_trial(root,task(),base,stage='zero_equivalence')
        with np.load(old/base/'integration_states.npz') as a,np.load(Path(r['folder'])/'integration_states.npz') as b:equal=bool(np.array_equal(a['states'],b['states']));difference=float(abs(a['states']-b['states']).max())
        x=pd.read_csv(old/base/'physics.csv.gz',float_precision='round_trip');y=pd.read_csv(Path(r['folder'])/'physics.csv.gz',float_precision='round_trip')
        columns=[f'{prefix}_{j}' for prefix,n in [('qpos',43),('qvel',42),('ctrl',12),('force',6),('p',3),('rpy',3),('actuatorpos',6)] for j in range(n)];signals=bool(np.array_equal(x[columns].to_numpy(),y[columns].to_numpy()));rows.append(dict(base=base,full_integration_state_bitwise_equal=equal,all_physical_signals_equal=signals,maximum_state_difference=difference,folder=r['folder'],same_terminal_reason=json.loads((old/base/'summary.json').read_text())['terminal_reason']==json.loads((Path(r['folder'])/'summary.json').read_text())['terminal_reason']));assert equal and signals
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2b2_zero_residual_equivalence.csv',index=False);write_json(root/'zero_gate.json',dict(passed=True,rows=rows));return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();root=args.run_root;root.mkdir(parents=True,exist_ok=False);write_json(root/'protocol_before_experiments.json',protocol());write_json(ROOT/'results/phase2b2_latest_run.json',dict(root=str(root)));zero_checks(root)
    quiet=task('quiet',wrench=[0]*6);passed_amp=0.
    for amp in (.1,.25,.5):
        r=run_trial(root,quiet,'C0',dict(kind='probe',space='leg',frequency=20,vector=[amp,0,0,0,0,0]),variant=f'leg1_amp{amp}')
        if not r['success']:break
        passed_amp=amp
    if passed_amp<.1:raise RuntimeError('Leg action gate failed')
    for j in range(6):
        for sign in (-1,1):
            vector=[0.]*6;vector[j]=sign*.1;r=run_trial(root,quiet,'C0',dict(kind='probe',space='leg',frequency=20,vector=vector),variant=f'leg{j+1}_sign{sign}')
            if not r['success']:raise RuntimeError('Per-leg gate failed; no stage B')
    for j,amp in enumerate((.5,.025,.025)):
        for sign in (-1,1):
            vector=[0.]*3;vector[j]=sign*amp;r=run_trial(root,quiet,'C0',dict(kind='probe',space='task',frequency=20,vector=vector),variant=f'task{j}_sign{sign}')
            if not r['success']:raise RuntimeError('Task action gate failed; no stage B')
    for space,vector in [('task',[.5,.01,-.01]),('leg',[.1,0,0,0,0,0])]:
        for frequency in (20,50):run_trial(root,quiet,'C0',dict(kind='ramp',space=space,frequency=frequency,vector=vector),variant=f'ramp_{space}_{frequency}')
    run_trial(root,quiet,'C3',dict(kind='probe',space='task',frequency=50,vector=[.5,0,0]),variant='C3_known_input_no_external')
    # New tasks: safety-only gate, before any residual parameter choice.
    held=[task('test_force8_time',wrench=[0,0,-8,0,0,0],start_step=1650,stop_step=3550),task('test_periodic_roll',trajectory='periodic',wrench=[0,0,0,-.4,0,0],start_step=1350,stop_step=3950)]
    safe=[]
    for t in held:
        if run_trial(root,t,'C3',stage='test_safety_precheck',use_residual=False)['success']:safe.append(t)
    write_json(root/'independent_tests_frozen.json',dict(tasks=safe,safety_only_prechecked=True,not_used_to_select_actions=True))
    dev=[task(),task('roll05',wrench=[0,0,0,.5,0,0])];rows=[]
    for kind in ('damping','lead'):
        for frequency in (20,50):
            for t in dev:rows.append(run_trial(root,t,'C3',dict(kind=kind,space='task',frequency=frequency),stage='development',variant=f'C3_{kind}_{frequency}'))
    for frequency in (20,50):
        rows.append(run_trial(root,dev[0],'C0',dict(kind='damping',space='task',frequency=frequency),stage='development',variant=f'C0_damping_{frequency}'))
        rows.append(run_trial(root,dev[0],'C3',dict(kind='damping',space='leg',frequency=frequency),stage='development',variant=f'C3_leg_damping_{frequency}'))
    candidates=[]
    for kind in ('damping','lead'):
        for frequency in (20,50):
            sub=[r for r in rows if r['variant']==f'C3_{kind}_{frequency}'];candidates.append(dict(kind=kind,frequency=frequency,score=float(np.mean([r['objective'] for r in sub])) if all(r['success'] for r in sub) else None))
    valid=[x for x in candidates if x['score'] is not None]
    selected=min(valid,key=lambda x:x['score']) if valid else dict(kind='zero',frequency=20,score=None);write_json(root/'actions_frozen.json',dict(selected=selected,candidates=candidates,all_budgets_shared=True,policy='Finite development-only normalized two-window IAE; control cost separately evaluated, not claimed Pareto-optimal'))
    methods=[dict(label=m,base=m,use_residual=False) for m in ('C0','C1','C1-L','C2','C3')]+[dict(label=f'{base}_{space}',base=base,use_residual=True,action=dict(kind=selected['kind'] if base=='C3' else 'damping',space=space,frequency=selected['frequency'])) for base in ('C0','C3') for space in ('task','leg')]
    write_json(root/'evaluation_frozen.json',dict(tasks=safe,methods=methods,action_parameters_sha256=sha(root/'actions_frozen.json'),frozen_before_evaluation=True))
    for t in safe:
        for method in methods:
            spec=dict(method);label=spec.pop('label');base=spec.pop('base');r=run_trial(root,t,base,stage='frozen_evaluation',variant=label,record=t['name']=='test_force8_time' and label in ('C3','C3_task','C3_leg'),evaluation_sha256=sha(root/'evaluation_frozen.json'),**spec)
    for t in dev:
        for base in ('C0','C3'):run_trial(root,t,base,dict(kind='oracle',space='leg',frequency=1000),stage='oracle_diagnostic',variant=base+'_oracle')
    write_json(root/'actions_finished.json',dict(complete=True,rl_training=False));print('ACTIONS FINISHED',flush=True)
if __name__=='__main__':main()
