"""Frozen finite classical-controller study, never trains a network."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.control.phase2b3_experiment import YawOwner
from src.disturbance.recording import install_colored_capture
from src.control.phase2b1_experiment import trial_metrics
from src.disturbance.metrics import vec

def protocol():return json.loads((ROOT/'configs/phase2b3_protocol.json').read_text(encoding='utf-8'))
def model():return json.loads((ROOT/protocol()['nominal_model_source']).read_text(encoding='utf-8'))
def run_trial(root,stage,task,label,spec,gains=None,record=False,base='C3'):
    folder=Path(root)/stage/task['name']/label
    if folder.exists():raise FileExistsError(folder)
    cfg=dict(folder=str(folder),controller=base,task_model=model(),bandwidth_rad_s=30,compensation_gain=1,leak_per_s=1,condition='nominal',record=record,pace=False,max_sim_seconds=6.,max_wall_seconds=180,exit_on_failure=True,stage=stage,split=stage,task=task['name'],trajectory=task.get('trajectory','home'),reference_scale=task.get('reference_scale',1.),disturbance=dict(start_step=task.get('start_step',2000),stop_step=task.get('stop_step',4000),wrench=task.get('wrench',[0]*6),frame='world',kind='zero' if not any(task.get('wrench',[0]*6)) else 'step'),yaw_policy=spec,actual_perturbation=dict(gain_scales=gains) if gains else {},protocol_sha256=sha(ROOT/'configs/phase2b3_protocol.json'))
    x=YawOwner(cfg)
    if record:install_colored_capture(x.recorder,x.model)
    x.run();r=trial_metrics(folder,protocol());d=pd.read_csv(folder/'physics.csv.gz');r.update(variant=label,yaw_IAE_deg_s=float(np.sum(abs(d.rpy_2-d.rpy_ref_2))*.001),yaw_peak_deg=float(max(abs(d.rpy_2-d.rpy_ref_2))));write_json(folder/'metrics.json',r)
    print(stage,task['name'],label,'PASS' if r['success'] else 'FAIL',r['failure_reason'],flush=True);return r

def zero(root):
    rows=[]
    for base in ['C0','C3']:
        t=dict(name='force10',wrench=[0,0,-10,0,0,0]);r=run_trial(root,'zero_equivalence',t,base,dict(kind='zero4'),base=base)
        old=ROOT/f'results/runs/phase2b2_20261029_v1/zero_equivalence/force10/{base}';new=Path(r['folder'])
        with np.load(old/'integration_states.npz') as a,np.load(new/'integration_states.npz') as b:equal=np.array_equal(a['states'],b['states']);maximum=float(abs(a['states']-b['states']).max())
        a=pd.read_csv(old/'physics.csv.gz',float_precision='round_trip');b=pd.read_csv(new/'physics.csv.gz',float_precision='round_trip');cols=[c for c in a.columns if c.startswith(('qpos_','qvel_','ctrl_','force_','actuatorpos_','diag_'))];assert np.array_equal(a[cols].to_numpy(),b[cols].to_numpy()) and equal
        protection=lambda p:[json.loads(x) for x in (p/'events.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(x)['type'] in ('protect_hold','motion_aborted','exception')]
        assert protection(old)==protection(new);assert json.loads((old/'summary.json').read_text())['terminal_reason']==json.loads((new/'summary.json').read_text())['terminal_reason']
        rows.append(dict(base=base,complete_state_bitwise_equal=bool(equal),maximum_difference=maximum,signals_and_observer_equal=True,protection_and_termination_equal=True))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2b3_zero_equivalence.csv',index=False);write_json(Path(root)/'zero_gate.json',dict(passed=True,rows=rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root;root.mkdir(parents=True,exist_ok=False);plan=protocol();write_json(root/'protocol_before_experiments.json',plan);write_json(ROOT/'results/phase2b3_latest_run.json',dict(root=str(root)));zero(root)
    alt=(1+.005*np.array([-1,1,-1,1,-1,1])).tolist();quiet=dict(name='quiet',wrench=[0]*6)
    first=run_trial(root,'safety_preflight',dict(name='known_alt_halfpercent_home'),'C3',dict(kind='zero4'),alt)
    if not first['success']:raise RuntimeError('Known safe gain hold failed; no escalation')
    probe=[]
    for amp in plan['yaw_probe_Nm']:
        passed=True
        for sign in [-1,1]:
            r=run_trial(root,'action_preflight',quiet,f'Mz_{sign*amp}',dict(kind='probe4',vector=[0,0,0,sign*amp]));probe.append(r);passed &= r['success']
        if not passed:raise RuntimeError('Probe failed, stop higher amplitudes')
    mass=np.array(model()['mass'])[5,5];candidates=[dict(kind='pd',kp=float(k),kd=float(2*np.sqrt(mass*k))) for k in plan['PD_kp_Nm_rad']]+[dict(kind='observer',gain=g) for g in plan['observer_gains']];write_json(root/'development_parameters_before_runs.json',dict(candidates=candidates,heldout_parameters_frozen_in_protocol=True));dev=[]
    for i,spec in enumerate(candidates):
        for task,gain in [(dict(name='nominal_home'),None),(dict(name='alt_home'),alt),(dict(name='alt_periodic',trajectory='periodic'),alt)]:dev.append((spec,run_trial(root,'development',task,f'candidate{i}',spec,gain)))
    chosen={}
    for kind in ['pd','observer']:
        scores=[]
        for spec in [x for x in candidates if x['kind']==kind]:
            group=[row for s,row in dev if s==spec];scores.append(dict(spec=spec,success=all(row['success'] for row in group),score=float(np.mean([row['yaw_IAE_deg_s'] for row in group]))))
        usable=[x for x in scores if x['success']]
        chosen[kind]=min(usable,key=lambda x:x['score']) if usable else None
        write_json(root/f'{kind}_selection.json',scores)
    if chosen['pd'] is None:raise RuntimeError('PD unsafe, stop formal matrix and preserve logs')
    eligible=[x for x in chosen.values() if x];strong=min(eligible,key=lambda x:x['score']);write_json(root/'controllers_frozen.json',dict(chosen=chosen,strongest=strong,no_final_test_tuning=True))
    pd_spec=chosen['pd']['spec'];methods={'C3':dict(kind='zero4'),'C4-PD':pd_spec,'C3_4d20':dict(kind='causal4',kp=pd_spec['kp'],kd=pd_spec['kd']),'C4_4d20':dict(kind='c4lead',base=strong['spec'])}
    if chosen['observer']:methods['C4-Observer']=chosen['observer']['spec']
    # Safety-only gain hold checks after parameter freeze; no scores tune controllers.
    approved=[]
    for name,gain in plan['frozen_test_patterns'].items():
        r=run_trial(root,'safety_preflight',dict(name=f'{name}_home_gate'),'C3',dict(kind='zero4'),gain)
        if r['success']:approved.append(name)
    write_json(root/'evaluation_frozen.json',dict(tasks=plan['test_tasks'],approved_gain_holds=approved,methods=methods,controllers_sha256=sha(root/'controllers_frozen.json')))
    unexecuted=[]
    for task in plan['test_tasks']:
        if task['gain'] not in approved:unexecuted.append(dict(task=task['name'],reason='Gain hold precheck failed'));continue
        for label,spec in methods.items():run_trial(root,'frozen_evaluation',task,label,spec,plan['frozen_test_patterns'][task['gain']],record=task['name']=='reverse_z' and label in ('C3','C4-PD','C4-Observer'))
    # Information advantage only; never used for method selection.
    for task in [plan['test_tasks'][0],plan['test_tasks'][2]]:
        if task['gain'] in approved:run_trial(root,'oracle_diagnostic',task,'oracle',dict(kind='oracle'),plan['frozen_test_patterns'][task['gain']])
    count=len(list(root.glob('*/*/*/summary.json')));assert count<=plan['maximum_research_trials'];write_json(root/'finished.json',dict(trials=count,unexecuted=unexecuted,rl_training=False));print('PHASE2B3 EXPERIMENTS FINISHED',count,flush=True)
if __name__=='__main__':main()
