"""Finite classical-control development, gated observer validation, frozen evaluation."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.control.phase2b1_experiment import RobustOwner,trial_metrics
from src.control.nominal_task_model import build_model
from src.control.observer_replay import replay_observer
from src.disturbance.metrics import vec
from src.disturbance.recording import install_colored_capture

def protocol():return json.loads((ROOT/'configs/phase2b1_protocol.json').read_text(encoding='utf-8'))
def trial(root,task,method,stage,variant=None,record=False,safety_only=False,**params):
    p=protocol();folder=Path(root)/stage/task['name']/(variant or method);w=task['wrench'];cfg=dict(folder=str(folder),controller=method,condition='nominal',record=record,pace=False,trajectory=task['trajectory'],max_sim_seconds=6.,max_wall_seconds=180,exit_on_failure=True,stage=stage,split=task.get('split',stage),task=task['name'],disturbance=dict(start_step=task['start_step'],stop_step=task['stop_step'],wrench=w,frame=task.get('frame','world'),kind='zero' if not any(w) else 'composite' if sum(x!=0 for x in w)>1 else 'step'),frozen_original_controller_sha256=sha(ROOT/'src/phase1/controllers.py'),frozen_original_safety_sha256=sha(ROOT/'configs/phase1_protocol.json'),protocol_sha256=sha(ROOT/'configs/phase2b1_protocol.json'),**params)
    x=RobustOwner(cfg)
    if record:install_colored_capture(x.recorder,x.model)
    x.run();r=trial_metrics(folder,p);write_json(folder/'metrics.json',r)
    print(stage,task['name'],variant or method,'PASS' if r['success'] else 'FAIL',r['failure_reason'],'' if safety_only else f"objective={r['objective']}",flush=True)
    return r

def observer_validation(root,model,dev):
    p=protocol();records=[];passed=[]
    for omega in p['observer_bandwidth_candidates_rad_s']:
        score=0.;valid=True
        for task in dev:
            folder=root/'development'/task['name']/'C0';estimate,df=replay_observer(folder,model,omega);truth=vec(df,'external_world',6);on=task['start_step'];off=task['stop_step'];steady=slice(off-400,off-50);pre=slice(1500,1900);tail=slice(off+800,off+1200)
            scales=np.array([1,1,5,.25,.25,.25]);rmse=np.sqrt(np.mean((estimate-truth)**2,axis=0));target=[j for j in (2,3,4) if task['wrench'][j]!=0]
            direction_ok=all(np.mean(estimate[steady,j])*task['wrench'][j]>0 for j in target)
            steady_ok=all(abs(np.mean(estimate[steady,j])-task['wrench'][j])<=.5*abs(task['wrench'][j]) for j in target)
            zero_ok=bool(abs(estimate[pre,2]).mean()<.5 and abs(estimate[pre,3:5]).mean()<.02)
            delays=[];decays=[]
            for j in target:
                magnitude=abs(task['wrench'][j]);ix=np.flatnonzero(abs(estimate[on:off,j])>=.1*magnitude);delays.append(float(ix[0]*.001) if len(ix) else None);iy=np.flatnonzero(abs(estimate[off:,j])<=.1*magnitude);decays.append(float(iy[0]*.001) if len(iy) else None)
            gate=bool(direction_ok and steady_ok and zero_ok and all(x is not None and x<=1 for x in delays+decays));valid&=gate;score+=float(np.mean(rmse[[2,3,4]]/scales[[2,3,4]]))
            record=dict(task=task['name'],bandwidth_rad_s=omega,gate_passed=gate,direction_ok=direction_ok,steady_ok=steady_ok,zero_bias_gate=zero_ok,force_bias_pre_N=float(np.mean(estimate[pre,2])),roll_bias_pre_Nm=float(np.mean(estimate[pre,3])),force_rmse_N=float(rmse[2]),roll_rmse_Nm=float(rmse[3]),pitch_rmse_Nm=float(rmse[4]),response_delay_s=max(delays) if delays and all(x is not None for x in delays) else None,decay_delay_s=max(decays) if decays and all(x is not None for x in decays) else None,force_steady_estimate_N=float(np.mean(estimate[steady,2])),roll_steady_estimate_Nm=float(np.mean(estimate[steady,3])))
            records.append(record);out=root/'observer_replay'/f'omega{omega}'/task['name'];out.mkdir(parents=True);pd.DataFrame(dict(time_s=df.time_s,**{f'estimate_{j}':estimate[:,j] for j in range(6)},**{f'world_truth_{j}':truth[:,j] for j in range(6)})).to_csv(out/'observer.csv.gz',index=False)
        if valid:passed.append((score,omega))
    pd.DataFrame(records).to_csv(ROOT/'results/phase2b1_observer_validation.csv',index=False);write_json(root/'observer_validation.json',dict(records=records,truth_used_offline_only=True,physical_reintegration_steps=0))
    if not passed:return None
    return min(passed)[1]

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run-root',type=Path,required=True);args=a.parse_args();root=args.run_root;root.mkdir(parents=True,exist_ok=False);p=protocol();write_json(root/'protocol_before_development.json',p);write_json(ROOT/'results/phase2b1_latest_run.json',dict(root=str(root)));print('RUN_ROOT='+str(root),flush=True)
    model=build_model();write_json(root/'nominal_task_model.json',model)
    # Small new Pitch load first; no controller tuning uses these precheck responses.
    for amp in (.05,.10):
        task=dict(name=f'pitch_prerequisite_{amp}',trajectory='home',wrench=[0,0,0,0,amp,0],start_step=2000,stop_step=4000)
        if not trial(root,task,'C0','test_safety_precheck',safety_only=True)['success']:raise RuntimeError('New Pitch-load precheck failed; stop this direction')
    independent=[]
    for task in p['independent_test_proposals']:
        r=trial(root,task,'C0','test_safety_precheck',safety_only=True)
        if r['success']:independent.append(dict(task,split='independent_test'))
    write_json(root/'independent_test_frozen.json',dict(tasks=independent,safety_only_prechecked=True,not_used_in_tuning=True,not_claimed_operator_blind=True))
    dev=p['development_tasks'];leak_scores=[]
    for leak in p['lambda_candidates_per_s']:
        rows=[trial(root,t,'C1-L','development',variant=f'C1L_lambda{leak}',leak_per_s=leak) for t in dev]
        leak_scores.append(dict(leak_per_s=leak,score=float(np.mean([r['objective'] for r in rows])) if all(r['success'] for r in rows) else None,all_success=all(r['success'] for r in rows)))
    valid=[r for r in leak_scores if r['all_success']]
    if not valid:raise RuntimeError('No safe C1-L development candidate')
    chosen_leak=min(valid,key=lambda r:r['score'])['leak_per_s'];write_json(root/'leak_selection.json',dict(candidates=leak_scores,chosen=chosen_leak,development_only=True))
    # One full six-second C1 lambda-zero equivalence, separate from tuning.
    ref=trial(root,dev[1],'C1','equivalence');a0=root/'development'/dev[1]['name']/'C1L_lambda0/integration_states.npz';b0=Path(ref['folder'])/'integration_states.npz'
    with np.load(a0) as x,np.load(b0) as y:equivalent=bool(np.array_equal(x['states'],y['states']))
    write_json(root/'lambda_zero_equivalence.json',dict(bitwise_equal=equivalent,physics_steps=6000));assert equivalent
    for t in dev:trial(root,t,'C0','development')
    omega=observer_validation(root,model,dev);comp_scores=[];chosen_gain=None
    if omega is not None:
        for gain in p['compensation_gain_candidates']:
            rows=[trial(root,t,'C3','development',variant=f'C3_gain{gain}',task_model=model,bandwidth_rad_s=omega,compensation_gain=gain) for t in dev]
            comp_scores.append(dict(gain=gain,score=float(np.mean([r['objective'] for r in rows])) if all(r['success'] for r in rows) else None,all_success=all(r['success'] for r in rows)))
        valid=[r for r in comp_scores if r['all_success']]
        if valid:chosen_gain=min(valid,key=lambda r:r['score'])['gain']
    chosen=dict(leak_per_s=chosen_leak,bandwidth_rad_s=omega,compensation_gain=chosen_gain,compensation_candidates=comp_scores,C3_status='READY' if chosen_gain is not None else 'BLOCKED',no_final_data_used=True);write_json(root/'controller_parameters_frozen.json',chosen)
    historical=[dict(name='benchmark_force10',trajectory='home',wrench=[0,0,-10,0,0,0]),dict(name='benchmark_rollp05',trajectory='home',wrench=[0,0,0,.5,0,0]),dict(name='benchmark_rolln05',trajectory='home',wrench=[0,0,0,-.5,0,0]),dict(name='benchmark_z_roll_combo',trajectory='z_roll',wrench=[0,0,-5,.1,0,0]),dict(name='benchmark_legacy_local',trajectory='z_roll',wrench=[0,0,-10,.5,0,0],frame='site-local')]
    historical=[dict(t,start_step=2000,stop_step=4000,split='historical_benchmark') for t in historical];methods=['C0','C1','C2','C1-L']+(['C3'] if chosen_gain is not None else []);tasks=historical+independent
    final=dict(tasks=tasks,methods=methods,parameters=chosen,parameters_sha256=sha(root/'controller_parameters_frozen.json'),independent_tests_sha256=sha(root/'independent_test_frozen.json'),frozen_before_evaluation=True);write_json(root/'final_protocol_frozen.json',final)
    for t in tasks:
        for method in methods:
            params=dict(leak_per_s=chosen_leak) if method=='C1-L' else dict(task_model=model,bandwidth_rad_s=omega,compensation_gain=chosen_gain) if method=='C3' else {}
            trial(root,t,method,'final',record=t['name']=='benchmark_force10' and method in ('C0','C1','C1-L','C3'),final_protocol_sha256=sha(root/'final_protocol_frozen.json'),**params)
    write_json(root/'finished.json',dict(final_trials=len(tasks)*len(methods),C3_status=chosen['C3_status'],no_rl=True));print('FINISHED',flush=True)
if __name__=='__main__':main()
