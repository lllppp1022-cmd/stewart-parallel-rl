"""Bounded plant/sensor feasibility protocol, no parameter search or training."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.control.phase2c0_experiment import PhysicalOwner
from src.control.phase2b1_experiment import trial_metrics
from src.disturbance.recording import install_colored_capture

def protocol():return json.loads((ROOT/'configs/phase2c0_protocol.json').read_text(encoding='utf-8'))
def model():return json.loads((ROOT/protocol()['nominal_model_source']).read_text(encoding='utf-8'))
def task(name):return dict(protocol()['task_patterns'][name],name=name)
def config(folder,t,method='C4',tau=0,enabled=False,measurement=None,diagnostic=None,stage='development',gain=None,record=False):
    return dict(folder=str(folder),controller='C0' if method=='C0' else 'C3',yaw_policy=dict(kind='zero4') if method in ('C0','C3') else dict(kind='observer',gain=1.),task_model=model(),bandwidth_rad_s=30,compensation_gain=1,leak_per_s=1,condition='nominal',record=record,pace=False,max_sim_seconds=6.,max_wall_seconds=180,exit_on_failure=True,stage=stage,split=stage,task=t['name'],trajectory=t.get('trajectory','home'),reference_scale=t.get('reference_scale',1.),disturbance=dict(start_step=t['start_step'],stop_step=t['stop_step'],wrench=t['wrench'],frame='world',kind='zero' if not any(t['wrench']) else 'step'),actuator_dynamics=dict(enabled=enabled,tau_s=tau),measurement=measurement or {},actual_perturbation=dict(gain_scales=gain) if gain else {},diagnostic=diagnostic,method=method,protocol_sha256=sha(ROOT/'configs/phase2c0_protocol.json'))
def run_trial(root,stage,t,label,method='C4',**kwargs):
    folder=Path(root)/stage/t['name']/label
    if folder.exists():raise FileExistsError(folder)
    cfg=config(folder,t,method,stage=stage,**kwargs);start=time.perf_counter();owner=PhysicalOwner(cfg)
    if cfg['record']:install_colored_capture(owner.recorder,owner.model)
    owner.run();summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not summary['physics_steps']:raise RuntimeError('Zero-step engineering exception preserved: '+summary['failure'])
    r=trial_metrics(folder,protocol());r.update(variant=label,method=method,tau=cfg['actuator_dynamics'],measurement=cfg['measurement'],diagnostic=cfg['diagnostic'],total_wall_s=time.perf_counter()-start);write_json(folder/'metrics.json',r)
    print(stage,t['name'],label,'PASS' if r['success'] else 'FAIL',r['failure_reason'],flush=True);return r

def equivalence(root):
    rows=[]
    for method,enabled in [('C0',False),('C3',False),('C4',False),('C4',True)]:
        t=dict(name='force10',trajectory='home',wrench=[0,0,-10,0,0,0],start_step=2000,stop_step=4000);old=ROOT/f'results/runs/phase2b2_20261029_v1/zero_equivalence/force10/{method}'
        gain=None
        if method=='C4':
            t=dict(task('zload'),name='reverse_z');old=ROOT/'results/runs/phase2b3_20261029_v1/frozen_evaluation/reverse_z/C4-Observer';gain=[1.004,.996]*3
        label=f'{method}_enabled{enabled}';r=run_trial(root,'zero_equivalence',t,label,method,gain=gain,enabled=enabled,tau=0);new=Path(r['folder'])
        with np.load(old/'integration_states.npz') as a,np.load(new/'integration_states.npz') as b:equal=np.array_equal(a['states'],b['states']);maximum=float(abs(a['states']-b['states']).max())
        a=pd.read_csv(old/'physics.csv.gz',float_precision='round_trip');b=pd.read_csv(new/'physics.csv.gz',float_precision='round_trip');cols=[x for x in a.columns if x.startswith(('qpos_','qvel_','ctrl_','force_','f_cmd_','actuatorpos_','diag_'))];signals=np.array_equal(a[cols].to_numpy(),b[cols].to_numpy())
        protected=lambda p:[json.loads(x) for x in (p/'events.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(x)['type'] in ['protect_hold','motion_aborted','exception']]
        assert equal and signals and protected(old)==protected(new)
        assert json.loads((old/'summary.json').read_text())['terminal_reason']==json.loads((new/'summary.json').read_text())['terminal_reason']
        rows.append(dict(method=method,enabled=enabled,tau_s=0,bitwise_equal=bool(equal),maximum_difference=maximum,signals_and_observer_equal=bool(signals),protection_termination_equal=True))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c0_zero_equivalence.csv',index=False);write_json(Path(root)/'zero_gate.json',dict(passed=True,rows=rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root;root.mkdir(parents=True,exist_ok=False);plan=protocol();write_json(root/'protocol_before_experiments.json',plan);write_json(ROOT/'results/phase2c0_latest_run.json',dict(root=str(root)));equivalence(root)
    # New strict pairs for the old scaled dynamic reference, not invented history.
    dyn=task('dynamic');uneven=[.996,1.002,.999,1.004,.997,1.001]
    for label,load,gain in [('nominal_no_load',False,None),('nominal_with_load',True,None),('uneven_no_load',False,uneven)]:
        t=dict(dyn,name='paired_dynamic',wrench=dyn['wrench'] if load else [0]*6);run_trial(root,'paired_decomposition',t,label,gain=gain)
    # Local task-direction sensitivity; no random six-dimensional box.
    t=dict(dyn,name='authority_dynamic',wrench=[0]*6)
    for axis,magnitude in [(2,.2),(3,.01),(4,.01),(5,.005),(0,.02),(1,.02)]:
        for sign in [-1,1]:
            w=[0.]*6;w[axis]=sign*magnitude;r=run_trial(root,'authority_preflight',t,f'axis{axis}_sign{sign}',diagnostic=dict(kind='probe',wrench=w))
            if not r['success']:raise RuntimeError('Local authority preflight failed: no larger probes')
    run_trial(root,'authority_diagnostic',dict(dyn,name='uneven_xy_feedback'),'C4_localXY',gain=uneven,diagnostic=dict(kind='xy_feedback'))
    safe_tau=[];skipped=[];failed=False;matrix=[]
    for tau in plan['tau_s_candidates']:
        if failed:skipped.append(dict(tau=tau,reason='Earlier level failed; no unsupported increase'));continue
        if tau:
            r=run_trial(root,'lag_safety',task('home'),f'C4_tau{tau}',tau=tau,enabled=True)
            if not r['success']:failed=True;continue
        current=[]
        for method in plan['controllers']:
            for name in ['home','zload','dynamic']:
                r=run_trial(root,'lag_development',task(name),f'{method}_tau{tau}',method,tau=tau,enabled=bool(tau));matrix.append(r);current.append(r)
        current.append(run_trial(root,'lag_development',task('rollload'),f'C4_tau{tau}',tau=tau,enabled=bool(tau)))
        if any(not r['success'] for r in current):failed=True
        else:safe_tau.append(tau)
    # Differential lag has no gain/noise/delay mixture; only if enclosing6ms is safe.
    if .01 in safe_tau:
        for method in plan['controllers']:
            for name in ['home','dynamic']:run_trial(root,'differential_lag',task(name),method,method,tau=plan['differential_tau_s'],enabled=True)
    else:skipped.append(dict(condition='differential_tau',reason='No verified enclosing uniform lag range'))
    # Independent platform observation factors, actuator dynamics OFF.
    for delay in plan['measurement_delay_steps']:
        current=[]
        for method in plan['controllers']:current.append(run_trial(root,'delay_development',task('zload'),f'{method}_delay{delay}',method,measurement=dict(delay_steps=delay,seed=42)))
        if any(not r['success'] for r in current):skipped.append(dict(condition='higher_delays',reason='Failed smaller delay; stop escalation'));break
    for i,spec in enumerate(plan['noise_single_factors']):
        for method in plan['controllers']:run_trial(root,'noise_development',task('zload'),f'{method}_noise{i}',method,measurement=dict(spec,seed=42))
    write_json(root/'robustness_finished.json',dict(safe_uniform_tau=safe_tau,unexecuted=skipped,thresholds_unchanged=True));print('ROBUSTNESS FINISHED',flush=True)
if __name__=='__main__':main()
