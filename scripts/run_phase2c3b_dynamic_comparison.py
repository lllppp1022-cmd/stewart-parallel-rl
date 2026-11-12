"""Frozen49-run ceiling:9dev+24formal+6lag+3prefix+3load+4authority. No RL."""
import _bootstrap,json,time,argparse,concurrent.futures,gc
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.workspace.feedforward_study import RUN,make_owner
from src.disturbance.recording import install_colored_capture
from src.workspace.feedforward_metrics import metrics

METHODS=['C0','C4','fourDlead','FF-V','FF-VA','FF-VA-Lead']

def description(method,amp,T,tau=0.,load=0.,pattern='none',hz=20,prefix=None):return dict(method=method,amplitude=amp,rise=T,tau=tau,load=load,pattern=pattern,hz=hz,prefix=prefix)
def name(spec):return f'{spec["method"]}_L{round(spec["amplitude"]*1000)}_T{spec["rise"]:g}_tau{spec["tau"]:g}_load{spec["load"]:g}_{spec["pattern"]}'+('_prefix' if spec['prefix'] else '')

def trial(stage,spec):
    folder=RUN/stage/name(spec);method=spec['method'];record=(stage=='formal' and spec['rise'] in [4,10] and method in ['C4','FF-VA','FF-VA-Lead']) or (stage=='load' and method.startswith('FF'))
    o,ref=make_owner(folder,method,spec['amplitude'],spec['rise'],spec['tau'],spec['load'],record)
    if record:install_colored_capture(o.recorder,o.model)
    last_probe=0.;start=time.perf_counter();err=None
    try:
        end=ref.duration if spec['prefix'] is None else spec['prefix']
        while o.data.time<end-1e-10:
            if spec['pattern']!='none' and o.n%(1000//spec['hz'])==0:
                t=float(o.data.time);p=spec['pattern'];last_probe=(.25 if p=='positive' else -.25 if p=='negative' else .25*np.sin(2*np.pi*1.3*t)) if 1.2<=t<ref.duration-1.2 else 0.;o.residual.injection.probe_N=last_probe
            if time.perf_counter()-start>360 and not o.failure:o.hold('wall_budget_exceeded',failed=True)
            o.step()
            if o.failure and o.data.time>=o.protect_end-1e-10:break
    except Exception as exc:err=repr(exc);o.event('exception',error=err);raise
    finally:o.recorder.close(dict(physics_steps=o.n,complete=bool(o.n>=round(ref.duration*1000) and not o.failure),failure=o.failure,exception=err,source='LARGE-MOTION DYNAMIC FEEDFORWARD CONTROL STUDY',terminal_reason='prefix_budget' if spec['prefix'] else 'protected_failure' if o.failure else 'finished',no_live_qpos_writes=True))
    r=metrics(folder,ref);d=pd.read_csv(folder/'physics.csv.gz');requested=d[[f'ff_desired_total_add_N_{j}' for j in range(6)]].to_numpy();actual=d[[f'ff_actual_total_add_N_{j}' for j in range(6)]].to_numpy();valid=d.failure.isna().to_numpy();diff=np.diff(actual[valid],axis=0)/.001
    r.update(spec=spec,stage=stage,method=method,requested_duration_s=ref.duration,reference=ref.audit,wall_seconds=time.perf_counter()-start,physical_steps=o.n,additional_force_peak_N=float(abs(actual[valid]).max()),additional_force_rate_peak_N_s=float(abs(diff).max()) if len(diff) else 0.,budget_projection_fraction=float(np.mean(np.any(abs(actual[valid]-requested[valid])>1e-9,axis=1))),prefix_accepted=bool(spec['prefix'] is not None and not o.failure and o.n==round(spec['prefix']*1000)),probe_effect_peak_leg_N=float(abs(d[[f'ff_probe_effect_N_{j}' for j in range(6)]]).to_numpy().max()))
    write_json(folder/'trial_result.json',r);print(stage,name(spec),'success',r['success'],'Z',r['z_rmse_mm'],'sharedPeak',r['additional_force_peak_N'],flush=True);del o,d;gc.collect();return r

def load_results(stage):return json.loads((RUN/f'{stage}_results.json').read_text(encoding='utf-8'))
def require(stage,count):
    rows=load_results(stage)
    if len(rows)!=count or not all(r['success'] for r in rows):raise RuntimeError(f'Gate {stage} failed/incomplete')

def freeze_methods():
    require('dev',9);dev=load_results('dev');scores={m:float(np.mean([r['z_rmse_mm'] for r in dev if r['method']==m])) for m in METHODS[3:]};value=dict(strong_method=min(scores,key=scores.get),development_scores=scores,parameters_sha256=sha(ROOT/'results/phase2c3b_feedforward_parameters.json'),final_timing_not_run_at_freeze=True)
    file=RUN/'methods_frozen.json'
    if file.exists():
        if json.loads(file.read_text(encoding='utf-8'))!=value:raise RuntimeError('Frozen method selection changed')
    else:write_json(file,value)

def matrix(stage):
    if stage=='dev':return [description(m,a,T) for a,T in [(.03,10.),(.05,10.),(.05,6.)] for m in METHODS[3:]]
    if stage=='formal':return [description(m,.05,T) for T in [10.,6.,4.,8.] for m in METHODS]
    if stage=='lag':
        strong=json.loads((RUN/'methods_frozen.json').read_text(encoding='utf-8'))['strong_method'];return [description(m,.05,6.,tau) for tau in [.005,.010] for m in ['C4','fourDlead',strong]]
    if stage in ['joint_preflight','load']:
        strong=json.loads((RUN/'methods_frozen.json').read_text(encoding='utf-8'))['strong_method'];return [description(m,.05,6.,.005,-5.5,prefix=7. if stage=='joint_preflight' else None) for m in ['C4','fourDlead',strong]]
    strong=json.loads((RUN/'methods_frozen.json').read_text(encoding='utf-8'))['strong_method'];return [description(strong,.05,6.,.005,-5.5,pattern=p,hz=hz) for p,hz in [('positive',20),('negative',20),('sine20',20),('sine50',50)]]

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dev','formal','lag','joint_preflight','load','authority']);p.add_argument('--worker-id',type=int,choices=[0,1]);p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.stage=='prepare':
        cfg=ROOT/'configs/phase2c3b_feedforward_study.json'
        if cfg.exists():raise FileExistsError(cfg)
        workspace=json.loads((ROOT/'configs/phase2c3a_large_workspace.json').read_text(encoding='utf-8'));write_json(cfg,dict(version='phase2c3b',workspace=workspace,maximum_formal_and_development_runs=49,physics_dt_s=.001,no_RL_training=True));write_json(RUN/'protocol_frozen.json',dict(methods=METHODS,dev_tasks=['L30-T10','L50-T10','L50-T6'],formal_timings=[10,6,4,8],independent_timing=8,parameters=ROOT.joinpath('results/phase2c3b_feedforward_parameters.json').read_text(encoding='utf-8'),candidate_count=1,maximum_runs=49,formal_before_robustness=True,selection='Lowest mean dev Z-RMSE among safe FF methods; fixed before all formal/T8/robustness evaluations',stopping='Any unexpected failure or shared-budget violation stops further stage execution; preserve logs',robustness='Matched T6; tau5/10ms without load, then tau5ms Fz=-5.5N, first7s prefix checks',authority='Fixed ±0.25N/sine requests, not learned, budget retained;20/50Hz same physical task'))
        return
    if a.stage=='formal':
        freeze_methods()
    elif a.stage=='lag':require('formal',24)
    elif a.stage=='joint_preflight':require('lag',6)
    elif a.stage=='load':
        rows=load_results('joint_preflight')
        if len(rows)!=3 or not all(r['prefix_accepted'] for r in rows):raise RuntimeError('Joint prefix gate failed')
    elif a.stage=='authority':require('load',3)
    work=matrix(a.stage);results=[]
    if a.collect:
        for worker in [0,1]:
            part=json.loads((RUN/f'{a.stage}_results_worker{worker}.json').read_text(encoding='utf-8'))
            if len(part)!=len(work[worker::2]):raise RuntimeError('Worker results incomplete')
            results+=part
        results.sort(key=lambda r:[name(s) for s in work].index(name(r['spec'])));write_json(RUN/f'{a.stage}_results.json',results)
        if a.stage=='dev':freeze_methods()
        print('COLLECTED',a.stage,len(results));return
    if a.worker_id is not None:work=work[a.worker_id::2]
    filename=f'{a.stage}_results'+(f'_worker{a.worker_id}' if a.worker_id is not None else '')+'.json'
    for spec in work:
        if (RUN/f'{a.stage}_STOP.json').exists():raise RuntimeError('Other worker triggered frozen stop rule')
        r=trial(a.stage,spec);results.append(r);write_json(RUN/filename,results)
        if not (r['prefix_accepted'] if a.stage=='joint_preflight' else r['success']) or r['additional_force_peak_N']>5+1e-8 or r['additional_force_rate_peak_N_s']>20+1e-6:
            write_json(RUN/f'{a.stage}_STOP.json',dict(spec=spec,reason='Frozen safety/budget gate failed'));raise RuntimeError('Frozen safety gate failed; no retry or threshold relaxation')
if __name__=='__main__':main()
