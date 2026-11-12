"""Bounded orchestration. Every transition requires archived validation evidence."""
import concurrent.futures
import subprocess
import sys
import time
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,write_json


def execute(args,label):
    with (RUN/f'{label}.log').open('w',encoding='utf8') as stream:
        process=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
    if process.returncode:
        raise RuntimeError(f'{label} exited {process.returncode}; see archived log')


def batch(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(execute,args,label) for args,label in jobs]
        for f in concurrent.futures.as_completed(futures):f.result()


def main():
    expected=len(cfg('tasks')['layers'])+2*sum(len(cfg('tasks')[g]) for g in ('training','development','final'))
    deadline=time.monotonic()+3600
    while True:
        count=len(list((RUN/'preflight').rglob('metrics.json')))
        proofs=(RUN/'interface_validation.json').exists() and (RUN/'resume_validation.json').exists()
        authority=len(list((RUN/'authority').glob('*/audit.json')))==3
        if count==expected and proofs and authority:break
        if time.monotonic()>deadline:raise TimeoutError('Pretraining evidence not complete within bounded wait')
        print('WAITING PREFLIGHT',count,expected,'interfaces/resume',proofs,'authority',authority,flush=True)
        time.sleep(30)
    execute(['scripts/validate_phase2c9b_protocol.py','admission'],'admission')
    batch([(['scripts/evaluate_phase2c9b.py','tune','--worker',str(i)],f'tune_{i}') for i in range(3)])
    execute(['scripts/analyze_phase2c9b.py','tune'],'adaptive_selection')
    print('TRAINING SIX MATCHED PILOTS',flush=True)
    batch([(['scripts/train_phase2c9b_ppo.py','--method',m,'--seed',str(s)],f'pilot_{m}_{s}') for m in METHODS for s in SEEDS])
    execute(['scripts/analyze_phase2c9b.py','training'],'pilot_audit')
    print('DEVELOPMENT CHECKPOINT EVALUATION',flush=True)
    batch([(['scripts/evaluate_phase2c9b.py','development','--method',m,'--seed',str(s),'--decisions',str(n)],f'dev_{m}_{s}_{n}') for m in METHODS for s in SEEDS for n in (0,8192)])
    execute(['scripts/evaluate_phase2c9b.py','collect'],'collect_pilot')
    execute(['scripts/analyze_phase2c9b.py','pilot'],'pilot_gate')
    if read(RUN/'pilot_gate.json')['passed']:
        print('COMMON PILOT GATE PASSED; MATCHED EXTENSION TO 32768',flush=True)
        batch([(['scripts/train_phase2c9b_ppo.py','--method',m,'--seed',str(s),'--continue-training'],f'extension_{m}_{s}') for m in METHODS for s in SEEDS])
        execute(['scripts/analyze_phase2c9b.py','training'],'full_audit')
        batch([(['scripts/evaluate_phase2c9b.py','development','--method',m,'--seed',str(s),'--decisions',str(n)],f'dev_{m}_{s}_{n}') for m in METHODS for s in SEEDS for n in (16384,24576,32768)])
    execute(['scripts/analyze_phase2c9b.py','selection'],'final_selection')
    n=read(RUN/'final_selection.json')['common_budget']
    print('FROZEN FINAL EVALUATION',n,flush=True)
    jobs=[(['scripts/evaluate_phase2c9b.py','final','--method',m,'--seed',str(s),'--decisions',str(n)],f'final_{m}_{s}') for m in METHODS for s in SEEDS]
    jobs += [(['scripts/evaluate_phase2c9b.py','final','--method',m,'--seed','0','--decisions','0'],f'final_{m}') for m in ('baseline','classical50','adaptive')]
    batch(jobs)
    execute(['scripts/evaluate_phase2c9b.py','collect'],'collect_final')
    execute(['scripts/plot_phase2c9b.py'],'plots')
    write_json(RUN/'core_pipeline_complete.json',dict(passed=True,budget=n,final_trials=8*9))
    print('CORE PIPELINE COMPLETE; media, GUI, reports and final tests still required',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(RUN/'pipeline_error.json',dict(error=repr(exc)))
        raise
