"""Resume post-training stages without retraining any saved policy."""
from run_phase2c9b import execute,batch
from phase2c9b_protocol import RUN,read,write_json,METHODS,SEEDS
import time

def main():
    deadline=time.monotonic()+1800
    while read(RUN/'training_audit.json')['formal_decisions']!=196608:
        if time.monotonic()>deadline:raise TimeoutError('Full audit must pass before final evaluation')
        time.sleep(10)
    audit=read(RUN/'training_audit.json')
    assert audit['formal_decisions']==196608 and audit['all_gae_recomputed']
    batch([(['scripts/evaluate_phase2c9b.py','development','--method',m,'--seed',str(s),'--decisions',str(n)],f'dev_{m}_{s}_{n}') for m in METHODS for s in SEEDS for n in (16384,24576,32768)])
    if not (RUN/'final_selection.json').exists():execute(['scripts/analyze_phase2c9b.py','selection'],'final_selection')
    n=read(RUN/'final_selection.json')['common_budget']
    jobs=[(['scripts/evaluate_phase2c9b.py','final','--method',m,'--seed',str(s),'--decisions',str(n)],f'final_{m}_{s}') for m in METHODS for s in SEEDS]
    jobs += [(['scripts/evaluate_phase2c9b.py','final','--method',m,'--seed','0','--decisions','0'],f'final_{m}') for m in ('baseline','classical50','adaptive')]
    batch(jobs)
    execute(['scripts/evaluate_phase2c9b.py','collect'],'collect_final')
    execute(['scripts/plot_phase2c9b.py'],'plots')
    write_json(RUN/'core_pipeline_complete.json',dict(passed=True,budget=n,final_trials=72))

if __name__=='__main__':main()
