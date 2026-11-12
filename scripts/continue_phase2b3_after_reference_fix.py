"""Continue only missing frozen trials; preserve the zero-row engineering failure."""
import _bootstrap,json,argparse
from pathlib import Path
from src.model import ROOT,write_json
from run_phase2b3_yaw_compensation import run_trial,protocol

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();a.run_root.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((a.parent_root/'evaluation_frozen.json').read_text(encoding='utf-8'));plan=protocol();methods=frozen['methods'];executed=[]
    for task in frozen['tasks']:
        for label,spec in methods.items():
            old=a.parent_root/'frozen_evaluation'/task['name']/label
            if (old/'summary.json').exists() and json.loads((old/'summary.json').read_text(encoding='utf-8'))['physics_steps']>0:continue
            if task['gain'] not in frozen['approved_gain_holds']:continue
            r=run_trial(a.run_root,'frozen_evaluation',task,label,spec,plan['frozen_test_patterns'][task['gain']]);executed.append(r)
    for task in [plan['test_tasks'][0],plan['test_tasks'][2]]:executed.append(run_trial(a.run_root,'oracle_diagnostic',task,'oracle',dict(kind='oracle'),plan['frozen_test_patterns'][task['gain']]))
    roots=[a.parent_root,a.run_root];attempts=list(p for root in roots for p in root.glob('*/*/*/summary.json'));assert len(attempts)<=plan['maximum_research_trials']
    write_json(ROOT/'results/phase2b3_run_index.json',dict(roots=[str(x) for x in roots],attempts=len(attempts),zero_row_engineering_failure_preserved=True,parameters_unchanged=True,completed=True))
    write_json(a.run_root/'finished.json',dict(executed=len(executed),total_attempts=len(attempts),rl_training=False));print('FROZEN CONTINUATION FINISHED',len(attempts),flush=True)
if __name__=='__main__':main()
