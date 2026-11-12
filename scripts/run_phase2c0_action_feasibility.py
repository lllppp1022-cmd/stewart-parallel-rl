"""Run at most two eligible classes with three frozen causal methods each."""
import _bootstrap
import argparse,json
from pathlib import Path
from src.model import ROOT,write_json,sha
from src.analysis.rl_task_feasibility import select_candidates
from run_phase2c0_robustness import protocol,task,run_trial

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root;plan=protocol();assert (root/'robustness_finished.json').exists();chosen=select_candidates(root,plan)
    methods=dict(C4=None,C4_fourD=dict(kind='four_d_lead'),C4_fixedLead=dict(kind='fixed_command_lead'))
    write_json(root/'evaluation_frozen.json',dict(candidates=chosen,methods=methods,task=plan['frozen_evaluation_task'],gate=plan['benefit_gate'],protocol_sha256=sha(ROOT/'configs/phase2c0_protocol.json'),C4_frozen_source='Phase2B3 selected Observer gain1, clip.3; unchanged'))
    results=[]
    for i,candidate in enumerate(chosen):
        t=dict(plan['frozen_evaluation_task'],name='eval_'+candidate['kind']);kwargs=dict(tau=candidate['tau']['tau_s'],enabled=candidate['tau']['enabled'],measurement=candidate['measurement'])
        for method,diagnostic in methods.items():results.append(run_trial(root,'frozen_evaluation',t,method,diagnostic=diagnostic,record=i==0,**kwargs))
        if candidate['kind']=='actuator_lag':run_trial(root,'privileged_diagnostic',t,'privileged',diagnostic=dict(kind='privileged'),**kwargs)
    if not chosen:
        # Recording only: repeat direct and safest nonzero C4, not a fabricated RL task.
        safe=json.loads((root/'robustness_finished.json').read_text(encoding='utf-8'))['safe_uniform_tau'];safe_nonzero=[t for t in safe if t]
        for tau in [0]+([max(safe_nonzero)] if safe_nonzero else []):run_trial(root,'visualization_only',task('zload'),f'C4_tau{tau}',tau=tau,enabled=bool(tau),record=True)
    count=len(list(root.glob('*/*/*/summary.json')));assert count<=plan['maximum_research_trials'];write_json(root/'finished.json',dict(research_trials=count,eligible_classes=len(chosen),unexecuted='No eligible class: causal final evaluation not run' if not chosen else [],no_RL=True));print('FEASIBILITY FINISHED',count,'classes',len(chosen),flush=True)
if __name__=='__main__':main()
