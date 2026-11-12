"""Bounded artifact-only workflow after frozen final evaluation; no training."""
import time
from run_phase2c9b import execute,batch
from phase2c9b_protocol import RUN,read,write_json

def main():
    deadline=time.monotonic()+3600
    while not (RUN/'core_pipeline_complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Frozen final evaluation not completed')
        time.sleep(10)
    execute(['scripts/analyze_phase2c9b_history.py'],'physical_final_audit')
    batch([(['scripts/ablate_phase2c9b_inputs.py'],'input_sensitivity'),
           (['scripts/plot_phase2c9b_behavior.py'],'behavior_plots'),
           (['scripts/tensorboard_phase2c9b.py'],'derived_tensorboard'),
           (['scripts/render_phase2c9b_videos.py','single_vs_history'],'video_comparison')])
    for method in ('single_frame','history_5frame'):
        execute(['scripts/view_phase2c9b_policy.py','--method',method,'--seed','42','--validate-steps','1000'],f'viewer_{method}')
    execute(['-m','pytest','tests/test_phase2c9b.py','-q','--junitxml',str(RUN/'new_tests.xml')],'new_tests')
    write_json(RUN/'postprocessing_complete.json',dict(passed=True,training_started=False))

if __name__=='__main__':main()
