"""Finite physical recording invariance + model audits; never calls a learned policy."""
import _bootstrap,json
import numpy as np
from src.model import ROOT,write_json,sha
from src.workspace.geometry import ExpandedGeometryQuery
from src.workspace.reference import LargeReference
from src.workspace.execution import LargeOwner
from src.rl.stewart_residual_env import owner_config
from src.phase1.plant import create_plant
from src.disturbance.recording import install_colored_capture
from analyze_phase2c3a_workspace import RUN

def main():
    cfg=json.loads((ROOT/'configs/phase2c3a_large_workspace.json').read_text(encoding='utf-8'));arrays=[];summaries=[]
    for record in [False,True]:
        q=ExpandedGeometryQuery();r=LargeReference(create_plant(),.05,10.,RUN/'references',q);task=dict(id='record_on' if record else 'record_off',tau_s=0.,Fz_N=0.,on_step=1700,off_step=2500);folder=RUN/'observer'/task['id'];o=LargeOwner(owner_config(folder,task,record),r,q,cfg)
        if record:install_colored_capture(o.recorder,o.model)
        try:
            for _ in range(1800):o.step()
            arrays.append(np.array(o.recorder.states));summaries.append(dict(steps=o.n,time=float(o.data.time),failure=o.failure,frames=o.recorder.nframes,record_errors=o.recorder.render_failures))
        finally:o.recorder.close(dict(physics_steps=o.n,complete=False,terminal_reason='bounded_recording_diagnostic',failure=o.failure))
    passed=np.array_equal(arrays[0],arrays[1]) and all(s['steps']==1800 and not s['failure'] and not s['record_errors'] for s in summaries)
    write_json(RUN/'recording_invariance.json',dict(passed=passed,bitwise_integration_state_equal=bool(np.array_equal(arrays[0],arrays[1])),branches=summaries,extra_physical_steps=3600,no_extra_step_from_renderer=True));assert passed
if __name__=='__main__':main()
