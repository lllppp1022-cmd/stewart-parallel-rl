"""Read actual experiment evidence in addition to live controller unit tests."""
import json
import numpy as np,pandas as pd
from src.model import ROOT,sha

def js(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def test_zero4_actual_complete_state_and_observer_equivalence():
    d=pd.read_csv(ROOT/'results/phase2b3_zero_equivalence.csv');assert len(d)==2 and d.complete_state_bitwise_equal.all() and d.signals_and_observer_equal.all() and d.protection_and_termination_equal.all() and d.maximum_difference.max()==0

def test_actual_yaw_probe_positive_and_negative_signs():
    root=ROOT/'results/runs/phase2b3_20261029_v1/action_preflight/quiet'
    for amp in [.005,.025,.1,.25]:
        for sign in [-1,1]:
            d=pd.read_csv(root/f'Mz_{sign*amp}'/'physics.csv.gz',usecols=['input_step','rpy_2','rpy_ref_2','failure']);s=(d.input_step>=1000)&(d.input_step<1700);assert sign*(d.rpy_2[s]-d.rpy_ref_2[s]).mean()>0 and d.failure.isna().all()

def test_full_rate_physical_validation_actual_and_safe():
    v=js('results/phase2b3_validation.json');assert v['trials']==58 and v['physical_steps']==348000 and v['all_state_archives_exact'] and v['all_shared_budgets_valid'] and v['all_native_limits_valid'] and v['all_channel_isolation_valid'] and v['max_scratch_mapping_error']<2e-8
    assert len(v['engineering_failures'])==1 and v['engineering_failures'][0]['physics_steps']==0

def test_frozen_independent_evaluation_no_hidden_changes():
    v=js('results/runs/phase2b3_20261029_v1/evaluation_frozen.json');assert v['controllers_sha256']==sha(ROOT/'results/runs/phase2b3_20261029_v1/controllers_frozen.json')
    f=pd.read_csv(ROOT/'results/phase2b3_controller_comparison.csv');assert len(f)==30 and f.task.nunique()==6 and f.variant.nunique()==5 and not f.privileged.any()
    for task,g in f.groupby('task'):
        configs=[js(str((__import__('pathlib').Path(folder)/'config.json').as_posix())) for folder in g.folder]
        assert all(c['actual_perturbation']==configs[0]['actual_perturbation'] and c['disturbance']==configs[0]['disturbance'] and c.get('reference_scale')==configs[0].get('reference_scale') for c in configs)

def test_actual_gui_and_recording_nonintrusive():
    g=js('results/phase2b3_observer_validation.json');assert g['bitwise_equal'] and g['physics_steps_each']==2048 and g['gui_qt_and_native'] and not g['interrupted'] and g['mirror_exitcode']==0 and not g['recording_errors']

def test_actual_rgb_frame_state_validation():
    v=js('assets/videos/stewart_phase2b3_yaw_compensation.validation.json');assert v['frames']==450 and v['unique_decoded_frames']==450 and v['all_state_hashes_match'] and v['all_frames_decoded'] and v['no_future_curves'] and v['mj_step_calls']==0
    assert v['sha256']==sha(ROOT/'assets/videos/stewart_phase2b3_yaw_compensation.mp4')
