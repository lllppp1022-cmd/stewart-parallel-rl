"""Acceptance evidence complements live unit tests; reads real full-rate validation."""
import json
import pandas as pd
from src.model import ROOT,sha

def artifact(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def test_real_full_rate_archives_and_budget():
    v=artifact('results/phase2b2_validation.json')
    assert v['research_trials']==90 and v['research_physical_steps']==534584
    assert v['full_rate_budget_and_native_limits_passed'] and v['actual_generalized_force_recomputed']
    assert v['kinematic_map_independent_scratch_max_error']<2e-8

def test_real_failure_preserved_and_censored():
    d=pd.read_csv(ROOT/'results/phase2b2_all_trials.csv');r=d[~d.success].iloc[0]
    assert r.variant=='asymmetric_0.01_C3' and not r.load_started and not r.unload_reached
    assert r.physical_steps==584 and pd.isna(r.load_recovery_s) and 'CENSORED' in r.load_recovery_status

def test_native_gui_and_recording_bitwise_equivalent():
    v=artifact('results/phase2b2_observer_validation.json')
    assert v['gui_qt_and_native'] and v['bitwise_equal'] and v['physics_steps_each']==2048 and not v['interrupted'] and not v['recording_errors']

def test_actual_video_state_time_and_frame_proof():
    v=artifact('assets/videos/stewart_phase2b2_residual_action_validation.validation.json')
    assert v['frames']==450 and v['unique_decoded_frames']==450 and v['all_state_hashes_match'] and v['all_frames_decoded'] and v['no_future_curves']
    assert v['sha256']==sha(ROOT/'assets/videos/stewart_phase2b2_residual_action_validation.mp4')

def test_evaluation_frozen_before_selection_and_oracle_separate():
    p=ROOT/'results/runs/phase2b2_20261029_v1'
    v=artifact('results/runs/phase2b2_20261029_v1/evaluation_frozen.json')
    assert v['action_parameters_sha256']==sha(p/'actions_frozen.json')
    d=pd.read_csv(ROOT/'results/phase2b2_all_trials.csv');assert not d[d.stage=='frozen_evaluation'].privileged.any()
    assert d[d.privileged].stage.eq('oracle_diagnostic').all()

def test_known_residual_not_treated_as_external_truth():
    v=artifact('results/phase2b2_known_input_check.json');assert max(abs(x) for x in v['observer_Fz_range_N'])<.001 and v['corrected_memory_steps']>0
