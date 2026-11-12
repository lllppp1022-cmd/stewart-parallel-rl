import json
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,sha

def js(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def test_actual_disabled_and_zero_tau_all_baselines_equal():
    d=pd.read_csv(ROOT/'results/phase2c0_zero_equivalence.csv');assert len(d)==4 and d.bitwise_equal.all() and d.signals_and_observer_equal.all() and d.protection_termination_equal.all() and d.maximum_difference.max()==0

def test_actual_all_research_physics_and_causality():
    v=js('results/phase2c0_validation.json');assert v['research_trials']==94 and v['physical_steps']==564000 and v['all_zoh_signals_verified'] and v['all_three_force_signals_verified'] and v['all_states_csv_exact'] and v['all_observer_memory_uses_command'] and v['all_native_and_shared_limits_verified'] and v['sensor_pairing_verified_groups']==3

def test_historical_same_time_xyz_attribution_not_combined_peaks():
    d=pd.read_csv(ROOT/'results/phase2c0_error_decomposition.csv');r=d[(d.method=='C4-Observer')&(d.window=='load_all')].iloc[0];assert abs(r.peak_time_s-3.585)<1e-9 and r.y_squared_contribution_percent>80 and r.z_squared_contribution_percent<.01
    assert abs(np.linalg.norm([r.x_at_norm_peak_mm,r.y_at_norm_peak_mm,r.z_at_norm_peak_mm])-r.peak_norm_mm)<1e-10

def test_strict_new_pairs_and_local_authority_verified():
    v=js('results/phase2c0_paired_vector_attribution.json');assert v['preload_physical_pairs_bitwise_equal']
    a=js('results/phase2c0_authority_linear_diagnostic.json');assert a['before_probe_states_equal'] and a['not_random_6d_exploration']

def test_observer_command_memory_not_hidden_drive():
    d=pd.read_csv(ROOT/'results/phase2c0_all_trials.csv');q=d[(d.method=='C4')&(d.stage=='lag_development')&d.dynamics_enabled]
    assert q.nominal_memory_hidden_drive_difference_peak.max()>1e-4

def test_frozen_gate_positive_and_classical_negative_preserved():
    g=pd.read_csv(ROOT/'results/phase2c0_benefit_gate.csv');q=g[g.method=='C4_fourD'];assert len(q)==2 and q.passed.all() and q.primary_error_reduction.min()>=.10
    fixed=g[(g.task=='eval_actuator_lag')&(g.method=='C4_fixedLead')].iloc[0];assert fixed.primary_error_reduction<0 and not fixed.passed
    v=js('results/runs/phase2c0_20261029_v1/evaluation_frozen.json');assert v['protocol_sha256']==sha(ROOT/'configs/phase2c0_protocol.json')

def test_native_gui_recording_commands_drive_and_observer_same():
    v=js('results/phase2c0_observer_validation.json');s=js('results/phase2c0_supplemental_validation.json');assert v['bitwise_equal'] and v['gui_qt_and_native'] and v['physics_steps_each']==2048 and not v['recording_errors'] and s['gui_commands_drive_observer_all_equal']

def test_actual_rgb_and_raw_motion_provenance():
    v=js('assets/videos/stewart_phase2c0_dynamic_robustness.validation.json');assert v['frames']==450 and v['all_frames_decoded'] and v['all_state_hashes_match'] and v['no_future_curves'] and v['mj_step_calls']==0
    assert all(s['unique_actual_RGB_frames']>1 for s in v['sources']);assert all(s['recorded_qpos_max_range']>1e-6 for s in v['motion']);assert v['sha256']==sha(ROOT/'assets/videos/stewart_phase2c0_dynamic_robustness.mp4')

def test_benchmark_real_steps_and_logging_nonintrusive():
    b=js('results/phase2c0_simulation_benchmark.json');s=js('results/phase2c0_supplemental_validation.json');assert b['total_benchmark_physical_steps']==4000 and all(x['physical_steps_per_wall_second']>0 for x in b['records']) and s['isolated_benchmark_journal_thinning_same_physics']
