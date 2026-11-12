import json
import numpy as np,pandas as pd,pytest
from src.model import ROOT,sha
R=ROOT/'results/runs/phase2c3a_20261103_v1'

def test_workspace_scan_and_limit_unknowns_are_preserved():
    d=pd.read_csv(ROOT/'results/phase2c3a_workspace_scan.csv');assert len(d)==101 and d.delta_z_m.min()==pytest.approx(-.1) and d.delta_z_m.max()==pytest.approx(.1)
    assert d.IK_REACHABLE.all() and d.COLLISION_COVERAGE_INCOMPLETE.all() and (d.unlimited_passive_count==30).all();assert d.passive_limit_margin_rad.isna().all()
    assert d.platform_gap_is_lower_bound.all() and (d.platform_gap_query_2m_m>.15).all()
    limits=json.loads((ROOT/'results/phase2c3a_workspace_limits.json').read_text(encoding='utf-8'));assert limits['bounds']['positive']['scan_endpoint_only'] and limits['bounds']['negative']['scan_endpoint_only']

def test_mesh_filter_audit_and_corner_evidence():
    a=json.loads((R/'model_geometry_audit.json').read_text(encoding='utf-8'));assert a['mesh_pairs']==887 and len(a['excluded_pairs'])==103 and len(a['branch_connections'])==6 and a['nativeccd']
    assert all(x['reason'] for x in a['excluded_pairs']);c=json.loads((R/'envelope_corner_audit.json').read_text(encoding='utf-8'));assert c['passed'] and c['samples']==160

def test_actual_slow_runs_have_physical_motion_and_no_early_success():
    r=json.loads((R/'slow_results.json').read_text(encoding='utf-8'));assert len(r)==6
    for x in r:
        assert x['complete'] and x['success'] and x['physical_steps']==23000
        assert x['actual_z_peak_m']>=x['amplitude_m']-.001 and x['native_saturation_steps']==0
        assert x['max_weld_translation_m']<=1e-4 and x['max_weld_angle_rad']<=1e-4
        assert x['expanded_mesh_gap_m']>.005

def test_speed_and_lag_stages_fully_archived():
    speed=json.loads((R/'speed_results.json').read_text(encoding='utf-8'));load=json.loads((R/'load_results.json').read_text(encoding='utf-8'));assert len(speed) in [3,6] and len(load)==6
    assert all(x['amplitude_m']==.05 for x in speed+load)
    for x in speed+load:assert (ROOT/x['folder']/'summary.json').exists()

def test_independent_csv_recomputation_and_clock():
    r=json.loads((R/'independent_recomputation.json').read_text(encoding='utf-8'));assert r['passed'] and len(r['trials'])>=12
    d=pd.read_csv(ROOT/'results/phase2c3a_controller_comparison.csv');assert np.isfinite(d.z_rmse_mm).all() and (d.shared_compensation_budget_peak_fraction<=1+1e-8).all()
    x=pd.read_csv(ROOT/'results/phase2c3a_dynamic_tracking.csv');assert (x.aggregate_sample_period_s==.020).all()

def test_recording_invariance_and_video_state_alignment():
    r=json.loads((R/'recording_invariance.json').read_text(encoding='utf-8'));v=json.loads((R/'video_validation.json').read_text(encoding='utf-8'));assert r['passed'] and r['bitwise_integration_state_equal'] and r['extra_physical_steps']==3600
    assert v['passed'] and all(x['all_hashes_match'] for x in v['clips']);assert any(x['actual_z_peak_m']>.049 for x in v['clips'])

def test_historical_model_controllers_checkpoints_unchanged():
    hashes=json.loads((R/'history_freeze.json').read_text(encoding='utf-8'))
    from phase2c6_history_audit import check_hashes
    check_hashes(ROOT, hashes)

def test_no_large_motion_policy_training_or_old_policy_deployment():
    assert not list(R.rglob('*.zip'))
    for path in [ROOT/'scripts/run_phase2c3a_large_motion.py',ROOT/'src/workspace/execution.py']:
        s=path.read_text(encoding='utf-8');assert '.learn(' not in s and 'PPO.load(' not in s
