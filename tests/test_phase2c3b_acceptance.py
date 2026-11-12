import json
from pathlib import Path
import numpy as np,pandas as pd,pytest
from src.model import ROOT,sha
from src.workspace.feedforward_study import RUN

def test_frozen_parameter_origin_and_no_final_fitting():
    p=json.loads((ROOT/'results/phase2c3b_feedforward_parameters.json').read_text(encoding='utf-8'));source=json.loads(Path(p['source']).read_text(encoding='utf-8'));assert sha(p['source'])==p['source_sha256'];assert p['candidate_count']==1 and p['ka_kg']!=20
    assert p['kv_N_s_m']==np.array(source['damping'])[2,2] and p['ka_kg']==np.array(source['mass'])[2,2]
    f=json.loads((RUN/'methods_frozen.json').read_text(encoding='utf-8'));assert f['final_timing_not_run_at_freeze'] and sha(ROOT/'results/phase2c3b_feedforward_parameters.json')==f['parameters_sha256']

def test_all49_bounded_tasks_and_prefix_semantics():
    d=pd.read_csv(ROOT/'results/phase2c3b_controller_comparison.csv');assert len(d)==49
    prefix=d[d.stage=='joint_preflight'];assert len(prefix)==3 and prefix.prefix_accepted.all() and not prefix.success.any() and not prefix.complete.any()
    full=d[d.stage!='joint_preflight'];assert len(full)==46 and full.complete.all() and full.success.all();assert (full.native_saturation_steps==0).all()

def test_main_budget_and_independent_recompute():
    r=json.loads((RUN/'independent_audit.json').read_text(encoding='utf-8'));assert r['passed'] and len(r['trials'])==49
    assert max(x['extra_budget_peak_N'] for x in r['trials'])<=5+1e-8;assert max(x['extra_budget_rate_peak_N_s'] for x in r['trials'])<=20+1e-6

@pytest.mark.parametrize('method',['C0','C4','fourDlead'])
def test_frozen_classical_dynamics_match_history(method):
    for T in [4,6,10]:
        old=ROOT/f'results/runs/phase2c3a_20261103_v1/dynamics/{method}_L50_T{T}_tau0_load0/integration_states.npz';new=RUN/f'formal/{method}_L50_T{T}_tau0_load0_none/integration_states.npz'
        with np.load(old) as a,np.load(new) as b:np.testing.assert_array_equal(a['states'],b['states'])

def test_still_reference_ff_zero_and_command_force_separate():
    folder=RUN/'formal/FF-VA_L50_T4_tau0_load0_none';d=pd.read_csv(folder/'physics.csv.gz');hold=(d.time_s<1)|(d.time_s>10);assert np.max(abs(d.ff_velocity_force_N[hold]))==0 and np.max(abs(d.ff_acceleration_force_N[hold]))==0
    l=pd.read_csv(RUN/'lag/FF-VA_L50_T6_tau0.01_load0_none/physics.csv.gz');assert max(abs(l.F_command_0-l.F_drive_0))>1e-5;np.testing.assert_allclose(l.F_drive_0,l.force_0,atol=1e-12)

def test_new_timing_and_matched_perturbations():
    d=pd.read_csv(ROOT/'results/phase2c3b_controller_comparison.csv');s=d[(d.stage=='formal')&(d.rise_s==8)];assert len(s)==6 and (s.physical_steps==19000).all()
    p=pd.read_csv(ROOT/'results/phase2c3b_paired_time_series.csv');assert p.groupby('method').size().eq(15000).all();assert np.isfinite(p.delta_lag_position_mm).all()

def test_actual_viewer_and_video():
    r=json.loads((RUN/'live_viewer_validation.json').read_text(encoding='utf-8'));v=json.loads((RUN/'video_validation.json').read_text(encoding='utf-8'));assert r['passed'] and r['physics_steps_each']==4000 and r['single_physics_owner_thread'];assert v['passed'] and v['decoded_frames']>0;assert all(x['all_hashes_match'] for x in v['sources'])

def test_history_preservation_and_no_learning_models():
    hashes=json.loads((RUN/'history_freeze.json').read_text(encoding='utf-8'))
    from phase2c6_history_audit import check_hashes
    check_hashes(ROOT, hashes)
    assert not list(RUN.rglob('*.zip'))
