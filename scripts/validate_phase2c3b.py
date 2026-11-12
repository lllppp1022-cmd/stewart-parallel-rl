"""Read-only acceptance audit for the frozen Phase2C3B actual experiments."""
import _bootstrap,json,platform,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,sha,write_json
from src.workspace.feedforward_study import RUN

def main():
    d=pd.read_csv(ROOT/'results/phase2c3b_controller_comparison.csv');full=d[d.stage!='joint_preflight'];prefix=d[d.stage=='joint_preflight']
    old=json.loads((RUN/'history_freeze.json').read_text(encoding='utf-8'));changed=[f for f,h in old.items() if not (ROOT/f).is_file() or sha(ROOT/f)!=h]
    baseline=[]
    for method in ['C0','C4','fourDlead']:
        for T in [4,6,10]:
            before=ROOT/f'results/runs/phase2c3a_20261103_v1/dynamics/{method}_L50_T{T}_tau0_load0/integration_states.npz';after=RUN/f'formal/{method}_L50_T{T}_tau0_load0_none/integration_states.npz'
            with np.load(before) as a,np.load(after) as b:equal=bool(np.array_equal(a['states'],b['states']))
            baseline.append(dict(method=method,rise_s=T,bitwise_equal=equal))
    independent=json.loads((RUN/'independent_audit.json').read_text(encoding='utf-8'));viewer=json.loads((RUN/'live_viewer_validation.json').read_text(encoding='utf-8'));video=json.loads((RUN/'video_validation.json').read_text(encoding='utf-8'))
    peaks={k:float(full[k].max()) for k in ['max_force_N','max_force_rate_N_s','max_weld_translation_m','max_weld_angle_rad','max_jacobian_condition','additional_force_peak_N','additional_force_rate_peak_N_s','xy_peak_mm','so3_rmse_deg']}
    minima={k:float(full[k].min()) for k in ['expanded_mesh_gap_m','legacy_proxy_gap_m','remaining_shared_force_N']}
    passed=bool(len(d)==49 and len(full)==46 and full.success.all() and len(prefix)==3 and prefix.prefix_accepted.all() and not prefix.complete.any() and not changed and all(x['bitwise_equal'] for x in baseline) and independent['passed'] and viewer['passed'] and video['passed'] and peaks['max_force_N']<400 and peaks['additional_force_peak_N']<=5+1e-8 and peaks['additional_force_rate_peak_N_s']<=20+1e-6 and peaks['max_weld_translation_m']<=1e-4 and peaks['max_weld_angle_rad']<=1e-4 and peaks['max_jacobian_condition']<=50 and minima['expanded_mesh_gap_m']>=.005 and minima['legacy_proxy_gap_m']>=.005 and full.warning_count.eq(0).all())
    result=dict(passed=passed,history_files_checked=len(old),changed_history_files=changed,baseline_bitwise_checks=baseline,research_runs=49,completed_full_trajectories=46,accepted_incomplete_joint_prefixes=3,research_physical_steps=int(d.physical_steps.sum()),research_sim_seconds=float(d.physical_steps.sum()/1000),maxima=peaks,minima=minima,native_saturation_steps=int(full.native_saturation_steps.sum()),native_contact_observations=int(full.native_contact_observations.sum()),native_contacts_do_not_prove_collision_safety=True,geometry_scope=dict(query_pairs=887,connected_assembly_exclusions=103,passive_hinges_without_native_hard_stops=30,coverage='PARTIAL; mesh convex hulls and connected socket exclusions are not full mechanical safety certification'),independent_audit=independent['passed'],viewer=viewer,video=video,RL_training_executed=False,environment=dict(python=sys.version,executable=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),unexecuted=['PPO/SAC/TD3/BC training','tau10ms + load joint condition','negative-Z dynamics','full 6DoF large-motion dynamics','mass/gain/random-load distributions','hardware and mechanical-hardstop safety certification'])
    write_json(ROOT/'results/phase2c3b_safety_validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));assert passed

if __name__=='__main__':main()
