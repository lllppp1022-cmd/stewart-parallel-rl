"""Raw archival, phase separation, mapping, history and video acceptance; no tuning."""
import _bootstrap
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha,ids
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.control.observer_replay import replay_observer
from src.disturbance.metrics import vec,error_series

def validate(root):
    root=Path(root);frozen=json.loads((root/'final_protocol_frozen.json').read_text(encoding='utf-8'));params=json.loads((root/'controller_parameters_frozen.json').read_text(encoding='utf-8'));assert frozen['parameters_sha256']==sha(root/'controller_parameters_frozen.json');assert frozen['independent_tests_sha256']==sha(root/'independent_test_frozen.json')
    hist=json.loads((ROOT/'results/phase2b1_frozen_history.json').read_text(encoding='utf-8'));changed=[name for name,row in hist['files'].items() if sha(ROOT/name)!=row['sha256']];assert not changed
    nominal=create_plant();qadr,_=ids(nominal);ranges=StewartGeometry(nominal).stroke_ranges;audits=[];references={};initials=set();differences=[]
    for stage in ('test_safety_precheck','development','equivalence','final','implementation_repro'):
        for folder in sorted((root/stage).glob('*/*')):
            if not (folder/'summary.json').exists():continue
            cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));df=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');p,a,_=error_series(df);k=df.input_step.to_numpy(int);profile=cfg['disturbance'];expected=np.array(profile['wrench'])[None,:]*((k>=profile['start_step'])&(k<profile['stop_step']))[:,None]
            np.testing.assert_allclose(vec(df,'external_ctrl' if profile['frame']=='site-local' else 'external_world',6),expected,atol=1e-12)
            assert np.array_equal(k,np.arange(len(df)));np.testing.assert_allclose(df.time_s,(k+1)*.001,atol=1e-9);np.testing.assert_allclose(df.controller_measurement_time_s,k*.001,atol=1e-12)
            assert df.weld_translation_m.max()<1e-4 and df.weld_angle_rad.max()<1e-4 and df.proxy_clearance_m.min()>=.005 and df.warning_count.max()==0
            assert abs(vec(df,'force',6)).max()<=400 and abs(vec(df,'ctrl',12)[:,:6]).max()<=400
            with np.load(folder/'integration_states.npz') as states:
                assert len(states['states'])==len(df);np.testing.assert_array_equal(states['states'][:,1:44],vec(df,'qpos',43));np.testing.assert_array_equal(states['states'][:,44:86],vec(df,'qvel',42))
                margin=float(min(np.min(states['states'][:,1:44][:,qadr]-ranges[:,0]),np.min(ranges[:,1]-states['states'][:,1:44][:,qadr])));assert margin>=0
            initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'));initials.add(hashlib.sha256(np.asarray(initial['integration_state']).tobytes()).hexdigest());mapping=float(df.generalized_mapping_error.max());assert mapping<1e-10
            diagcomp=vec(df,'diag_compensation_N',6);rate=float(abs(np.diff(diagcomp,axis=0)/.001).max()) if len(df)>1 else 0
            assert abs(diagcomp).max()<=5+1e-10 and rate<=20+1e-8
            if stage=='final':
                assert cfg['final_protocol_sha256']==sha(root/'final_protocol_frozen.json');key=cfg['task'];reference=np.c_[vec(df,'p_ref',3),vec(df,'rpy_ref',3),vec(df,'q_ref',6)]
                if key in references:np.testing.assert_array_equal(reference,references[key])
                else:references[key]=reference
            audits.append(dict(stage=stage,task=cfg['task'],controller=cfg['controller'],folder=str(folder),physical_steps=len(df),force_peak_N=float(abs(vec(df,'force',6)).max()),minimum_stroke_margin_m=margin,minimum_partial_proxy_m=float(df.proxy_clearance_m.min()),maximum_weld_translation_m=float(df.weld_translation_m.max()),maximum_weld_angle_rad=float(df.weld_angle_rad.max()),maximum_mapping_error=mapping,maximum_compensation_N=float(abs(diagcomp).max()),maximum_compensation_rate_N_s=rate,position_log_error_mm=float(abs(p-df.position_error_mm).max()),so3_log_error_deg=float(abs(a-df.orientation_error_deg).max())))
    # Fifteen complete frozen baseline trajectories, including exact legacy timing.
    paths={'benchmark_force10':'formal/A_home_force','benchmark_rollp05':'formal/B_home_roll_positive','benchmark_rolln05':'formal/B_home_roll_negative','benchmark_z_roll_combo':'formal/C_z_roll_composite'}
    for task in [t for t in frozen['tasks'] if t['split']=='historical_benchmark']:
        for method in ('C0','C1','C2'):
            old=(ROOT/'results/runs/phase2a_20261028_v1'/paths[task['name']]/method) if task['name'] in paths else ROOT/'results/runs/phase2a_20261028_legacy_recheck_v2/legacy_recheck/legacy_local_z_roll_exact'/method
            new=root/'final'/task['name']/method
            with np.load(old/'integration_states.npz') as x,np.load(new/'integration_states.npz') as y:equal=bool(np.array_equal(x['states'],y['states']));delta=float(abs(x['states']-y['states']).max())
            differences.append(dict(task=task['name'],controller=method,bitwise_equal=equal,maximum_state_difference=delta));assert equal
    observer_recompute=[];model=json.loads((root/'nominal_task_model.json').read_text(encoding='utf-8'))
    for task in frozen['tasks']:
        folder=root/'final'/task['name']/'C3'
        if not folder.exists():continue
        estimate,df=replay_observer(folder,model,params['bandwidth_rad_s']);delta=float(abs(estimate-vec(df,'diag_equivalent_disturbance',6)).max());observer_recompute.append(dict(task=task['name'],maximum_offline_causal_estimate_difference=delta));assert delta<1e-8
    final=pd.read_csv(ROOT/'results/phase2b1_controller_comparison.csv');development=pd.read_csv(ROOT/'results/phase2b1_development_comparison.csv');assert len(final)==45 and len(development)==24
    assert len([x for x in audits if x['stage']=='equivalence'])==1 and len([x for x in audits if x['stage']=='test_safety_precheck'])==6
    gui=json.loads((ROOT/'results/phase2b1_observer_validation.json').read_text(encoding='utf-8'));assert gui['bitwise_equal'] and not gui['interrupted'] and gui['mirror_exitcode']==0
    equivalent=json.loads((root/'lambda_zero_equivalence.json').read_text(encoding='utf-8'));repro=json.loads((ROOT/'results/phase2b1_implementation_reproduction.json').read_text(encoding='utf-8'));assert equivalent['bitwise_equal'] and repro['bitwise_equal']
    video=json.loads((ROOT/'assets/videos/stewart_phase2b1_robust_control_comparison.validation.json').read_text(encoding='utf-8'));assert video['frames']==600 and video['all_frames_decoded'] and video['all_state_hashes_match']
    output=dict(root=str(root),research_trials=76,research_physical_steps=sum(x['physical_steps'] for x in audits if x['stage']!='implementation_repro'),implementation_reproduction_physical_steps=6000,gui_equivalence_physical_steps=4096,final_trials=len(final),final_successes=int(final.success.sum()),independent_final_trials=int((final.split=='independent_test').sum()),historical_benchmark_trials=int((final.split=='historical_benchmark').sum()),same_initial_state_count=len(initials),historical_files_checked=hist['count'],historical_files_changed=changed,baseline_full_state_equivalence=differences,lambda_zero_equivalence=equivalent,current_implementation_reproduction=repro,observer_causal_recomputation=observer_recompute,gui_nonintrusivity=gui,video=dict(path=video['output'],frames=video['frames'],duration_s=video['duration_s'],all_frames_decoded=True,all_state_hashes_match=True),frozen_parameters=params,raw_audits=audits,rl_training_executed=False,action_sensitivity_experiments_executed=False,decision='GO TO PHASE 2B-2',decision_scope='Only bounded action-sensitivity research if user approves; current classical controller already meets strict bands; no evidence that PPO is necessary')
    write_json(ROOT/'results/phase2b1_validation.json',output);pd.DataFrame(audits).to_csv(ROOT/'results/phase2b1_raw_validation.csv',index=False);print('VALIDATED',len(audits),'archived trials,',len(final),'final comparisons, baseline',len(differences),'all bitwise equal',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);a=p.parse_args();validate(a.run_root)
