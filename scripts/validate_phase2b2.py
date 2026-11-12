"""Independent full-rate archive validation; never runs a controller or integration."""
import _bootstrap
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd, mujoco
from src.model import ROOT, write_json, sha, load_model
from src.control.phase2b2_experiment import actual_force_map
from src.disturbance.metrics import vec, error_series

def validate(root):
    root=Path(root); all_rows=pd.read_csv(ROOT/'results/phase2b2_all_trials.csv'); rows=[]; maximum={}; nominal=load_model(); scratch=mujoco.MjData(nominal)
    frozen=json.loads((root/'actions_frozen.json').read_text(encoding='utf-8'))['selected']
    for ix,r in all_rows.iterrows():
        folder=Path(r.folder); cfg=json.loads((folder/'config.json').read_text(encoding='utf-8')); plant=json.loads((folder/'plant.json').read_text(encoding='utf-8'))
        prefixes=['qpos','qvel','f_cmd','force','F_C0','F_C3','F_residual_requested','F_residual_applied','F_add_total','W_physical_residual','W_physical_add','true_kinematic_map','sensor_noise','p','p_ref','rpy','rpy_ref']
        scalars={'time_s','step','input_step','failure','budget_applicable','sensor_read_step','sensor_capture_step','load_active','controller_measurement_time_s'}
        d=pd.read_csv(folder/'physics.csv.gz',usecols=lambda k:k in scalars or any(k.startswith(p+'_') and k[len(p)+1:].isdigit() for p in prefixes),float_precision='round_trip')
        with np.load(folder/'integration_states.npz') as z:
            states=z['states']; assert len(states)==len(d) and np.isfinite(states).all()
            np.testing.assert_array_equal(states[:,0],d.time_s)
            np.testing.assert_array_equal(states[:,1:44],vec(d,'qpos',43));np.testing.assert_array_equal(states[:,44:86],vec(d,'qvel',42))
            # Independently re-evaluate several pre-step kinematic maps on scratch data.
            initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'))
            map_error=0.
            for i in np.linspace(0,len(d)-1,5,dtype=int):
                scratch.qpos[:]=initial['qpos'] if i==0 else states[i-1,1:44]
                scratch.qvel[:]=initial['qvel'] if i==0 else states[i-1,44:86]
                mujoco.mj_forward(nominal,scratch)
                map_error=max(map_error,float(abs(actual_force_map(nominal,scratch)-vec(d.iloc[[i]],'true_kinematic_map',36).reshape(6,6)).max()))
        assert map_error<2e-8, (folder,map_error)
        active=d.budget_applicable.to_numpy(bool);cmd=vec(d,'f_cmd',6);force=vec(d,'force',6);c0=vec(d,'F_C0',6);add=vec(d,'F_add_total',6)
        if active.any():
            np.testing.assert_allclose(cmd[active]-c0[active],add[active],rtol=0,atol=1e-12)
            assert abs(add[active]).max()<=5+1e-9
            assert abs(np.diff(np.r_[np.zeros((1,6)),add[active]],axis=0)).max()<=.02+1e-9
        assert abs(cmd).max()<=400+1e-10 and abs(force).max()<=400+1e-10
        a=vec(d,'true_kinematic_map',36).reshape(-1,6,6);g=np.asarray(plant['gain'])
        fb=cmd-vec(d,'F_residual_applied',6)
        for field,base in [('W_physical_residual',fb),('W_physical_add',c0)]:
            expected=np.einsum('nij,ni->nj',a,force-np.clip(g*base,-400,400))
            np.testing.assert_allclose(expected,vec(d,field,6),atol=1e-11,rtol=1e-10)
        n=d.sensor_read_step.to_numpy();capture=d.sensor_capture_step.to_numpy();valid=n>=0
        assert np.all(capture[valid]<=n[valid]);delay=cfg.get('measurement',{}).get('delay_steps',0)
        np.testing.assert_array_equal(capture[active],np.maximum(0,n[active]-delay))
        pr=cfg['disturbance'];expected_load=(d.input_step>=pr['start_step'])&(d.input_step<pr['stop_step'])
        np.testing.assert_array_equal(d.load_active,expected_load & (np.linalg.norm(pr['wrench'])>0))
        p,angle,_=error_series(d);assert abs(np.sqrt(np.mean(p*p))-r.position_rmse_mm)<1e-10
        # Model perturbations may only affect their explicitly named parameter group.
        before=plant['nominal_plant'];pert=cfg.get('actual_perturbation',{})
        for key in ['ctrlrange','forcerange','eq_solref','eq_solimp','timestep_s']:
            assert plant[key]==before[key],(folder,key)
        assert plant['platform_mass_kg']==before['platform_mass_kg']*pert.get('mass_scale',1)
        np.testing.assert_allclose(plant['platform_inertia_kg_m2'],np.asarray(before['platform_inertia_kg_m2'])*pert.get('mass_scale',1),rtol=0,atol=1e-14)
        np.testing.assert_allclose(plant['gain'],np.asarray(before['gain'])*pert.get('gain_scales',[1]*6),rtol=0,atol=0)
        if cfg['stage']=='frozen_evaluation' and cfg['controller']=='C3' and cfg['action']['kind']!='zero':
            assert cfg['action']['kind']==frozen['kind'] and cfg['action']['frequency']==frozen['frequency']
        rows.append(dict(folder=str(folder),steps=len(d),map_max_error=map_error,finite=True,state_csv_exact=True,shared_budget_checked=bool(active.any()),mapping_checked=True,partial_sensor_checked=True))
        if (ix+1)%10==0:print('VALIDATED',ix+1,flush=True)
    history=json.loads((ROOT/'results/phase2b2_frozen_history.json').read_text(encoding='utf-8'))
    for name,entry in history['files'].items():assert sha(ROOT/name)==entry['sha256'],name
    assert len(rows)==90 and all_rows.success.sum()==89
    bad=all_rows[~all_rows.success].iloc[0];assert bad.variant=='asymmetric_0.01_C3' and not bad.load_started and not bad.unload_reached
    assert bad.physical_steps==584 and abs(bad.first_failure_time_s-.384)<1e-10
    eq=pd.read_csv(ROOT/'results/phase2b2_zero_residual_equivalence.csv');assert eq.full_integration_state_bitwise_equal.all() and eq.all_physical_signals_equal.all()
    gui=json.loads((ROOT/'results/phase2b2_observer_validation.json').read_text(encoding='utf-8'));assert gui['bitwise_equal'] and gui['physics_steps_each']==2048 and not gui['interrupted']
    video=json.loads((ROOT/'assets/videos/stewart_phase2b2_residual_action_validation.validation.json').read_text(encoding='utf-8'));assert video['all_state_hashes_match'] and video['all_frames_decoded'] and video['no_future_curves']
    result=dict(research_trials=len(rows),successful_trials=int(all_rows.success.sum()),failed_trials=1,research_physical_steps=int(sum(x['steps'] for x in rows)),history_files_unchanged=len(history['files']),zero_residual_bitwise=True,all_raw_states_finite=True,full_rate_budget_and_native_limits_passed=True,actual_generalized_force_recomputed=True,kinematic_map_independent_scratch_max_error=max(x['map_max_error'] for x in rows),failed_trial=dict(variant=bad.variant,time_s=bad.first_failure_time_s,unloaded_censored=True,reason=bad.failure_reason,hold_steps=200),observer_gui=gui,video_frames=video['frames'],video_unique_frames=video['unique_decoded_frames'],rl_training=False,trials=rows)
    write_json(ROOT/'results/phase2b2_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('trials','observer_gui')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);validate(p.parse_args().run_root)
