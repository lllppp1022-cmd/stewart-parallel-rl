"""Full-rate archival, coordinate, historical and constraint acceptance from raw data."""
import _bootstrap
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,write_json,sha,ids
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.disturbance.metrics import vec,error_series

def validate():
    root=ROOT/'results/runs/phase2a_20261028_v1';extra=ROOT/'results/runs/phase2a_20261028_legacy_recheck_v2';m=create_plant();d=mujoco.MjData(m);site=m.site('upper_center').id;body=int(m.site_bodyid[site]);qadr,_=ids(m);ranges=StewartGeometry(m).stroke_ranges
    folders=list((root/'pilot').glob('*/*'))+list((root/'formal').glob('*/*'))+list((extra/'legacy_recheck').glob('*/*'));audits=[];recomputations=[];initial_hashes=set()
    for folder in sorted(folders):
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));profile=json.loads((folder/'disturbance_profile.json').read_text(encoding='utf-8'));initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'));df=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');p,a,_=error_series(df)
        with np.load(folder/'integration_states.npz') as z:
            states=z['states'];assert len(states)==len(df)==6000
            assert np.max(abs(states[:,1:44]-vec(df,'qpos',43)))<1e-14
            assert np.max(abs(states[:,44:86]-vec(df,'qvel',42)))<1e-14
            np.testing.assert_allclose(z['time'],df.time_s,atol=1e-12)
            k=df.input_step.to_numpy(int);assert np.array_equal(k,np.arange(6000));np.testing.assert_allclose(df.time_s,(k+1)*.001,atol=1e-9)
            local=vec(df,'external_ctrl',6);world=vec(df,'external_world',6);rot=vec(df,'external_rotation',9).reshape(-1,3,3);want=np.asarray(profile['wrench']);mask=(k>=profile['start_step'])&(k<profile['stop_step'])
            if profile['kind']=='zero':mask[:]=False
            expected=np.where(mask[:,None],want,0.);np.testing.assert_allclose(local if profile['frame']=='site-local' else world,expected,atol=1e-12)
            converted=np.c_[np.einsum('nij,nj->ni',rot,local[:,:3]),np.einsum('nij,nj->ni',rot,local[:,3:])];np.testing.assert_allclose(converted,world,atol=1e-12);np.testing.assert_array_equal(vec(df,'ctrl',12)[:,6:],local)
            assert np.max(abs(vec(df,'force',6)))<=400 and df.warning_count.max()==0
            assert df.weld_translation_m.max()<=1e-4 and df.weld_angle_rad.max()<=1e-4 and df.proxy_clearance_m.min()>=.005
            q=states[:,1:44][:,qadr];margin=float(min(np.min(q-ranges[:,0]),np.min(ranges[:,1]-q)));assert margin>=0
            maximum=0.
            for i in [profile['start_step'],min(profile['start_step']+333,5999),profile['stop_step']-1]:
                mujoco.mj_setState(m,d,states[i-1] if i else np.array(initial['integration_state']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);np.testing.assert_allclose(d.site_xmat[site].reshape(3,3),rot[i],atol=1e-12)
                force=np.zeros(m.nv);mujoco.mj_applyFT(m,d,world[i,:3],world[i,3:],d.site_xpos[site],body,force);delta=float(np.max(abs(force-vec(df.iloc[[i]],'external_generalized',m.nv)[0])));maximum=max(maximum,delta)
            assert maximum<1e-10
            active=np.flatnonzero(np.any(local!=0,axis=1));events=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            if profile['kind']!='zero':
                on=[e for e in events if e['type']=='load_on'][0];off=[e for e in events if e['type']=='load_off'][0];assert on['step']==profile['start_step'] and off['step']==profile['stop_step']
            else:on=None
            initial_hashes.add(hashlib.sha256(np.array(initial['integration_state']).tobytes()).hexdigest());audits.append(dict(folder=str(folder),stage=cfg['stage'],task=cfg['task'],controller=cfg['controller'],steps=len(df),initial_hash=hashlib.sha256(np.array(initial['integration_state']).tobytes()).hexdigest(),first_nonzero_result_step=int(active[0]+1) if len(active) else None,last_nonzero_result_step=int(active[-1]+1) if len(active) else None,maximum_mapping_error_independent=maximum,minimum_stroke_margin_m=margin,reference_moving_on_load=on.get('reference_is_moving') if on else None,position_log_max_error_mm=float(np.max(abs(p-df.position_error_mm))),so3_log_max_error_deg=float(np.max(abs(a-df.orientation_error_deg))),native_contacts=int(df.native_contacts.sum())))
            recomputations.append(dict(folder=str(folder),position_rmse_mm=float(np.sqrt(np.mean(p*p))),so3_rmse_deg=float(np.sqrt(np.mean(a*a))),position_log_max_error_mm=float(np.max(abs(p-df.position_error_mm))),so3_log_max_error_deg=float(np.max(abs(a-df.orientation_error_deg)))))
            plant=json.loads((folder/'plant.json').read_text(encoding='utf-8'))
            if 'legacy_base_external_channel_policy' not in plant:plant['legacy_base_external_channel_policy']=plant.pop('external_channel_policy')
            plant['external_channel_policy']='Phase2A explicit independent profile; authoritative disturbance_profile.json, input_step intervals and physical external_ctrl/world logs; old base policy retained only for audit'
            write_json(folder/'plant.json',plant)
    legacy=[];old=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');old=old[(old.condition=='external_load')&(old.timestep_s==.001)]
    for row in old.itertuples():
        original=pd.read_csv(ROOT/row.physics_log,float_precision='round_trip');new=pd.read_csv(extra/'legacy_recheck/legacy_local_z_roll_exact'/row.controller/'physics.csv.gz',float_precision='round_trip');oldq=original[[f'qpos_{i}' for i in range(1,44)]].to_numpy();newq=vec(new,'qpos',43);oldctrl=original[[f'ctrl_leg_N_{i}' for i in range(1,7)]].to_numpy();newctrl=vec(new,'ctrl',12)[:,:6];oldext=original[[f'external_ctrl_{i}' for i in range(1,7)]].to_numpy();newext=vec(new,'external_ctrl',6)
        legacy.append(dict(controller=row.controller,old_source=str(ROOT/row.physics_log),steps=len(new),external_commands_bitwise_equal=bool(np.array_equal(oldext,newext)),qpos_bitwise_equal=bool(np.array_equal(oldq,newq)),maximum_qpos_difference=float(np.max(abs(oldq-newq))),maximum_leg_ctrl_difference_N=float(np.max(abs(oldctrl-newctrl)))))
        assert legacy[-1]['external_commands_bitwise_equal'] and legacy[-1]['maximum_qpos_difference']<1e-12
    frozen=json.loads((ROOT/'results/phase2a_frozen_history.json').read_text(encoding='utf-8'));changed=[name for name,value in frozen['files'].items() if sha(ROOT/name)!=value['sha256']];assert not changed
    observer=json.loads((ROOT/'results/phase2a_observer_validation.json').read_text(encoding='utf-8'));video=json.loads((ROOT/'assets/videos/stewart_phase2a_disturbance_rejection.validation.json').read_text(encoding='utf-8'));formal=pd.read_csv(ROOT/'results/phase2a_controller_comparison.csv');assert len(formal)==33
    write_json(ROOT/'results/phase2a_validation.json',dict(root=str(root),legacy_recheck_root=str(extra),physical_trial_count=len(folders),trial_physical_steps=sum(x['steps'] for x in audits),formal_trials=len(formal),formal_successes=int(formal.success.sum()),pilot_trials=sum(x['stage']=='pilot' for x in audits),legacy_offset_diagnostic_trials=3,identical_initial_state_count=len(initial_hashes),historical_files_checked=frozen['count'],historical_files_changed=changed,maximum_independent_mapping_error=max(x['maximum_mapping_error_independent'] for x in audits),legacy_reproduction=legacy,observer=observer,video=dict(path=video['output'],frames=video['frames'],duration_s=video['duration_s'],all_frames_decoded=video['all_frames_decoded'],all_state_hashes_match=video['all_state_hashes_match']),raw_audits=audits,decision='GO TO PHASE 2B',rl_training_executed=False))
    pd.DataFrame(recomputations).to_csv(ROOT/'results/phase2a_independent_metric_checks.csv',index=False);print('VALIDATED',len(folders),'trials; legacy',legacy,flush=True)
if __name__=='__main__':validate()
