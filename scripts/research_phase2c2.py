"""Bounded staged moving-task research. No automatic unbounded training."""
import _bootstrap
import argparse,json,time,copy,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from src.rl.phase2c2_env import MovingResidualEnv
from src.rl.stewart_residual_env import env_config
from src.disturbance.metrics import compute,vec

RUN=ROOT/'results/runs/phase2c2_20261030_v1'
PROTOCOL=dict(version='phase2c2_v1',development_seeds=[7401],training_seeds=[42,123],stage_decisions=[256,2048,4096],max_decisions_per_seed=4096,physical_dt_s=.001,initial_decision_hz=20,frequency_diagnostic_hz=[20,50],amplitude_candidates_m=[.002,.003],original_z_envelope_m=.005,action_scale_N=.5,
    main_metric='full-episode Z tracking RMSE (mm)',continuation_gate=dict(all_complete=True,no_new_contact_or_protection=True,each_seed_validation_Z_RMSE_ratio_max=1.10,each_seed_validation_XY_peak_ratio_max=1.10),
    expansion_gate=dict(mean_Z_RMSE_ratio_max=.90,all_two_seeds_better=True,no_safety_regression=True,force_energy_ratio_max=1.02,force_slew_ratio_max=1.05),
    diagnostics='6 amplitude ladder episodes; 24 action/frequency episodes; 6 matched classical baselines; no retries',final_test_locked=True,
    reference='6s: hold0-.5, quinticrise.5-2, hold2-3, quinticreturn3-4.5, hold4.5-6',budget_note='256 is prefix; both seeds proceed or stop together at2048; later8192/16384 requires a new frozen expansion protocol only if gate passes',
    train=dict(id='T_move',tau_s=.005,Fz_N=-5.5,on_step=1600,off_step=3500,amplitude_m=.003,shift_s=0.),
    validation=dict(id='V_move',tau_s=.0075,Fz_N=-5.25,on_step=1775,off_step=3775,amplitude_m=.0028,shift_s=.10),
    final=[dict(id='F_move1',tau_s=.006,Fz_N=-5.75,on_step=1525,off_step=3625,amplitude_m=.0026,shift_s=-.10),dict(id='F_move2',tau_s=.009,Fz_N=-5.0,on_step=1925,off_step=4025,amplitude_m=.0029,shift_s=.15)])

def cfg_for(task,baseline='four_d_lead',hz=20):
    cfg=env_config();cfg.update(version='phase2c2_v1',baseline=baseline,reference_cache=str(RUN/'references'),decision_physical_steps=1000//hz,training_tasks=[task],validation_tasks=[PROTOCOL['validation']],final_tasks=PROTOCOL['final'])
    # Same physically integrated R0 reward, normalisation and Gaussian interface.
    return cfg

def metrics(folder):
    d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');r=compute(folder,dict(dt=.001,recovery_position_mm=.25,recovery_angle_deg=.02,recovery_hold_s=.3));e=vec(d,'p',3)-vec(d,'p_ref',3)
    r.update(z_rmse_mm=float(np.sqrt(np.mean(e[:,2]**2))*1000),xy_peak_mm=float(np.linalg.norm(e[:,:2],axis=1).max()*1000),z_ptp_mm=float(np.ptp(d.p_2)*1000),reference_ptp_mm=float(np.ptp(d.p_ref_2)*1000),actual_rl_peak_leg_N=float(abs(vec(d,'rl_applied_command',6)).max()),mean_rl_abs_leg_N=float(abs(vec(d,'rl_applied_command',6)).mean()),cost_cap_fraction=float((d.reward_cap_factor<1).mean()))
    write_json(folder/'metrics.json',r);return r

def episode(folder,task,baseline='four_d_lead',pattern='zero',hz=20,record=False,model=None):
    env=MovingResidualEnv(folder,tasks=[task],config=cfg_for(task,baseline,hz),record_episodes=[0] if record else [],mode='Checkpoint Evaluation' if model else 'Baseline Comparison');obs,_=env.reset(seed=7401);done=False
    try:
        while not done:
            t=env.owner.data.time
            a=0. if pattern=='zero' else (1. if pattern=='positive' else -1. if pattern=='negative' else np.sin(2*np.pi*2*t)) if .8<=t<4.5 else 0.
            action=model.predict(obs,deterministic=True)[0] if model else np.array([a]);obs,r,te,tr,i=env.step(action);done=te or tr
    finally:env.close()
    result=metrics(Path(folder)/'episode_0000');result.update(task_id=task['id'],method=baseline,pattern=pattern,decision_hz=hz,task=task)
    write_json(Path(folder)/'metrics.json',result);print('EPISODE',Path(folder).name,'success',result['success'],'Z_RMSE',result['z_rmse_mm'],'motion',result['z_ptp_mm'],flush=True);return result

def init():
    RUN.mkdir(parents=True,exist_ok=False);write_json(RUN/'protocol_frozen.json',PROTOCOL)
    frozen={}
    for sub in ['src','configs','docs','tests','third_party']:
        for f in (ROOT/sub).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and 'phase2c2' not in f.name and f.name!='moving_reference.py':frozen[str(f.relative_to(ROOT))]=sha(f)
    for f in (ROOT/'results').glob('phase2c1*'):
        if f.is_file():frozen[str(f.relative_to(ROOT))]=sha(f)
    write_json(RUN/'historical_file_hashes.json',frozen);print('FROZEN',len(frozen),flush=True)

def ladder():
    rows=[]
    for amp in PROTOCOL['amplitude_candidates_m']:
        for name,tau,load in [('nominal',0.,0.),('lag',.005,0.),('lag_load',.005,-5.5)]:
            task=dict(PROTOCOL['train'],id=f'ladder_{amp}_{name}',amplitude_m=amp,tau_s=tau,Fz_N=load)
            r=episode(RUN/'ladder'/f'{amp}_{name}',task,record=name=='lag_load');rows.append(r);write_json(RUN/'amplitude_ladder.json',rows)
            if not r['success'] or r['warning_count'] or r['native_contact_observations']:write_json(RUN/'amplitude_gate.json',dict(passed=False,failure=r));return
    write_json(RUN/'amplitude_gate.json',dict(passed=True,selected_amplitude_m=.003,rows=len(rows)))

def diagnose():
    assert json.loads((RUN/'amplitude_gate.json').read_text(encoding="utf-8"))['passed'];rows=[]
    for tau in [.002,.005,.010]:
        task=dict(PROTOCOL['train'],id=f'probe_tau{tau}',tau_s=tau)
        for hz in [20,50]:
            for pattern in ['zero','positive','negative','sine']:
                r=episode(RUN/'action_diagnostics'/f'tau{tau}_{hz}_{pattern}',task,pattern=pattern,hz=hz);rows.append(r);write_json(RUN/'action_diagnostics.json',rows)
    for task in [PROTOCOL['train'],PROTOCOL['validation']]:
        for method in ['c4','fz_lead','four_d_lead']:
            episode(RUN/'baselines'/task['id']/method,task,baseline=method)
    safe=all(r['success'] and not r['native_contact_observations'] and not r['warning_count'] for r in rows);effective=min(r['actual_rl_peak_leg_N'] for r in rows if r['pattern']!='zero')>1e-5
    write_json(RUN/'training_gate.json',dict(passed=bool(safe and effective),safe=safe,action_effective=effective,decision_hz=20,frequency_reason='50Hz is responsiveness diagnostic only; no demonstrated learning advantage, retain20Hz to avoid confounding sample budget',baseline='four_d_lead',scope='Single-task learnability pilot, not generalisation claim'))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','ladder','diagnose']);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.stage=='init':init()
    elif a.stage=='ladder':ladder()
    elif a.stage=='diagnose':diagnose()
if __name__=='__main__':main()
