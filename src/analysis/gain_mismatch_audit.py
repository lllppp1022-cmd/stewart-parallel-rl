"""Offline audit only: hidden gain and actual tangent never enter causal control."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from ..model import ROOT,write_json
from ..disturbance.metrics import vec

def audit():
    base=ROOT/'results/runs/phase2b2_20261029_v1';model=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'));M=np.array(model['mass']);K=np.array(model['stiffness']);D=np.array(model['damping']);w0=np.array(model['equilibrium_wrench']);rh=np.array(model['home_rotation']);rows=[];summaries=[]
    folders={'nominal':base/'zero_equivalence/force10/C3','alternating_0.005':base/'robustness/robust_force10/asymmetric_0.005_C3','historical_failed_0.01':base/'robustness/robust_force10/asymmetric_0.01_C3'}
    for name,folder in folders.items():
        d=pd.read_csv(folder/'physics.csv.gz');a=vec(d,'true_kinematic_map',36).reshape(-1,6,6);f=vec(d,'force',6);cmd=vec(d,'f_cmd',6);gain=json.loads((folder/'plant.json').read_text(encoding='utf-8'))['gain']
        np.testing.assert_allclose(f,np.clip(cmd*np.array(gain),-400,400),atol=1e-10)
        actual=np.einsum('nij,ni->nj',a,f);known=np.einsum('nij,ni->nj',a,cmd);delta=actual-known
        velocity=vec(d,'twist_actual',6);prior=np.r_[np.zeros((1,6)),velocity[:-1]];acc=(velocity-prior)/.001
        pos=vec(d,'p',3)-np.array(model['home_position']);rot=Rotation.from_euler('xyz',vec(d,'rpy',3),degrees=True).as_matrix();x=np.c_[pos,Rotation.from_matrix(rot@rh.T).as_rotvec()];xp=np.r_[np.zeros((1,6)),x[:-1]]
        local_required=acc@M.T-(known-w0-xp@K.T-prior@D.T)-vec(d,'external_world',6)
        frame=pd.DataFrame(dict(case=name,time_s=d.time_s,input_step=d.input_step,yaw_deg=d.rpy_2,yaw_rate_rad_s=velocity[:,5],yaw_acceleration_rad_s2=acc[:,5],observer_Mz_Nm=d.diag_equivalent_disturbance_5,command_Mz_Nm=known[:,5],actual_Mz_Nm=actual[:,5],gain_error_Mz_Nm=delta[:,5],local_required_Mz_Nm=local_required[:,5],unexplained_local_Mz_Nm=local_required[:,5]-delta[:,5],failure=d.failure.fillna(''),x_m=d.p_0,y_m=d.p_1,roll_deg=d.rpy_0,pitch_deg=d.rpy_1))
        for j in range(6):
            frame[f'gain_{j}']=gain[j];frame[f'force_command_{j}']=cmd[:,j];frame[f'force_actual_{j}']=f[:,j];frame[f'force_difference_{j}']=f[:,j]-cmd[:,j];frame[f'wrench_error_{j}']=delta[:,j];frame[f'yaw_lever_arm_{j}']=a[:,j,5];frame[f'yaw_moment_contribution_{j}']=a[:,j,5]*(f[:,j]-cmd[:,j])
        rows.append(frame);s=(d.time_s>.15)&(d.time_s<min(1.8,float(d.time_s.iloc[-1])))&d.failure.isna()
        summaries.append(dict(case=name,preload_gain_Mz_mean_Nm=float(delta[s,5].mean()),preload_observer_Mz_mean_Nm=float(d.diag_equivalent_disturbance_5[s].mean()),local_unexplained_Mz_RMS_Nm=float(np.sqrt(np.mean((local_required[s,5]-delta[s,5])**2))),preload_yaw_rate_peak_rad_s=float(abs(velocity[s,5]).max()),failed=bool(d.failure.notna().any()),first_failure_s=float(d.time_s[d.failure.notna()].iloc[0]) if d.failure.notna().any() else None))
    out=pd.concat(rows,ignore_index=True);out.to_csv(ROOT/'results/phase2b3_gain_yaw_mapping.csv',index=False);write_json(ROOT/'results/phase2b3_historical_audit.json',dict(cases=summaries,source='Read-only Phase2B2 logs; no known-failed replay integration',local_model_limits='M/D/K constant home approximation; constraint compliance and variable mapping remain residual',privileged_offline_only=True))
    return out
