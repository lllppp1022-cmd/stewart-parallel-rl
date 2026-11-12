"""Read-only full-rate historical phase decomposition and nominal position audit."""
import _bootstrap,json
from pathlib import Path
import numpy as np,pandas as pd,mujoco,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json,sha
from src.workspace.feedforward_study import RUN
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.audit_control import scratch_reference_data
from src.control.nominal_task_model import tangent

def phases(times,T):
    t=np.asarray(times);return np.select([t<1,t<1+T/2,t<1+T,t<2+T,t<2+1.5*T,t<2+2*T],['home_initial','rise_accel','rise_decel','top_hold','fall_accel','fall_decel'],default='home_final')

def main():
    if RUN.exists():raise FileExistsError(RUN)
    RUN.mkdir(parents=True);old={}
    for directory in ['src','scripts','configs','docs','models','third_party','tests']:
        for f in (ROOT/directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and 'phase2c3b' not in str(f).lower() and f.name not in ['dynamic_reference_feedforward.py','large_motion_feedforward_wrapper.py','feedforward_study.py']:old[str(f.relative_to(ROOT))]=sha(f)
    for mf in ['phase2c1_artifact_manifest.json','phase2c2_artifact_manifest.json','phase2c3a_artifact_manifest.json']:
        for k,v in json.loads((ROOT/'results'/mf).read_text(encoding='utf-8'))['files'].items():old[k.replace('/','\\')]=v['sha256']
    write_json(RUN/'history_freeze.json',old);rows=[];joint=[]
    for method in ['C0','C4','fourDlead']:
        for T in [10,6,4]:
            folder=ROOT/f'results/runs/phase2c3a_20261103_v1/dynamics/{method}_L50_T{T}_tau0_load0';d=pd.read_csv(folder/'physics.csv.gz');error=(d.p_2-d.p_ref_2).to_numpy();v=d.twist_ref_2.to_numpy();a=d.reference_z_acceleration_m_s2.to_numpy();group=phases(d.time_s,T)
            for name in np.unique(group):
                s=group==name;X=np.c_[v[s],a[s],np.ones(s.sum())];coef=np.linalg.lstsq(X,error[s],rcond=None)[0];prediction=X@coef
                rows.append(dict(method=method,rise_s=T,phase=name,samples=int(s.sum()),z_rmse_mm=float(np.sqrt(np.mean(error[s]**2))*1000),z_mean_mm=float(error[s].mean()*1000),velocity_error_correlation=float(np.corrcoef(v[s],error[s])[0,1]) if np.std(v[s])>1e-12 and np.std(error[s])>1e-12 else None,acceleration_error_correlation=float(np.corrcoef(a[s],error[s])[0,1]) if np.std(a[s])>1e-12 and np.std(error[s])>1e-12 else None,velocity_coefficient_s=float(coef[0]),acceleration_coefficient_s2=float(coef[1]),regression_residual_rmse_mm=float(np.sqrt(np.mean((prediction-error[s])**2))*1000),observer_mean_Fz_N=float(d.diag_equivalent_disturbance_2[s].mean()),force_peak_N=float(abs(d[[f'force_{i}' for i in range(6)]][s]).to_numpy().max())))
            if method=='C4':
                fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True);ax[0].plot(d.time_s,error*1000);ax[0].set_ylabel('Z error (mm)');ax[1].plot(d.time_s,v*1000);ax[1].set_ylabel('Reference speed (mm/s)');ax[2].plot(d.time_s,a*1000);ax[2].set_ylabel('Reference acc (mm/s2)');ax[2].set_xlabel('Physical time (s)')
                for x in ax:x.grid(alpha=.2)
                fig.suptitle(f'Historical REAL C4 L50 T{T}: correlations are not causal proof');fig.tight_layout();fig.savefig(ROOT/f'assets/plots/phase2c3b_historical_T{T}.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c3b_error_phase_decomposition.csv',index=False)
    m=create_plant();g=StewartGeometry(m);seed=None
    for z in np.linspace(0,.05,26):
        p=g.home_position+[0,0,z];s=inverse_kinematics(g,p,g.home_rotation,seed);seed=s.chain_q;d=scratch_reference_data(m,g,s,p,g.home_rotation);T,_,_=tangent(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);joint.append(dict(z_m=z,effective_mass_kg=float((T.T@M@T)[2,2]),effective_damping_N_s_m=float((T.T@np.diag(m.dof_damping)@T)[2,2])))
    pd.DataFrame(joint).to_csv(ROOT/'results/phase2c3b_nominal_position_audit.csv',index=False)
    source=ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json';nom=json.loads(source.read_text(encoding='utf-8'));parameters=dict(kv_N_s_m=float(np.array(nom['damping'])[2,2]),ka_kg=float(np.array(nom['mass'])[2,2]),candidate_count=1,candidate_scales=[1.0],source=str(source),source_sha256=sha(source),no_gravity_term=True,selection='Nominal Home coefficients, frozen before development; no final-test fitting',scope='Local Z equivalent dynamics, not full inverse dynamics')
    write_json(ROOT/'results/phase2c3b_feedforward_parameters.json',parameters);print(parameters,flush=True)
if __name__=='__main__':main()
