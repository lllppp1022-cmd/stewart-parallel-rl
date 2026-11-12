"""Offline linear error prediction to test a mechanism; does not tune a controller."""
import _bootstrap,json
import numpy as np,pandas as pd
from src.model import ROOT,write_json
from src.phase1.controllers import Controller
from src.phase1.plant import create_plant
from analyze_phase2c3a_workspace import RUN

def main():
    nominal=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'));ctrl=Controller(create_plant(),'C0',.001);K=ctrl.k_task+np.array(nominal['stiffness']);D=np.array(nominal['damping']);M=np.array(nominal['mass']);rows=[]
    for folder in sorted((RUN/'dynamics').iterdir()):
        if not (folder/'trial_result.json').exists():continue
        result=json.loads((folder/'trial_result.json').read_text(encoding='utf-8'))
        if result['external_Fz_N'] or result['tau_s']:continue
        d=pd.read_csv(folder/'physics.csv.gz');v=np.zeros((len(d),6));a=v.copy();v[:,2]=d.twist_ref_2;a[:,2]=d.reference_z_acceleration_m_s2;prediction=np.linalg.solve(K,(-v@D.T-a@M.T).T).T[:,2];error=(d.p_2-d.p_ref_2).to_numpy();mask=abs(v[:,2])>1e-5
        rows.append(dict(trial=folder.name,method=result['method'],correlation_during_motion=float(np.corrcoef(prediction[mask],error[mask])[0,1]),actual_z_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),quasistatic_prediction_rmse_mm=float(np.sqrt(np.mean(prediction**2))*1000),difference_rmse_mm=float(np.sqrt(np.mean((error-prediction)**2))*1000),assumption='Home-linear quasi-static error from omitted reference damping/inertia feedforward; not a proof or changed controller'))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c3a_tracking_mechanism.csv',index=False)
    write_json(RUN/'tracking_mechanism_model.json',dict(Mzz=float(M[2,2]),Dzz=float(D[2,2]),effective_Kzz=float(K[2,2]),equation='M e_ddot+(D+Kd)e_dot+(Kp+Kpassive)e = -M reference_acceleration-D reference_velocity; quasi-static approximation only'))
if __name__=='__main__':main()
