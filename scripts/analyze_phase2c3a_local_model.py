"""Independent nominal STATIC/tangent audit; not dynamic execution or controller tuning."""
import _bootstrap,json
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,write_json
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.audit_control import scratch_reference_data
from src.control.nominal_task_model import tangent,pose_coordinates
from analyze_phase2c3a_workspace import RUN

def main():
    nominal=json.loads((ROOT/'results/runs/phase2b1_20261028_v1/nominal_task_model.json').read_text(encoding='utf-8'));m=create_plant();g=StewartGeometry(m);rows=[];seed=None
    for z in np.linspace(0,.05,26):
        p=g.home_position+[0,0,z];s=inverse_kinematics(g,p,g.home_rotation,seed);seed=s.chain_q;d=scratch_reference_data(m,g,s,p,g.home_rotation);T,_,_=tangent(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);reduced=T.T@M@T;actual=T.T@(d.qfrc_bias-d.qfrc_passive);linear=np.array(nominal['equilibrium_wrench'])+np.array(nominal['stiffness'])@pose_coordinates(nominal,p,g.home_rotation)
        rows.append(dict(delta_z_m=z,nominal_reduced_Mzz_kg=float(reduced[2,2]),home_Mzz_kg=float(np.array(nominal['mass'])[2,2]),static_z_model_error_N=float(actual[2]-linear[2]),static_wrench_model_error_norm=float(np.linalg.norm(actual-linear)),label='NOMINAL OFFLINE STATIC MODEL AUDIT; not measured disturbance'))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c3a_local_model_audit.csv',index=False)
if __name__=='__main__':main()
