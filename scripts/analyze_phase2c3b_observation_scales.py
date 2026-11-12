"""Read-only range audit of legal signals; does not create a new RL env."""
import _bootstrap,json
import pandas as pd,numpy as np
from src.model import ROOT,write_json
from src.workspace.feedforward_study import RUN

def main():
    d=pd.read_csv(RUN/'dev/FF-VA_L50_T6_tau0_load0_none/physics.csv.gz');old=json.loads((ROOT/'configs/phase2c1_env.yaml').read_text(encoding='utf-8'))['observation_scales'];rows=[]
    signals=[('Z_error_m',(d.p_2-d.p_ref_2).to_numpy(),old['position_m'][2],1e-5),('XY_error_m',d[['p_0','p_1']].to_numpy()-d[['p_ref_0','p_ref_1']].to_numpy(),old['position_m'][0],5e-5),('actual_Z_from_home_m',d.actual_z_from_home_m.to_numpy(),None,.05),('reference_Z_from_home_m',d.reference_z_from_home_m.to_numpy(),.005,.05),('relative_slide_m',d[[f'q_{i}' for i in range(6)]].to_numpy()-d[[f'q_{i}' for i in range(6)]].iloc[0].to_numpy(),old['slide_m'],.06),('reference_Z_velocity_m_s',d.twist_ref_2.to_numpy(),old['linear_velocity_m_s'],.03),('reference_Z_acceleration_m_s2',d.reference_z_acceleration_m_s2.to_numpy(),None,.03),('total_add_command_N',d[[f'ff_actual_total_add_N_{i}' for i in range(6)]].to_numpy(),None,5)]
    for name,a,b,c in signals:
        rows.append(dict(signal=name,absolute_peak=float(abs(a).max()),p99_absolute=float(np.quantile(abs(a),.99)),old_scale=b,old_peak_normalised=float(abs(a).max()/b) if b else None,old_fraction_over_clip10=float(np.mean(abs(a)/b>10)) if b else None,candidate_scale=c,candidate_peak_normalised=float(abs(a).max()/c),candidate_fraction_over_clip10=float(np.mean(abs(a)/c>10))))
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c3b_observation_scale_audit.csv',index=False);write_json(RUN/'observation_scale_audit.json',dict(source='Development FF-VA L50T6 only; signal ranges, not full85-feature or future53-feature execution',no_rl_env_modified=True,scales_are_design_candidates=True,rows=rows));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
