"""Independent nominal-model / equivalent-wrench validation, not controller training."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json
from src.control.nominal_task_model import build_model
from src.control.observer_replay import replay_observer
from src.disturbance.metrics import vec

def nominal_validation(root):
    model=json.loads((Path(root)/'nominal_task_model.json').read_text(encoding='utf-8'));out=[]
    for name in ('home','z','roll','pitch','z_roll','roll_pitch'):
        folder=ROOT/'results/runs/phase2a_20261028_v1/formal'/('zero_'+name)/'C0'
        if not folder.exists():continue
        est,df=replay_observer(folder,model,15.);vel=vec(df,'twist_actual',6);before=vec(df,'replay_measured_pre_velocity',6);drive=vec(df,'replay_nominal_drive',6);residual=(vel-before)@np.asarray(model['mass']).T/.001-drive;mask=df.time_s.to_numpy()>.5
        out.append(dict(task=name,source=str(folder),real_physics_steps_reintegrated=0,local_force_residual_rms_N=float(np.sqrt(np.mean(residual[mask,2]**2))),local_roll_residual_rms_Nm=float(np.sqrt(np.mean(residual[mask,3]**2))),local_pitch_residual_rms_Nm=float(np.sqrt(np.mean(residual[mask,4]**2))),filtered_zero_force_rms_N=float(np.sqrt(np.mean(est[mask,2]**2))),filtered_zero_roll_rms_Nm=float(np.sqrt(np.mean(est[mask,3]**2))),filtered_zero_pitch_rms_Nm=float(np.sqrt(np.mean(est[mask,4]**2))),maximum_position_shift_m=float(abs(vec(df,'p',3)-np.asarray(model['home_position'])).max()),maximum_roll_pitch_deg=float(abs(vec(df,'rpy',3)[:,:2]).max())))
    write_json(ROOT/'results/phase2b1_nominal_model_validation.json',dict(model=model,trials=out,coefficient_fitting_performed=False,validation_only=True,inputs='recorded pose/world velocity/known prior leg commands; no actual M/bias/qacc/constraint force used',scope='Home-local six-axis linearization; compliant closure, changing inertia and nonlinear velocities remain equivalent residuals'))
    pd.DataFrame(out).to_csv(ROOT/'results/phase2b1_nominal_model_validation.csv',index=False);print(pd.DataFrame(out).to_string(index=False),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();nominal_validation(a.run_root)
