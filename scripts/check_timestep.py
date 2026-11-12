"""Bounded numerical sensitivity audit; preserves the original 2ms baseline."""
import _bootstrap
import json,time
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,load_model,write_json,sha
from src.audit_control import AuditController,advance
config=json.loads((ROOT/'configs/phase0_protocol.json').read_text(encoding='utf-8'))
protocol=dict(timesteps_s=[.001,.0005],scenarios=['home','z_up','z_down','roll_pos','roll_neg','pitch_pos','pitch_neg'],duration_s=5,thresholds_unchanged=True,source_mjcf_unchanged=True,reason='Original 2ms baseline exceeded frozen angular gap criterion; audit solver/timestep sensitivity without relaxing it',maximum_additional_physics_steps=105000)
folder=ROOT/'results/runs'/f'timestep_audit_{time.time_ns()}';folder.mkdir(parents=True);write_json(folder/'frozen_protocol.json',protocol);rows=[]
for dt in protocol['timesteps_s']:
 for label in protocol['scenarios']:
  m=load_model();m.opt.timestep=dt;axis='z' if label.startswith('z_') else 'roll' if label.startswith('roll_') else 'pitch' if label.startswith('pitch_') else label;sign=-1 if label.endswith(('down','neg')) else 1
  c=AuditController(m,axis,sign,config);d=mujoco.MjData(m);mujoco.mj_forward(m,d);samples=[];error=None
  try:
   for k in range(1,round(5/dt)+1):samples.append(advance(m,d,c,k))
  except Exception as exc:error=repr(exc)
  df=pd.DataFrame(samples);path=folder/f'{label}_{dt}.csv';df.to_csv(path,index=False);tail=df[df.time_s>=4.5]
  passed=bool(len(samples)==round(5/dt) and tail.position_error_m.max()<config['position_acceptance_m'] and tail.orientation_error_rad.max()<config['orientation_acceptance_rad'] and df.max_site_gap_m.max()<config['max_weld_gap_m'] and df.max_site_angle_rad.max()<config['max_weld_angle_rad'] and df.warning_count.max()==0)
  row=dict(timestep_s=dt,scenario=label,steps=len(samples),passed_unchanged_criteria=passed,max_weld_gap_m=float(df.max_site_gap_m.max()),max_weld_angle_rad=float(df.max_site_angle_rad.max()),final_position_error_mm=float(df.position_error_m.iloc[-1]*1000),final_orientation_error_deg=float(np.rad2deg(df.orientation_error_rad.iloc[-1])),max_force_N=float(max(df[f'actuator_force_N_{j}'].abs().max() for j in range(1,7))),error=error,log=str(path.relative_to(ROOT)));rows.append(row);pd.DataFrame(rows).to_csv(ROOT/'results/phase0_timestep_sensitivity.csv',index=False);print(json.dumps(row),flush=True)
write_json(folder/'complete.json',dict(trials=len(rows),physical_steps=sum(x['steps'] for x in rows),source_protocol_sha256=sha(ROOT/'configs/phase0_protocol.json')))
