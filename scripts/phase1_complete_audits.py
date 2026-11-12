import _bootstrap
import json,subprocess
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,load_model,write_json,sha,UPSTREAM
m=load_model();path=ROOT/'results/phase1_collision_audit.json';r=json.loads(path.read_text(encoding='utf-8'))
r['original_geometries']=[dict(id=i,body=m.body(int(m.geom_bodyid[i])).name,type=int(m.geom_type[i]),mesh_id=int(m.geom_dataid[i]),contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),name=m.geom(i).name) for i in range(m.ngeom)]
r['source_to_query_differences']=dict(original_robot_meshes_unchanged=44,new_massless_leg_capsules=12,new_massless_base_top_cylinders=2,floor_recreated_at_original_plane=True,original_joints_and_weld_unchanged=True,full_cardan_collision_coverage=False,contact_dynamics_enabled_in_formal_plant=False,query_meshdir='Absolute local asset path; regenerate query XML when relocating project')
write_json(path,r)
table=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');groups=[]
for method,g in table[(table.condition=='nominal')&(table.timestep_s==.001)].groupby('controller'):
 w=g.physical_steps.to_numpy();groups.append(dict(controller=method,trials=len(g),completed=int(g.success.sum()),pooled_position_rmse_mm=float(np.sqrt(np.average(g.position_rmse_mm.to_numpy()**2,weights=w))),pooled_orientation_rmse_deg=float(np.sqrt(np.average(g.orientation_rmse_deg.to_numpy()**2,weights=w))),mean_task_position_rmse_mm=float(g.position_rmse_mm.mean()),mean_task_orientation_rmse_deg=float(g.orientation_rmse_deg.mean())))
pd.DataFrame(groups).to_csv(ROOT/'results/phase1_nominal_pooled_metrics.csv',index=False)
logs=[]
for folder in sorted((ROOT/'results/runs').glob('phase1_gui_*')):
 if (folder/'demo_exit.json').exists():logs.append(dict(folder=str(folder.relative_to(ROOT)),exit=json.loads((folder/'demo_exit.json').read_text(encoding='utf-8')),summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))))
write_json(ROOT/'results/phase1_gui_validation.json',dict(actual_native_viewer_observed=True,window_label='DYNAMIC ACTUATOR SIMULATION / C2 / z_roll',same_live_model_and_data=True,per_physics_viewer_lock=False,normal_python_exits=logs,no_long_training=True))
write_json(ROOT/'results/phase1_versions.json',dict(client_date='2026-10-27',mujoco=mujoco.__version__,python=subprocess.check_output([__import__('sys').executable,'--version'],text=True).strip(),protocol_sha256=sha(ROOT/'configs/phase1_protocol.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),no_rl_installation=True))
print(pd.DataFrame(groups).to_string(index=False));print('force max',table.max_force_N.max(),'weld max',table.max_weld_angle_rad.max(),'steps',table.physical_steps.sum())
