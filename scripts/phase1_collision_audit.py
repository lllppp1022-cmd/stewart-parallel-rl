import _bootstrap
import json
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,load_model,write_json,UPSTREAM
from src.phase1.collision import CollisionQuery,original_mesh_vertices_local
from src.phase1.trajectory import Trajectory,protocol
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.audit_control import scratch_reference_data

def main():
    query=CollisionQuery();m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);home=query.evaluate(d.qpos);rows=[]
    for name in protocol()['nominal_trajectories']:
        tr=Trajectory(m,name,ROOT/'results/phase1_preflight/references')
        for t in np.linspace(0,6,31):
            ref=tr.at(t);fit=inverse_kinematics(tr.geometry,ref.p,ref.r,tr.chain[np.argmin(abs(tr.times-t))]);scratch=scratch_reference_data(m,tr.geometry,fit,ref.p,ref.r);result=query.evaluate(scratch.qpos)
            rows.append(dict(trajectory=name,time_s=t,**result))
        print('COLLISION QUERY',name,min(r['minimum_clearance_m'] for r in rows if r['trajectory']==name),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase1_reference_clearance.csv',index=False)
    result=dict(**query.description,original_robot_contact_disabled=True,home=home,home_no_proxy_native_contacts=home['native_contacts']==0,reference_samples=len(rows),minimum_reference_clearance_m=min(r['minimum_clearance_m'] for r in rows),formal_plant_unchanged=True,status='PARTIAL_QUERY_ONLY; no full collision-safety qualification',workspace_policy='Frozen small Phase0 envelope only; no large-range RL exploration',distmax_m=.2,distmax_semantics='Returned 0.2 means lower bound >=0.2, not exact distance')
    write_json(ROOT/'results/phase1_collision_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
