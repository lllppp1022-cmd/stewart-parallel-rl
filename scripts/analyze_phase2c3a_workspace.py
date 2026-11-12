"""One bounded ±100mm offline full-chain scan; NEVER integrates or moves live state."""
import _bootstrap,json,time,itertools
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,write_json,sha
from src.workspace.geometry import ExpandedGeometryQuery,inspect_pose
from src.kinematics.stewart_geometry import StewartGeometry
from src.phase1.plant import create_plant
from src.visualization.source import ReadOnlyProxy
from src.audit_control import scratch_reference_data

RUN=ROOT/'results/runs/phase2c3a_20261103_v1'

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    frozen={}
    for sub in ['src','configs','docs','third_party','models']:
        for f in (ROOT/sub).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and 'phase2c3a' not in f.name and 'workspace' not in f.parts:frozen[str(f.relative_to(ROOT))]=sha(f)
    for mf in ['results/phase2c2_artifact_manifest.json','results/phase2c1_artifact_manifest.json']:
        prior=json.loads((ROOT/mf).read_text(encoding='utf-8'))['files']
        for name,meta in prior.items():frozen[name.replace('/','\\')]=meta['sha256']
    write_json(RUN/'history_freeze.json',frozen)
    m=create_plant();g=StewartGeometry(m);query=ExpandedGeometryQuery();proxy=ReadOnlyProxy();audit=dict(nq=m.nq,nv=m.nv,nu=m.nu,neq=m.neq,dt_s=m.opt.timestep,platform_mass_kg=float(m.body('upper').mass[0]),platform_inertia=m.body('upper').inertia.tolist(),platform_com=m.body('upper').ipos.tolist(),native_geoms=m.ngeom,native_collision_flags=[[int(a),int(b)] for a,b in zip(m.geom_contype,m.geom_conaffinity)],joint_audit=[dict(name=m.joint(j).name,type=int(m.jnt_type[j]),limited=bool(m.jnt_limited[j]),range=m.jnt_range[j].tolist(),qpos_adr=int(m.jnt_qposadr[j]),axis=m.jnt_axis[j].tolist()) for j in range(m.njnt)],equality_types=m.eq_type.tolist(),nativeccd=query.nativeccd,mesh_pairs=len(query.pairs),excluded_pairs=query.exclusions,full_mesh_pair_names=[dict(a=a,b=b,body1=ba,body2=bb) for a,b,ba,bb in query.pairs],missing='30 hinges have no native mechanical limits; connected socket interiors and real hardware stops unknown; no cables, hoses or support deformation model',reference_source='Full six-joint chains, not attachment chord approximation')
    write_json(RUN/'model_geometry_audit.json',audit);rows=[];start=time.perf_counter()
    # Separate continuation from Home in each direction prevents opposite-end seeds.
    for sign in [1,-1]:
        seed=None
        for dz in np.arange(0,.1000001,.002)*sign:
            if sign==-1 and dz==0:continue
            row,sol=inspect_pose(m,g,query,float(dz),seed,True)
            if sol.success:
                seed=sol.chain_q;data=scratch_reference_data(m,g,sol,g.home_position+[0,0,dz],g.home_rotation);row['legacy_proxy_gap_m']=proxy.evaluate(data.qpos)['minimum_clearance_m']
            rows.append(row)
    df=pd.DataFrame(rows).sort_values('delta_z_m');df.to_csv(ROOT/'results/phase2c3a_workspace_scan.csv',index=False);write_json(RUN/'workspace_scan_raw.json',rows)
    candidates={}
    for mm in [30,50]:
        sel=df[(df.delta_z_m>=-1e-10)&(df.delta_z_m<=mm/1000+1e-10)];candidates[f'L{mm}']=dict(simulation_preflight_passed=bool(sel.simulation_geometric_candidate.all()),all_modeled_limits_pass=bool(sel.JOINT_LIMIT_VALID.all()),minimum_gap_m=float(sel.expanded_mesh_gap_m.min()),maximum_condition=float(sel.jacobian_condition.max()),mechanical_passive_stop_certified=False,scope='Modeled constraints and conservative mesh clearance only; actual hardware/sockets incomplete')
    bounds={}
    for sign,label in [(1,'positive'),(-1,'negative')]:
        part=df[df.delta_z_m*sign>=-1e-10].sort_values('delta_z_m',ascending=sign==1);valid=part[part.simulation_geometric_candidate]
        bounds[label]=dict(last_sampled_candidate_m=float(valid.delta_z_m.iloc[-1]) if len(valid) else None,scan_endpoint_only=bool(len(valid)==len(part)),first_rejections=part[~part.simulation_geometric_candidate].head(3).fillna('UNKNOWN').to_dict('records'))
    write_json(ROOT/'results/phase2c3a_workspace_limits.json',dict(candidates=candidates,bounds=bounds,samples=len(df),scan_spacing_m=.002,seconds=time.perf_counter()-start,query_range_m=[-.1,.1],not_dynamic_authorisation=True,not_mechanical_or_continuous_certification=True));print(json.dumps(candidates),flush=True)
if __name__=='__main__':main()
