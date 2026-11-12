"""Conservative mesh-envelope proxies in an independent QUERY model.
Physical contact is not injected into the frozen control comparison plant.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import itertools
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ..model import ROOT,UPSTREAM,load_model,write_json,sha

PROXY_XML=ROOT/'models/phase1_proxy_query.xml'
def original_mesh_vertices_local(model,body_name):
    body=model.body(body_name);gid=int(body.geomadr[0]);mesh=int(model.geom_dataid[gid]);a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
    q=model.geom_quat[gid];r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return model.mesh_vert[a:a+n].astype(float)@r.T+model.geom_pos[gid]
def build_proxy_model():
    model=load_model();root=ET.parse(UPSTREAM/'stewart_platform.xml').getroot();root.find('compiler').set('meshdir',str(UPSTREAM/'assets'))
    bodies={b.attrib['name']:b for b in root.iter('body')};bounds=[]
    for i in range(1,7):
        for part in ['legbot','legup']:
            body=f'{part}_{i}';v=original_mesh_vertices_local(model,body);radius=float(np.linalg.norm(v[:,:2],axis=1).max());lo=float(v[:,2].min());hi=float(v[:,2].max())
            geom=ET.SubElement(bodies[body],'geom',name=f'proxy_{body}',type='capsule',fromto=f'0 0 {lo:.12g} 0 0 {hi:.12g}',size=f'{radius:.12g}',mass='0',contype='1',conaffinity='1',rgba='0.1 0.8 0.3 0.2')
            bounds.append(dict(name=geom.attrib['name'],body=body,radius_m=radius,z_min_m=lo,z_max_m=hi,coverage='Contains all original compiled mesh vertices; convex radial envelope, intentionally conservative'))
    for body in ['base','upper']:
        v=original_mesh_vertices_local(model,body);radius=float(np.linalg.norm(v[:,:2],axis=1).max());lo=float(v[:,2].min());hi=float(v[:,2].max());ET.SubElement(bodies[body],'geom',name=f'proxy_{body}',type='cylinder',pos=f'0 0 {(lo+hi)/2:.12g}',size=f'{radius:.12g} {(hi-lo)/2:.12g}',mass='0',contype='1',conaffinity='1',rgba='.8 .3 .1 .2')
        bounds.append(dict(name=f'proxy_{body}',body=body,radius_m=radius,z_min_m=lo,z_max_m=hi,coverage='Enclosing solid cylinder; holes/edge concavity filled, not an exact CAD collision surface'))
    world=root.find('worldbody');ET.SubElement(world,'geom',name='proxy_floor',type='plane',size='0 0 .05',contype='1',conaffinity='1')
    contacts=ET.SubElement(root,'contact');exclusions=[]
    for i in range(1,7):
        for a,b,reason in [(f'legbot_{i}',f'legup_{i}','Same telescopic branch: intentional nested parts'),('base',f'legbot_{i}','Base universal attachment'),('upper',f'legup_{i}','Platform spherical attachment')]:
            ET.SubElement(contacts,'exclude',body1=a,body2=b);exclusions.append(dict(body1=a,body2=b,reason=reason))
    ET.SubElement(contacts,'exclude',body1='base',body2='world');exclusions.append(dict(body1='base',body2='world',reason='Fixed support resting at floor; not a robot motion collision'))
    PROXY_XML.parent.mkdir(exist_ok=True);ET.ElementTree(root).write(PROXY_XML,encoding='utf-8',xml_declaration=True)
    query=mujoco.MjModel.from_xml_path(str(PROXY_XML));pairs=[]
    excluded={frozenset([e['body1'],e['body2']]) for e in exclusions}
    gids=[i for i in range(query.ngeom) if query.geom(i).name.startswith('proxy_')]
    for a,b in itertools.combinations(gids,2):
        ba=query.body(int(query.geom_bodyid[a])).name;bb=query.body(int(query.geom_bodyid[b])).name
        if ba==bb or frozenset([ba,bb]) in excluded:continue
        pairs.append((a,b))
    return query,pairs,dict(proxies=bounds,exclusions=exclusions,not_covered='Cardan joints have no separate proxy. Original meshes retain disabled contact flags; full-robot exact collision coverage is not certified.',query_xml=str(PROXY_XML.relative_to(ROOT)),source_xml_sha256=sha(UPSTREAM/'stewart_platform.xml'),query_xml_sha256=sha(PROXY_XML),native_mass_max_change=float(max(abs(query.body_mass-model.body_mass))),proxy_pairs=len(pairs))

class CollisionQuery:
    def __init__(self):
        self.model,self.pairs,self.description=build_proxy_model();self.data=mujoco.MjData(self.model)
    def evaluate(self,qpos):
        if np.shape(qpos)!=(self.model.nq,) or not np.all(np.isfinite(qpos)):raise ValueError('Invalid collision query qpos')
        self.data.qpos[:]=qpos;mujoco.mj_forward(self.model,self.data);minimum=.2;offending=None;truncated=True;cross_min=.2;cross_pair=None
        for a,b in self.pairs:
            value=float(mujoco.mj_geomDistance(self.model,self.data,a,b,.2,None))
            if value<minimum:minimum=value;offending=[self.model.geom(a).name,self.model.geom(b).name];truncated=False
            names=[self.model.geom(a).name,self.model.geom(b).name]
            if all(n.startswith(('proxy_legbot_','proxy_legup_')) for n in names) and names[0].split('_')[-1]!=names[1].split('_')[-1] and value<cross_min:cross_min=value;cross_pair=names
        return dict(minimum_clearance_m=minimum,distance_is_lower_bound=truncated,nearest_pair=offending,cross_leg_minimum_clearance_m=cross_min,cross_leg_nearest_pair=cross_pair,native_contacts=int(self.data.ncon),query_only=True)
