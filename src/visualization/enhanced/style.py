"""Single authoritative actuator-order colour map, display only."""
import numpy as np
LEG_COLORS=('#2196F3','#F59E0B','#22C55E','#EF4444','#A855F7','#06B6D4')
MODES=('Absolute','Delta from Home','Tracking Error')
def rgba(i):return np.array([int(LEG_COLORS[i][k:k+2],16)/255 for k in (1,3,5)]+[1.])
def leg_mapping(model):
    rows=[]
    for i in range(6):
        jid=int(model.actuator_trnid[i,0]);name=model.joint(jid).name
        if name!=f'joint_lu_{i+1}':raise ValueError('Unexpected native actuator order: '+name)
        bodies=[int(model.body(f'{part}_{i+1}').id) for part in ('legbot','legup')]
        geoms=np.flatnonzero(np.isin(model.geom_bodyid,bodies)).tolist()
        rows.append(dict(leg=i+1,actuator_id=i,joint=name,qpos_address=int(model.jnt_qposadr[jid]),geoms=geoms,color=LEG_COLORS[i]))
    return rows
def color_display_model(model):
    mapping=leg_mapping(model)
    for row in mapping:
        for gid in row['geoms']:model.geom_matid[gid]=-1;model.geom_rgba[gid]=rgba(row['leg']-1)
    return mapping
def series(actual,reference,group,mode,home=None,angle_unit='deg'):
    a=np.asarray(actual,float);r=np.asarray(reference,float)
    if group=='angles':return a*(1000 if angle_unit=='mdeg' else 1),r*(1000 if angle_unit=='mdeg' else 1),angle_unit
    if group=='force':return a,r,'N'
    if mode=='Tracking Error':return (a-r)*1000,np.zeros_like(r),'mm'
    if mode=='Delta from Home':
        if home is None:raise ValueError('Home baseline UNKNOWN; select Absolute/Error or provide recorded baseline')
        baseline=np.asarray(home['p_m' if group=='position' else 'slide_m'],float)
        a=a-baseline;r=r-baseline
    return a*1000,r*1000,'mm'
