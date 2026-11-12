"""Independent nominal/perturbed plant creation; source MJCF is never edited."""
import numpy as np,mujoco
from ..model import load_model

def create_plant(condition='nominal',dt=.001):
    m=load_model();m.opt.timestep=dt;b=m.body('upper').id
    if condition=='mass110':m.body_mass[b]*=1.1;m.body_inertia[b]*=1.1
    elif condition=='com_x3mm':m.body_ipos[b,0]+=.003
    elif condition=='gain090':m.actuator_gainprm[:6,0]*=.9
    elif condition=='asymmetric_gain':m.actuator_gainprm[:6,0]*=np.array([.9,1.05,.95,1.,.92,1.02])
    elif condition not in ('nominal','external_load'):raise ValueError('Unknown frozen perturbation')
    # Recompute constants after parameter edits, before any live integration.
    scratch=mujoco.MjData(m);mujoco.mj_setConst(m,scratch)
    return m
def plant_metadata(m,condition):
    b=m.body('upper').id
    return dict(condition=condition,timestep_s=float(m.opt.timestep),platform_mass_kg=float(m.body_mass[b]),platform_inertia_kg_m2=m.body_inertia[b].tolist(),platform_com_local_m=m.body_ipos[b].tolist(),gain=m.actuator_gainprm[:6,0].tolist(),ctrlrange=m.actuator_ctrlrange.tolist(),forcerange=m.actuator_forcerange.tolist(),eq_solref=m.eq_solref.tolist(),eq_solimp=m.eq_solimp.tolist(),refsafe_enabled=not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)),external_channel_policy='Only explicit external_load at 2<=t<4: site-local [0,0,-10N,0.5Nm,0,0], otherwise zeros')
def external_command(condition,t):
    return np.array([0.,0.,-10.,.5,0.,0.]) if condition=='external_load' and 2<=t<4 else np.zeros(6)
def platform_state(model,data):
    site=model.site('upper_center').id;speed=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,site,speed,0)
    return data.site_xpos[site].copy(),data.site_xmat[site].reshape(3,3).copy(),np.r_[speed[3:],speed[:3]]
