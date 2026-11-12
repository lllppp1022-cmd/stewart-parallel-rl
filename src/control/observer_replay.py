"""Causal OFFLINE replay of measured feedback and known commands, no reintegration.

This module is diagnostic only. Truth columns are never passed to the observer.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from ..phase1.controllers import Controller
from ..phase1.plant import create_plant
from ..phase1.trajectory import Trajectory
from ..disturbance.metrics import vec
from ..model import ROOT
from .disturbance_observer import MomentumDisturbanceObserver
from .nominal_task_model import pose_coordinates

def replay_observer(folder,model,bandwidth):
    folder=Path(folder);df=pd.read_csv(folder/'physics.csv.gz');initial=json.loads((folder/'initial_state.json').read_text(encoding='utf-8'));nominal=create_plant();c=Controller(nominal,'C0',.001);ref=Trajectory(nominal,'home',ROOT/'results/phase1_preflight/references').at(0);obs=MomentumDisturbanceObserver(model,bandwidth)
    p=np.array(model['home_position']);r=np.array(model['home_rotation']);v=np.zeros(6);q=np.array(initial['qpos']);dq=np.array(initial['qvel']);out=[];drives=[];measurements=[]
    positions=vec(df,'p',3);rotations=Rotation.from_euler('xyz',vec(df,'rpy',3),degrees=True).as_matrix();velocities=vec(df,'twist_actual',6);qs=vec(df,'qpos',43);dqs=vec(df,'qvel',42);commands=vec(df,'f_cmd',6)
    for k in range(len(df)):
        c.command(q,dq,p,r,v,ref,(k+1)*.001);x=pose_coordinates(model,p,r);out.append(obs.update(x,v));obs.remember_applied_command(x,v,c.jacobian.T@commands[k]);drives.append(obs.previous_drive.copy());measurements.append(v.copy());p=positions[k];r=rotations[k];v=velocities[k];q=qs[k];dq=dqs[k]
    added={**{f'replay_nominal_drive_{j}':np.asarray(drives)[:,j] for j in range(6)},**{f'replay_measured_pre_velocity_{j}':np.asarray(measurements)[:,j] for j in range(6)}}
    df=pd.concat([df,pd.DataFrame(added)],axis=1)
    return np.asarray(out),df
