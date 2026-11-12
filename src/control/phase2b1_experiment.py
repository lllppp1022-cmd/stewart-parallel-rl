"""Additive controller selection/logging; frozen Phase2A step and safety are reused."""
import json,time
from pathlib import Path
import numpy as np
from ..disturbance.executor import DisturbanceOwner
from ..disturbance.metrics import compute,vec,error_series
from ..model import ROOT,write_json
from .leaky_pid import LeakyPID
from .robust_controller_c3 import RobustControllerC3

class RobustOwner(DisturbanceOwner):
    def __init__(self,config,*channels):
        name=config['controller'];alias='C1' if name=='C1-L' else 'C0' if name=='C3' else name
        super().__init__(dict(config,controller=alias),*channels);self.config=dict(config)
        if name=='C1-L':self.controller=LeakyPID(self.nominal,config.get('leak_per_s',0))
        elif name=='C3':self.controller=RobustControllerC3(self.nominal,config['task_model'],config['bandwidth_rad_s'],config['compensation_gain'],config.get('force_bound_N',5),config.get('force_rate_N_s',20))
        self.previous_logged_force=np.zeros(6);append=self.recorder.append
        def extended(row,state):
            diagnostics=getattr(self.controller,'diagnostics',{});row['controller_measurement_time_s']=(self.n-1)*.001
            for key in ('position_error_m','integral_state','integral_force_N','leak_delta_m_s','leak_force_delta_N','antiwindup_mask','base_force_N','compensation_N','desired_compensation_N','equivalent_disturbance','compensation_target_wrench','realized_compensation_wrench','position_coordinates'):
                values=np.asarray(diagnostics.get(key,np.zeros(6)),float)
                for j,value in enumerate(values):row[f'diag_{key}_{j}']=float(value)
            for key in ('observer_clipped','compensation_rate_limited','compensation_amplitude_limited'):row['diag_'+key]=float(diagnostics.get(key,0))
            force=np.array([row[f'force_{j}'] for j in range(6)])
            for j,value in enumerate((force-self.previous_logged_force)/.001 if self.n>1 else np.zeros(6)):row[f'actual_force_rate_N_s_{j}']=float(value)
            self.previous_logged_force=force;append(row,state)
        self.recorder.append=extended;write_json(Path(config['folder'])/'config.json',config)
        plant=json.loads((Path(config['folder'])/'plant.json').read_text(encoding='utf-8'));plant['legacy_base_external_channel_policy']=plant.pop('external_channel_policy');plant['external_channel_policy']='Explicit independent Phase2B1 disturbance_profile; controller never receives profile';write_json(Path(config['folder'])/'plant.json',plant)

def trial_metrics(folder,protocol):
    import pandas as pd
    result=compute(folder,protocol);df=pd.read_csv(Path(folder)/'physics.csv.gz');p,a,rr=error_series(df);cfg=json.loads((Path(folder)/'config.json').read_text(encoding='utf-8'));pr=cfg['disturbance'];load=(df.input_step>=pr['start_step'])&(df.input_step<pr['stop_step']);post=(df.input_step>=pr['stop_step']);xy=np.linalg.norm(vec(df,'p',3)[:,:2]-vec(df,'p_ref',3)[:,:2],axis=1)*1000
    for prefix,mask in [('load',load),('unload',post)]:
        result[prefix+'_peak_position_mm']=float(p[mask].max()) if mask.any() else None;result[prefix+'_peak_orientation_deg']=float(a[mask].max()) if mask.any() else None;result[prefix+'_non_target_xy_peak_mm']=float(xy[mask].max()) if mask.any() else None;result[prefix+'_non_target_yaw_peak_deg']=float(abs(rr[mask,2]).max()*180/np.pi) if mask.any() else None
    position_z=(vec(df,'p',3)-vec(df,'p_ref',3))[:,2]*1000;actual_angle=-rr*180/np.pi;w=np.asarray(pr['wrench'])
    for name,error,command in [('z_mm',position_z,w[2]),('roll_deg',actual_angle[:,0],w[3]),('pitch_deg',actual_angle[:,1],w[4])]:result['unload_reverse_'+name]=float(max(0,np.max(-np.sign(command)*error[post]))) if post.any() and command!=0 else None
    result['actuator_force_margin_N']=400-result['max_force_N'];result['leak_per_s']=cfg.get('leak_per_s');result['bandwidth_rad_s']=cfg.get('bandwidth_rad_s');result['compensation_gain']=cfg.get('compensation_gain');result['split']=cfg.get('split',cfg['stage']);result['max_compensation_N']=float(abs(vec(df,'diag_compensation_N',6)).max());result['max_compensation_rate_N_s']=float(abs(np.diff(vec(df,'diag_compensation_N',6),axis=0)/.001).max()) if len(df)>1 else 0
    result['objective']=(result['load_position_iae_mm_s']+result['unload_position_iae_mm_s'])/.25+(result['load_orientation_iae_deg_s']+result['unload_orientation_iae_deg_s'])/.02 if result['success'] else None
    return result
