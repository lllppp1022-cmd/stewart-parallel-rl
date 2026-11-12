"""Actual force-input integration, full-rate logs, reactive holds and honest failures."""
import json,time,gzip,csv,traceback
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from scipy.spatial.transform import Rotation
from ..model import ROOT,write_json,ids,site_gap,equality_residual
from .trajectory import protocol
from .controllers import Controller
from .plant import create_plant,plant_metadata,external_command,platform_state

def errors(ref,p,r):return float(np.linalg.norm(ref.p-p)),float(np.linalg.norm(Rotation.from_matrix(ref.r@r.T).as_rotvec()))
def check_safety(model,data,p,r,geometry,gap,angle,condition,proxy,config=None):
    cfg=protocol() if config is None else config;qadr,_=ids(model)
    if not all(np.all(np.isfinite(v)) for v in [data.qpos,data.qvel,data.qacc,data.actuator_force]):return 'nonfinite_state'
    if any(w.number for w in data.warning):return 'mujoco_warning'
    if gap>cfg['weld_translation_limit_m']:return 'weld_translation'
    if angle>cfg['weld_angle_limit_rad']:return 'weld_angle'
    if np.any(data.qpos[qadr]<geometry.stroke_ranges[:,0]) or np.any(data.qpos[qadr]>geometry.stroke_ranges[:,1]):return 'slide_limit'
    if condition is None or condition>cfg['jacobian_condition_limit']:return 'jacobian_condition'
    shift=p-geometry.home_position
    if max(abs(shift[:2]))>cfg['actual_xy_limit_m'] or abs(shift[2])>cfg['actual_z_limit_m']:return 'restricted_position_envelope'
    euler=Rotation.from_matrix(r@geometry.home_rotation.T).as_euler('xyz',degrees=True)
    if max(abs(euler[:2]))>cfg['actual_roll_pitch_limit_deg'] or abs(euler[2])>cfg['actual_yaw_limit_deg']:return 'restricted_orientation_envelope'
    if proxy['minimum_clearance_m']<cfg['partial_proxy_min_clearance_m']:return 'covered_proxy_clearance'
    return None

def run_trial(nominal_model,trajectory,method,condition,dt,folder,collision_query,frame_callback=None,realtime=False,viewer=None,viewer_factory=None,time_budget_s=None):
    cfg=protocol();folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model=create_plant(condition,dt);data=mujoco.MjData(model);mujoco.mj_forward(model,data);qadr,dof=ids(model)
    controller=Controller(nominal_model,method,cfg['controller_period_s']);write_json(folder/'plant.json',plant_metadata(model,condition));write_json(folder/'protocol.json',cfg)
    duration=cfg['duration_s'] if time_budget_s is None else float(time_budget_s)
    if not 0<duration<=120:raise ValueError('A finite budget in (0,120] seconds is required')
    n=round(duration/dt);control_stride=round(cfg['controller_period_s']/dt);query_stride=round(cfg['proxy_query_period_s']/dt);frame_stride=round(.1/dt)
    if control_stride<1 or not np.isclose(control_stride*dt,cfg['controller_period_s']):raise ValueError('Controller/physics periods not commensurate')
    min_proxy=collision_query.evaluate(data.qpos);rows=[];allq=[];allv=[];allu=[];events=[];failure=None;protection_until=None;holdq=None;holdff=None;u=np.zeros(6);raw=np.zeros(6)
    info=dict(integral=np.zeros(6),condition=controller.condition,clipped=False,integral_clamp_count=0,antiwindup_count=0,wrench_projection_error=0.)
    if viewer_factory is not None:viewer=viewer_factory(model,data)
    wall=time.perf_counter();control_stamp=0.;last_ref=trajectory.at(0);closed=False;unexpected=None;csvfile=folder/'physics.csv.gz'
    write_json(folder/'started.json',dict(controller=method,condition=condition,trajectory=trajectory.name,dt=dt,maximum_tracking_steps=n,status='running; missing summary means failure/interruption'))
    try:
        with gzip.open(csvfile,'wt',encoding='utf-8',newline='',compresslevel=1) as stream:
            writer=None;k=0
            while k<n or (protection_until is not None and data.time<protection_until-1e-12):
                if viewer is not None and not viewer.is_running():closed=True;failure=failure or 'viewer_closed';break
                if protection_until is not None and data.time>=protection_until-1e-12:break
                previous=float(data.time);updated=(k%control_stride==0);control_seconds=0.
                p,r,twist=platform_state(model,data)
                if updated:
                    tick=time.perf_counter();control_stamp=previous+cfg['controller_period_s'];last_ref=trajectory.at(control_stamp)
                    if failure:
                        controller.reset();raw=holdff+cfg['kp_leg_N_per_m']*(holdq-data.qpos[qadr])-cfg['kd_leg_Ns_per_m']*data.qvel[dof];u=np.clip(raw,-400,400);info['integral']=np.zeros(6);info['clipped']=bool(np.any(raw!=u))
                    else:
                        try:u,info=controller.command(data.qpos,data.qvel,p,r,twist,last_ref,control_stamp);raw=info['raw']
                        except Exception as exc:
                            failure='controller_rejected:'+str(exc);controller.reset();holdq=data.qpos[qadr].copy();holdff=last_ref.ff.copy();raw=holdff.copy();u=np.clip(raw,-400,400);protection_until=data.time+cfg['protection_hold_s'];events.append(dict(time_s=previous,reason=failure))
                    control_seconds=time.perf_counter()-tick
                data.ctrl[:6]=u;data.ctrl[6:]=external_command(condition,previous+dt)
                physics_start=time.perf_counter();mujoco.mj_step(model,data);mujoco.mj_forward(model,data);physics_seconds=time.perf_counter()-physics_start;k+=1
                if not np.isclose(data.time,previous+dt,atol=1e-9):raise RuntimeError('Unexpected time reset / integration count')
                if not np.all(np.isfinite(data.qpos)):raise RuntimeError('Nonfinite live state; no continued hold possible')
                p,r,twist=platform_state(model,data);ref=trajectory.at(data.time);gap=site_gap(model,data);angles=np.array([np.linalg.norm(Rotation.from_matrix(data.site(f'site_c1_{i+12}').xmat.reshape(3,3)@data.site(f'site_upper_{i}').xmat.reshape(3,3).T).as_rotvec()) for i in range(1,7)])
                if k==1 or k%query_stride==0:min_proxy=collision_query.evaluate(data.qpos)
                rejection=check_safety(model,data,p,r,trajectory.geometry,float(max(gap)),float(max(angles)),info['condition'],min_proxy,cfg)
                if rejection and not failure:
                    failure=rejection;controller.reset();holdq=data.qpos[qadr].copy();holdff=ref.ff.copy();protection_until=data.time+cfg['protection_hold_s'];events.append(dict(time_s=float(data.time),reason=failure,actual_qpos=data.qpos.tolist(),proxy_pair=min_proxy['nearest_pair']))
                pe,ae=errors(ref,p,r)
                row=dict(controller=method,condition=condition,trajectory=trajectory.name,step=k,time_s=float(data.time),control_time_s=control_stamp,controller_updated=updated,phase='protect_hold' if failure else ('settle' if data.time>=5 else 'tracking'),position_error_m=pe,orientation_error_rad=ae,joint_rmse_m=float(np.sqrt(np.mean((ref.q-data.qpos[qadr])**2))),max_weld_translation_m=float(max(gap)),max_weld_angle_rad=float(max(angles)),mixed_equality_residual=float(max(abs(equality_residual(data)))),jacobian_condition=float(info['condition']),force_command_clipped=bool(info['clipped']),native_force_saturated=bool(max(abs(data.actuator_force[:6]))>=400-1e-8),integral_limit_count=info['integral_clamp_count'],antiwindup_count=info['antiwindup_count'],minimum_proxy_clearance_m=min_proxy['minimum_clearance_m'],proxy_distance_lower_bound=min_proxy['distance_is_lower_bound'],proxy_contacts=min_proxy['native_contacts'],original_native_contacts=int(data.ncon),controller_compute_s=control_seconds,physics_step_s=physics_seconds,warning_count=int(sum(w.number for w in data.warning)),reference_progress=min(float(data.time)/cfg['duration_s'],1.),failed=bool(failure))
                row['cross_leg_proxy_clearance_m']=min_proxy['cross_leg_minimum_clearance_m']
                vectors=[('p_ref_m',ref.p),('p_actual_m',p),('quat_ref_wxyz',Rotation.from_matrix(ref.r).as_quat()[[3,0,1,2]]),('quat_actual_wxyz',data.body('upper').xquat),('twist_ref_world',ref.twist),('twist_actual_world',twist),('slide_ref_m',ref.q),('slide_actual_m',data.qpos[qadr]),('dslide_ref_mps',ref.dq),('dslide_actual_mps',data.qvel[dof]),('ctrl_leg_N',u),('raw_force_N',raw),('F_actual_N',data.actuator_force[:6]),('integral_m_s',info['integral']),('external_ctrl',data.ctrl[6:]),('external_wrench_world',np.r_[r@data.ctrl[6:9],r@data.ctrl[9:12]]),('weld_gap_m',gap),('weld_angle_rad',angles),('qpos',data.qpos),('qvel',data.qvel)]
                for prefix,values in vectors:
                    for j,value in enumerate(values):row[f'{prefix}_{j+1}']=float(value)
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);rows.append(row);allq.append(data.qpos.copy());allv.append(data.qvel.copy());allu.append(data.ctrl.copy())
                if k%1000==0:stream.flush()
                if frame_callback is not None and (k%frame_stride==0):frame_callback(model,data,row,ref)
                if viewer is not None and k%round(.02/dt)==0:viewer.sync()
                if realtime:
                    delay=wall+k*dt-time.perf_counter()
                    if delay>0:time.sleep(min(delay,.01))
    except Exception as exc:
        unexpected=traceback.format_exc();failure=failure or ('exception:'+str(exc));events.append(dict(time_s=float(data.time),reason=failure,traceback=unexpected))
    finally:
        if viewer_factory is not None and viewer is not None and viewer.is_running():viewer.close()
    df=pd.DataFrame(rows);np.savez_compressed(folder/'states.npz',qpos=np.asarray(allq),qvel=np.asarray(allv),ctrl=np.asarray(allu),time=np.asarray(df['time_s']) if len(df) else np.array([]))
    write_json(folder/'events.json',events)
    if len(df)==0:raise RuntimeError('No executed physics rows: '+str(unexpected))
    tail=df[df.time_s>=cfg['duration_s']-cfg['held_tail_s']];complete=bool(not failure and len(rows)>=n and len(tail)>=round(.45/dt) and tail.position_error_m.max()<cfg['position_completion_limit_m'] and tail.orientation_error_rad.max()<cfg['angle_completion_limit_rad'])
    if not complete and failure is None:failure='completion_accuracy'
    updated_rows=df[df.controller_updated];summary=dict(controller=method,trajectory=trajectory.name,condition=condition,timestep_s=dt,controller_period_s=cfg['controller_period_s'],success=complete,failure_reason=failure or '',physical_steps=len(df),simulation_time_s=float(data.time),progress=float(min(data.time/cfg['duration_s'],1)),position_rmse_mm=float(np.sqrt(np.mean(df.position_error_m**2))*1000),orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(df.orientation_error_rad**2)))),slide_rmse_mm=float(np.sqrt(np.mean(df.joint_rmse_m**2))*1000),final_position_error_mm=float(df.position_error_m.iloc[-1]*1000),final_orientation_error_deg=float(np.rad2deg(df.orientation_error_rad.iloc[-1])),max_weld_translation_m=float(df.max_weld_translation_m.max()),max_weld_angle_rad=float(df.max_weld_angle_rad.max()),max_force_N=float(max(df[f'F_actual_N_{j}'].abs().max() for j in range(1,7))),command_saturated_steps=int(df.force_command_clipped.sum()),native_force_saturated_steps=int(df.native_force_saturated.sum()),integral_limit_count=controller.integral_clamp_count,antiwindup_count=controller.antiwindup_count,minimum_proxy_clearance_m=float(df.minimum_proxy_clearance_m.min()),native_contact_observations=int(df.original_native_contacts.sum()),proxy_contact_observations=int(df.proxy_contacts.sum()),controller_compute_median_us=float(updated_rows.controller_compute_s.median()*1e6),controller_compute_p95_us=float(updated_rows.controller_compute_s.quantile(.95)*1e6),maximum_condition=float(df.jacobian_condition.max()),warnings=int(df.warning_count.max()),first_failure_time_s=events[0]['time_s'] if events else None,viewer_closed=closed,wall_seconds=time.perf_counter()-wall,physics_log=str(csvfile.relative_to(ROOT)))
    write_json(folder/'summary.json',summary);return summary,df
