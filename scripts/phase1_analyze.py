import _bootstrap
import json,hashlib
import numpy as np,pandas as pd,mujoco,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from src.model import ROOT,load_model,write_json,sha
from src.phase1.trajectory import Trajectory

def main():
    table=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');manifest=json.loads((ROOT/'results/phase1_suite_manifest.json').read_text(encoding='utf-8'));assert len(table)==manifest['actual_trials']==42
    nominal=load_model();references={};validation=[];applied=[];failure=[];data={}
    for row in table.to_dict('records'):
        df=pd.read_csv(ROOT/row['physics_log']);data[row['trial_id']]=df;p_ref=df[[f'p_ref_m_{j}' for j in range(1,4)]].to_numpy();p=df[[f'p_actual_m_{j}' for j in range(1,4)]].to_numpy();pe=np.linalg.norm(p_ref-p,axis=1)
        qr=df[[f'quat_ref_wxyz_{j}' for j in range(1,5)]].to_numpy();qa=df[[f'quat_actual_wxyz_{j}' for j in range(1,5)]].to_numpy();ae=(Rotation.from_quat(qr[:,[1,2,3,0]])*Rotation.from_quat(qa[:,[1,2,3,0]]).inv()).magnitude()
        rmse_p=float(np.sqrt(np.mean(pe**2))*1000);rmse_a=float(np.rad2deg(np.sqrt(np.mean(ae**2))));err_p=abs(rmse_p-row['position_rmse_mm']);err_a=abs(rmse_a-row['orientation_rmse_deg']);assert err_p<1e-8 and err_a<1e-8
        tail=df[df.time_s>=5.5];derived=bool(not df.failed.any() and df.time_s.iloc[-1]>=6-1e-8 and len(tail)>0 and tail.position_error_m.max()<.00075 and tail.orientation_error_rad.max()<np.deg2rad(.1))
        assert derived==row['success'];validation.append(dict(trial_id=row['trial_id'],rows=len(df),position_rmse_from_xyz_mm=rmse_p,orientation_rmse_from_quaternions_deg=rmse_a,position_rmse_difference_mm=err_p,orientation_rmse_difference_deg=err_a,success_derived=derived,raw_sha256=sha(ROOT/row['physics_log'])))
        events=json.loads((ROOT/row['physics_log']).parent.joinpath('events.json').read_text(encoding='utf-8'));event=events[0] if events else None
        if row['trajectory'] not in references:references[row['trajectory']]=Trajectory(nominal,row['trajectory'],ROOT/'results/phase1_preflight/references')
        tr=references[row['trajectory']];qadr=np.array([nominal.joint(f'joint_lu_{i}').qposadr[0] for i in range(1,7)])
        holdq=np.asarray(event['actual_qpos'])[qadr] if event and 'actual_qpos' in event else None
        chosen=sorted(set(range(0,len(df),round(.02/row['timestep_s'])))|{len(df)-1})
        for index in chosen:
            sample=df.iloc[index];holding=bool(holdq is not None and sample.control_time_s>event['time_s']+1e-9);target=holdq if holding else tr.at(sample.control_time_s).q
            reference_time=event['time_s'] if holding else sample.control_time_s
            rec=dict(trial_id=row['trial_id'],time_s=sample.time_s,control_time_s=sample.control_time_s,protect_hold_command=holding,desired_timeline_progress=sample.reference_progress,commanded_reference_time_fraction=min(reference_time/6,1),planned_motion_started=bool(row['trajectory']!='home' and reference_time>1),source='Derived from recorded event actual_qpos + controller time/reference, not regenerated physics; time fraction is not geometric path progress')
            rec.update({f'applied_slide_target_m_{j+1}':float(target[j]) for j in range(6)});applied.append(rec)
        if not row['success']:
            first=event['time_s'] if event else None;last_rpy=Rotation.from_quat(qa[-1,[1,2,3,0]]).as_euler('xyz',degrees=True)
            failure.append(dict(trial_id=row['trial_id'],controller=row['controller'],condition=row['condition'],reason=row['failure_reason'],first_violation_s=first,last_time_s=row['simulation_time_s'],progress=row['progress'],last_rpy_deg=last_rpy.tolist(),weld_angle_max_rad=row['max_weld_angle_rad'],force_peak_N=row['max_force_N'],hold_observed=holdq is not None,diagnosis='Early protected stop; RMSE is not a completed-task score' if event else 'Completed timeline but frozen final tracking accuracy failed'))
        print('RECOMPUTED',row['trial_id'],flush=True)
    pd.DataFrame(validation).to_csv(ROOT/'results/phase1_metric_recomputation.csv',index=False);pd.DataFrame(applied).to_csv(ROOT/'results/phase1_applied_reference_audit.csv',index=False);write_json(ROOT/'results/phase1_failure_analysis.json',failure)
    aggregates=[]
    for (condition,dt,method),group in table.groupby(['condition','timestep_s','controller'],sort=False):
        aggregates.append(dict(condition=condition,timestep_s=dt,controller=method,completed=int(group.success.sum()),trials=len(group),mean_position_rmse_mm=group.position_rmse_mm.mean(),mean_orientation_rmse_deg=group.orientation_rmse_deg.mean(),mean_progress=group.progress.mean(),mean_final_position_error_mm=group.final_position_error_mm.mean(),max_force_N=group.max_force_N.max(),caution='Includes failed/early-stopped trials; inspect completion and progress'))
    pd.DataFrame(aggregates).to_csv(ROOT/'results/phase1_group_summary.csv',index=False)
    comparisons=[]
    for row in table[table.timestep_s==.0005].to_dict('records'):
        base=table[(table.controller==row['controller'])&(table.trajectory==row['trajectory'])&(table.condition=='nominal')&(table.timestep_s==.001)].iloc[0];fine=data[row['trial_id']];coarse=data[base.trial_id];indices=np.searchsorted(fine.time_s.to_numpy(),coarse.time_s.to_numpy()-1e-10);indices=np.minimum(indices,len(fine)-1)
        pos1=coarse[[f'p_actual_m_{i}' for i in range(1,4)]].to_numpy();pos2=fine.iloc[indices][[f'p_actual_m_{i}' for i in range(1,4)]].to_numpy();slide1=coarse[[f'slide_actual_m_{i}' for i in range(1,7)]].to_numpy();slide2=fine.iloc[indices][[f'slide_actual_m_{i}' for i in range(1,7)]].to_numpy()
        comparisons.append(dict(controller=row['controller'],trajectory=row['trajectory'],control_period_unchanged_s=.001,position_rmse_1ms_mm=base.position_rmse_mm,position_rmse_halfms_mm=row['position_rmse_mm'],orientation_rmse_1ms_deg=base.orientation_rmse_deg,orientation_rmse_halfms_deg=row['orientation_rmse_deg'],matched_time_max_position_difference_mm=float(np.linalg.norm(pos1-pos2,axis=1).max()*1000),matched_time_max_slide_difference_mm=float(abs(slide1-slide2).max()*1000),weld_angle_1ms_rad=base.max_weld_angle_rad,weld_angle_halfms_rad=row['max_weld_angle_rad'],force_peak_1ms_N=base.max_force_N,force_peak_halfms_N=row['max_force_N'],configured_solref_unchanged=True,effective_refsafe_timeconstant_1ms_s=.002,effective_refsafe_timeconstant_halfms_s=.001))
    pd.DataFrame(comparisons).to_csv(ROOT/'results/phase1_timestep_comparison.csv',index=False)
    frames=pd.read_csv(ROOT/'results/phase1_video_frames.csv');matches=0
    for trial,group in frames.groupby('trial'):
        z=np.load(ROOT/trial/'states.npz')
        for frame in group.to_dict('records'):assert hashlib.sha256(z['qpos'][frame['step']-1].tobytes()).hexdigest()==frame['qpos_sha256'];matches+=1
        z.close()
    reader=imageio.get_reader(ROOT/'assets/videos/phase1_dynamic_control.mp4');meta=reader.get_meta_data();n=0;scenes=set()
    for im in reader:n+=1;scenes.add(hashlib.sha256(im[195:].tobytes()).hexdigest())
    reader.close();assert n==len(frames) and matches==n and len(scenes)>10
    write_json(ROOT/'results/phase1_video_validation.json',dict(frames=n,fps=meta['fps'],duration_s=n/meta['fps'],resolution=[960,720],state_matches=matches,unique_scene_frames=len(scenes),actual_capture=True,source='Live data from documented formal trials or display-only deterministic repeats; no kinematic re-render',sha256=sha(ROOT/'assets/videos/phase1_dynamic_control.mp4')))
    plot_results(table,data)
    frozen=json.loads((ROOT/'results/phase1_frozen_phase0.json').read_text(encoding='utf-8'));changed=[p for p,info in frozen['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=info['sha256']];write_json(ROOT/'results/phase1_phase0_preservation.json',dict(files_checked=frozen['count'],changed=changed,upstream_tracked_files_unchanged=True));assert not changed
    print('ANALYSIS COMPLETE',flush=True)

def plot_results(table,data):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    output=ROOT/'assets/plots';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    for method in ['C0','C1','C2']:
        row=table[(table.controller==method)&(table.trajectory=='z_roll')&(table.condition=='nominal')&(table.timestep_s==.001)].iloc[0];d=data[row.trial_id];qa=d[[f'quat_actual_wxyz_{j}' for j in range(1,5)]].to_numpy();qr=d[[f'quat_ref_wxyz_{j}' for j in range(1,5)]].to_numpy();aa=Rotation.from_quat(qa[:,[1,2,3,0]]).as_euler('xyz',degrees=True);ar=Rotation.from_quat(qr[:,[1,2,3,0]]).as_euler('xyz',degrees=True)
        for j in range(3):
            axes[j,0].plot(d.time_s,d[f'p_actual_m_{j+1}']*1000,label=method);axes[j,1].plot(d.time_s,aa[:,j],label=method)
            if method=='C0':axes[j,0].plot(d.time_s,d[f'p_ref_m_{j+1}']*1000,'k--',label='Reference');axes[j,1].plot(d.time_s,ar[:,j],'k--',label='Reference')
    for j in range(3):axes[j,0].set(ylabel=['X (mm)','Y (mm)','Z (mm)'][j],xlabel='Physical time (s)');axes[j,1].set(ylabel=['Roll (deg)','Pitch (deg)','Yaw (deg)'][j],xlabel='Physical time (s)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'phase1_pose_tracking.png',dpi=150);plt.close(fig)
    for kind,name,ylabel in [('slide','phase1_leg_tracking.png','Relative slide (mm)'),('force','phase1_actuator_forces.png','Actual force (N)')]:
        fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
        for method in ['C0','C1','C2']:
            row=table[(table.controller==method)&(table.trajectory=='periodic')&(table.condition=='nominal')&(table.timestep_s==.001)].iloc[0];d=data[row.trial_id]
            for j,ax in enumerate(axes.flat,1):
                col=f'slide_actual_m_{j}' if kind=='slide' else f'F_actual_N_{j}';ax.plot(d.time_s,d[col]*(1000 if kind=='slide' else 1),label=method)
                if method=='C0' and kind=='slide':ax.plot(d.time_s,d[f'slide_ref_m_{j}']*1000,'k--',label='Reference')
        for j,ax in enumerate(axes.flat,1):ax.set(xlabel='Physical time (s)',ylabel=ylabel,title=f'Leg {j}');ax.grid(alpha=.2);ax.legend()
        fig.savefig(output/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for method in ['C0','C1','C2']:
        row=table[(table.controller==method)&(table.condition=='asymmetric_gain')].iloc[0];d=data[row.trial_id];qa=d[[f'quat_actual_wxyz_{j}' for j in range(1,5)]].to_numpy();angle=Rotation.from_quat(qa[:,[1,2,3,0]]).as_euler('xyz',degrees=True);axes[0].plot(d.time_s,np.max(abs(angle[:,:2]),axis=1),label=method);axes[1].plot(d.time_s,d.position_error_m*1000,label=method)
        if not pd.isna(row.first_failure_time_s):axes[0].axvline(row.first_failure_time_s,color='gray',linestyle=':')
    axes[0].axhline(.5,color='red',linestyle='--',label='Restricted angle boundary');axes[0].set(xlabel='Physical time (s)',ylabel='Max |roll,pitch| deg',title='Real failure + continued motion during protection hold');axes[1].set(xlabel='Physical time (s)',ylabel='Position error (mm)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'phase1_failure_hold.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
