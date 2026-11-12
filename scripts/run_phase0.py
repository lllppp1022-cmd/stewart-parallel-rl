import _bootstrap
import argparse,json,time,traceback
import numpy as np,pandas as pd,mujoco,imageio.v2 as imageio
from src.model import ROOT,load_model,write_json,sha,UPSTREAM
from src.audit_control import AuditController,advance
from src.rendering import annotated_frame

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-video',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/'configs/phase0_protocol.json').read_text(encoding='utf-8'));run=ROOT/'results/runs'/f'phase0_{time.time_ns()}';run.mkdir(parents=True)
    write_json(run/'frozen_protocol.json',config);write_json(run/'provenance.json',dict(config_sha256=sha(ROOT/'configs/phase0_protocol.json'),upstream_xml_sha256=sha(UPSTREAM/'stewart_platform.xml'),new_training=False))
    model=load_model();assert model.opt.timestep==config['timestep_s'];renderer=None;writer=None
    video=ROOT/'assets/videos/phase0_simulation.mp4';video.parent.mkdir(parents=True,exist_ok=True);(ROOT/'assets/screenshots').mkdir(parents=True,exist_ok=True)
    if video.exists():raise FileExistsError('Preserve existing video; archive it explicitly before another formal run')
    summaries=[];frame_metadata=[];failures=[]
    try:
        if not args.no_video:
            # Rendering-only buffer size; source MJCF and physical parameters are unchanged.
            model.vis.global_.offwidth=960;model.vis.global_.offheight=720
            renderer=mujoco.Renderer(model,height=720,width=960);writer=imageio.get_writer(video,fps=10,codec='libx264',quality=8,macro_block_size=16)
        for label in config['scenarios']:
            axis='z' if label.startswith('z_') else 'roll' if label.startswith('roll_') else 'pitch' if label.startswith('pitch_') else label;sign=-1 if label.endswith(('down','neg')) else 1
            folder=run/label;folder.mkdir();rows=[];states=[];qactual=[];qvelocity=[];controls=[]
            data=mujoco.MjData(model);mujoco.mj_forward(model,data);controller=AuditController(model,axis,sign,config);n=int(round(config['duration_s']/model.opt.timestep));start=time.perf_counter();reason='completed'
            try:
                for k in range(1,n+1):
                    row=advance(model,data,controller,k);rows.append(row);qactual.append(data.qpos.copy());qvelocity.append(data.qvel.copy());controls.append(data.ctrl.copy())
                    if k%50==0 and writer is not None:
                        frame=annotated_frame(renderer,model,data,row,label);writer.append_data(frame)
                        frame_metadata.append(dict(frame=len(frame_metadata),scenario=label,physics_step=k,time_s=float(data.time),position_error_m=row['position_error_m'],weld_gap_m=row['max_site_gap_m'],qpos_sha256=__import__('hashlib').sha256(data.qpos.tobytes()).hexdigest()))
                        if k==2000:imageio.imwrite(ROOT/'assets/screenshots'/f'{label}.png',frame)
            except Exception as exc:reason='failure:'+type(exc).__name__;failures.append(dict(scenario=label,error=str(exc),traceback=traceback.format_exc()));write_json(folder/'failure.json',failures[-1])
            df=pd.DataFrame(rows);df.to_csv(folder/'physics.csv',index=False);np.savez_compressed(folder/'physical_states.npz',qpos=np.asarray(qactual),qvel=np.asarray(qvelocity),ctrl=np.asarray(controls),time=np.asarray(df['time_s']))
            tail=df[df.time_s>=config['duration_s']-config['held_tail_s']]
            stable=bool(len(rows)==n and len(tail)>=200 and tail.position_error_m.max()<config['position_acceptance_m'] and tail.orientation_error_rad.max()<config['orientation_acceptance_rad'] and df.max_site_gap_m.max()<config['max_weld_gap_m'] and df.max_site_angle_rad.max()<config['max_weld_angle_rad'] and df.warning_count.max()==0)
            summary=dict(scenario=label,steps=len(rows),duration_s=float(data.time),stable_small_motion=stable if axis not in ('zero','pulse') else None,numerical_stability=bool(len(rows)==n and df.warning_count.max()==0),final_position_error_mm=float(df.position_error_m.iloc[-1]*1000),final_orientation_error_deg=float(np.rad2deg(df.orientation_error_rad.iloc[-1])),max_weld_gap_m=float(df.max_site_gap_m.max()),max_weld_angle_rad=float(df.max_site_angle_rad.max()),max_qvel=float(df.max_abs_qvel.max()),max_qacc=float(df.max_abs_qacc.max()),max_force_N=float(max(abs(df[f'actuator_force_N_{j}']).max() for j in range(1,7))),command_clip_steps=int(df.command_clipped.sum()),max_contact_count=int(df.contacts.max()),final_z_m=float(df.actual_position_m_3.iloc[-1]),first_z_m=float(df.actual_position_m_3.iloc[0]),reason=reason,wall_seconds=time.perf_counter()-start,log=str(folder.relative_to(ROOT)/'physics.csv'))
            summaries.append(summary);pd.DataFrame(summaries).to_csv(ROOT/'results/phase0_dynamics_check.csv',index=False);print(json.dumps(summary),flush=True)
    except Exception as exc:
        failures.append(dict(stage='run_setup_or_archive',error=str(exc),traceback=traceback.format_exc()));write_json(run/'setup_failure.json',failures[-1]);raise
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        pd.DataFrame(frame_metadata).to_csv(ROOT/'results/phase0_video_frames.csv',index=False);write_json(run/'failures.json',failures)
    write_json(ROOT/'results/phase0_run_manifest.json',dict(run=str(run.relative_to(ROOT)),protocol_sha256=sha(ROOT/'configs/phase0_protocol.json'),scenarios=len(summaries),physics_steps=sum(s['steps'] for s in summaries),video_frames=len(frame_metadata),video=str(video.relative_to(ROOT)) if writer else None,failures=failures,source_unchanged=sha(UPSTREAM/'stewart_platform.xml')==json.loads((run/'provenance.json').read_text())['upstream_xml_sha256']))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for row in summaries:
        df=pd.read_csv(ROOT/row['log']);axs[0].plot(df.time_s,df.actual_position_m_3,label=row['scenario']);axs[1].plot(df.time_s,df.position_error_m*1000,label=row['scenario']);axs[2].plot(df.time_s,df.max_site_gap_m*1e6,label=row['scenario'])
    for ax,y in zip(axs,['Actual platform Z (m)','Position tracking error (mm)','Max weld translation gap (um)']):ax.set(xlabel='Physical time (s)',ylabel=y);ax.legend(ncol=3);ax.grid(alpha=.2)
    (ROOT/'assets/plots').mkdir(exist_ok=True);fig.savefig(ROOT/'assets/plots/phase0_dynamics.png',dpi=150);plt.close(fig)
    print('FORMAL RUN COMPLETE',run,flush=True)
if __name__=='__main__':main()
