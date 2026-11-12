"""Bounded Phase2A pilot then frozen paired C0/C1/C2 comparisons. No RL."""
import _bootstrap
import argparse,json,time,sys,platform,subprocess
from pathlib import Path
import numpy as np,mujoco
from src.model import ROOT,MODEL,UPSTREAM,sha,write_json
from src.disturbance.executor import DisturbanceOwner
from src.disturbance.metrics import compute
from src.phase1.plant import create_plant

def load_protocol():return json.loads((ROOT/'configs/phase2a_disturbance_protocol.json').read_text(encoding='utf-8'))
def run_trial(root,name,controller,trajectory,wrench,stage='formal',duration_steps=2000,frame='world',start_step=2000,record=False,**extra):
    p=load_protocol();folder=Path(root)/stage/name/controller
    cfg=dict(folder=str(folder),controller=controller,condition='nominal',record=record,pace=False,native_viewer=False,max_sim_seconds=p['duration_s'],max_wall_seconds=p['maximum_wall_s_per_trial'],exit_on_failure=True,trajectory=trajectory,stage=stage,task=name,disturbance=dict(start_step=start_step,stop_step=start_step+duration_steps,wrench=list(wrench),frame=frame,kind='zero' if not any(wrench) else 'composite' if sum(x!=0 for x in wrench)>1 else 'step'),model_sha256=sha(MODEL),controller_sha256=sha(ROOT/'src/phase1/controllers.py'),phase1_protocol_sha256=sha(ROOT/'configs/phase1_protocol.json'),**extra)
    owner=DisturbanceOwner(cfg)
    if record:
        from src.disturbance.recording import install_colored_capture
        install_colored_capture(owner.recorder,owner.model)
    owner.run();metrics=compute(folder,p);write_json(folder/'metrics.json',metrics)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));summary.update(success=metrics['success'],task_completed=metrics['completed'],phase2a_metrics='metrics.json');write_json(folder/'summary.json',summary)
    print(stage,name,controller,'PASS' if metrics['success'] else 'FAIL',metrics['failure_reason'],f"peak {metrics['peak_position_mm']} mm / {metrics['peak_orientation_deg']} deg",flush=True)
    return metrics

def audit(root):
    m=create_plant();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    rows=[dict(id=i,name=m.actuator(i).name,transmission_type=int(m.actuator_trntype[i]),target_ids=m.actuator_trnid[i].tolist(),gear=m.actuator_gear[i].tolist(),control_range=m.actuator_ctrlrange[i].tolist(),force_range=m.actuator_forcerange[i].tolist(),gain=m.actuator_gainprm[i].tolist(),units='N' if i<9 else 'N m',frame='joint scalar' if i<6 else 'upper_center site-local') for i in range(m.nu)]
    write_json(root/'interface_audit.json',dict(model=str(MODEL),source_files={str(f.relative_to(ROOT)):sha(f) for f in UPSTREAM.rglob('*.xml')},upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),nq=m.nq,nv=m.nv,nu=m.nu,actuators=rows,dt=m.opt.timestep,python=sys.version,platform=platform.platform(),mujoco=mujoco.__version__,nominal_is_independent=True,control_period_s=.001,external_channels_separate=True,legacy_rule='external_command(external_load,t_next): local [0,0,-10,.5,0,0] for 2<=t_next<4; new profiles explicitly use input intervals',safety_file='configs/phase1_protocol.json',controller_file_sha256=sha(ROOT/'src/phase1/controllers.py')))

def pilot(root):
    p=load_protocol();selection={};rows=[]
    for axis,values,index in [('force',p['pilot_force_N'],2),('roll_positive',p['pilot_roll_positive_Nm'],3),('roll_negative',p['pilot_roll_negative_Nm'],3)]:
        selected=None
        for value in values:
            w=[0.]*6;w[index]=value;r=run_trial(root,f'{axis}_{value:g}_0p5','C0','home',w,'pilot',500);rows.append(r)
            if not r['success']:break
            selected=value
        if selected is not None:
            for duration in (1000,2000):
                w=[0.]*6;w[index]=selected;r=run_trial(root,f'{axis}_{selected:g}_{duration}','C0','home',w,'pilot',duration);rows.append(r)
                if not r['success']:selected=None;break
        selection[axis]=selected
    write_json(root/'pilot_selection.json',dict(selection=selection,trials=rows,policy='Stop increasing a failed direction; duration failures block that direction; no retrospective safety changes'))
    return selection

def formal(root,selection,record=True):
    p=load_protocol();tasks=[]
    for trajectory in ('home','roll','pitch','z_roll'):tasks.append(dict(name='zero_'+trajectory,trajectory=trajectory,wrench=[0.]*6))
    f=selection['force'];plus=selection['roll_positive'];minus=selection['roll_negative']
    if f is not None:tasks.append(dict(name='A_home_force',trajectory='home',wrench=[0,0,f,0,0,0]))
    for sign,value in [('positive',plus),('negative',minus)]:
        if value is not None:tasks.append(dict(name='B_home_roll_'+sign,trajectory='home',wrench=[0,0,0,value,0,0]))
    if all(selection[k] is not None for k in selection):
        for trajectory in ('roll','pitch'):tasks.append(dict(name='C_'+trajectory+'_force',trajectory=trajectory,wrench=[0,0,max(f,-5),0,0,0]))
        tasks.append(dict(name='C_z_roll_composite',trajectory='z_roll',wrench=[0,0,max(f,-5),min(plus,.1),0,0]))
    # Archived Phase1 rows show first nonzero step=2001, last=4000 (float t_next).
    # Preserve integer intervals rather than guessing from ideal decimal t_next.
    if f==-10 and plus==.5:tasks.append(dict(name='legacy_local_z_roll',trajectory='z_roll',wrench=[0,0,-10,.5,0,0],frame='site-local',start_step=2000))
    frozen=dict(protocol=p,selection=selection,tasks=tasks,controllers=p['controllers'],frozen_before_formal=True,source='Pilot selected only; no formal results read to choose thresholds');write_json(root/'formal_protocol_frozen.json',frozen)
    print('FORMAL FROZEN',len(tasks)*3,'trials',flush=True)
    assert len(tasks)*3<=p['maximum_formal_trials']
    for task in tasks:
        for method in p['controllers']:
            spec=dict(task);name=spec.pop('name');capture=record and (name=='A_home_force' or method=='C1' and name in ('B_home_roll_positive','C_z_roll_composite'))
            run_trial(root,name,method,stage='formal',record=capture,formal_protocol_sha256=sha(root/'formal_protocol_frozen.json'),**spec)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--stage',choices=['pilot','formal','all'],default='all');a.add_argument('--run-root',type=Path);a.add_argument('--no-record',action='store_true');args=a.parse_args()
    root=args.run_root or ROOT/'results/runs'/f'phase2a_{time.time_ns()}'
    root.mkdir(parents=True,exist_ok=args.stage=='formal');print('RUN_ROOT='+str(root),flush=True);audit(root)
    write_json(root/'pilot_protocol_frozen.json',load_protocol()) if args.stage!='formal' else None
    if args.stage in ('pilot','all'):selection=pilot(root)
    else:selection=json.loads((root/'pilot_selection.json').read_text(encoding='utf-8'))['selection']
    if args.stage in ('formal','all'):formal(root,selection,not args.no_record)
    write_json(ROOT/'results/phase2a_latest_run.json',dict(root=str(root),stage=args.stage))
if __name__=='__main__':main()
