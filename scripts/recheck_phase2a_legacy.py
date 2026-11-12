"""Correct the documented one-step legacy timing assumption, retaining all v1 data."""
import _bootstrap
import argparse,json,time
from pathlib import Path
import pandas as pd
from src.model import ROOT,write_json,sha
from run_phase2a_disturbance import run_trial

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    table=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');old=table[(table.condition=='external_load')&(table.timestep_s==.001)];spec=[]
    for r in old.itertuples():
        df=pd.read_csv(ROOT/r.physics_log);active=df[df.external_ctrl_3!=0];start=int(active.step.iloc[0])-1;stop=int(active.step.iloc[-1]);spec.append(dict(controller=r.controller,start_step=start,stop_step=stop,source=str(ROOT/r.physics_log)))
    frozen=dict(parent=str(a.parent),reason='v1 guessed ideal t_next boundary k=1999; actual archived floating-time policy first nonzero result row=2001, so input k=2000. Retain v1 as offset diagnostic, rerun only three legacy reproductions.',trials=spec,controllers_unchanged=True,thresholds_unchanged=True,protocol_frozen_before_recheck=True)
    write_json(a.output/'formal_protocol_frozen.json',frozen)
    for s in spec:run_trial(a.output,'legacy_local_z_roll_exact',s['controller'],'z_roll',[0,0,-10,.5,0,0],stage='legacy_recheck',frame='site-local',start_step=s['start_step'],duration_steps=s['stop_step']-s['start_step'],parent_run=str(a.parent),formal_protocol_sha256=sha(a.output/'formal_protocol_frozen.json'))
    write_json(a.parent/'legacy_recheck_pointer.json',dict(root=str(a.output)))
