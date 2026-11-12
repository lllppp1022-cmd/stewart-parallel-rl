"""Measured safe full physics/controller stack, no GUI; no GPU speed inference."""
import _bootstrap
import time,argparse
from pathlib import Path
from src.model import ROOT,write_json
from src.control.phase2c0_experiment import PhysicalOwner
from run_phase2c0_robustness import config,task

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);r=p.parse_args().output_root;r.mkdir(parents=True,exist_ok=False);records=[]
    for stride in [1,20]:
        cfg=config(r/f'journal_stride{stride}',task('home'),tau=.002,enabled=True,stage='benchmark');cfg['max_sim_seconds']=2.;start=time.perf_counter();x=PhysicalOwner(cfg);constructed=time.perf_counter()-start;append=x.recorder.append;log_time=[0.];counts=[0]
        def measured(row,state):
            if int(row['step'])%stride:return
            tick=time.perf_counter();append(row,state);log_time[0]+=time.perf_counter()-tick;counts[0]+=1
        x.recorder.append=measured;tick=time.perf_counter();x.run();elapsed=time.perf_counter()-tick
        records.append(dict(journal_stride=stride,physical_steps=x.n,simulation_seconds=float(x.data.time),construction_seconds=constructed,run_and_archive_seconds=elapsed,physical_steps_per_wall_second=x.n/elapsed,append_pipeline_seconds=log_time[0],recorded_rows=counts[0],failure=x.failure,includes='Original1kHz C4, lag2ms, actual Jacobian+closed-chain safety, original mj_step+forward, optional full journal; no RGB/GUI',scope='2-second Home benchmark, no claim of arbitrary task throughput'))
    write_json(ROOT/'results/phase2c0_simulation_benchmark.json',dict(records=records,total_benchmark_physical_steps=sum(x['physical_steps'] for x in records),no_GPU_inference=True));print(records,flush=True)
