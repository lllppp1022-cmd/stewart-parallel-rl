import _bootstrap
import time,json
from src.model import ROOT,write_json
from src.visualization.owner import PhysicsOwner
if __name__=='__main__':
    folder=ROOT/'results/runs'/f'phase1_5_1_legacy_baseline_{time.time_ns()}'
    cfg=dict(folder=str(folder),controller='C0',condition='nominal',record=False,pace=False,max_sim_seconds=32,max_wall_seconds=300,events=json.loads((ROOT/'configs/phase1_5_1_demo_events.json').read_text(encoding='utf-8')))
    PhysicsOwner(cfg).run();write_json(ROOT/'results/phase1_5_1_baseline_pointer.json',dict(folder=str(folder)));print(folder,flush=True)
