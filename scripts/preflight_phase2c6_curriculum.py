"""Run a bounded Home baseline and one seeded uniform action sequence per candidate."""
import argparse

import torch

from collect_phase2c6 import admission
from evaluate_phase2c6 import trial
from phase2c6_protocol import RUN, config, read


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("indices",nargs="+")
    a=p.parse_args()
    if not (RUN/"development_gate.json").exists() or "42" not in read(RUN/"development_gate.json")["checks"]:
        raise RuntimeError("Finish and evaluate the fixed-distribution ablation first")
    if a.indices==["collect"]:
        admission()
    else:
        torch.set_num_threads(1)
        tasks=config("curriculum")["candidates"]
        for i in map(int,a.indices):
            for method in ("baseline","random"):
                trial(tasks[i],method,"curriculum_preflight")
