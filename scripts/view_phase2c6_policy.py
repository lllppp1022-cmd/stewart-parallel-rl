"""Open the original live Qt display using independent Phase 2C-6 paths."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from phase2c6_protocol import ROOT, RUN, config, write_json
from view_phase2c5_policy import QtWidgets, Window


def main():
    cfg=config("eval")
    tasks={t["id"]:t for t in cfg["development"]+cfg["final"]+[cfg["regression"]]}
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=tasks,default="E_mix")
    p.add_argument("--model",type=Path,default=ROOT/"models/phase2c6/reward_v2/seed42/final.zip")
    p.add_argument("--validate-steps",type=int,choices=[1000,6000])
    a=p.parse_args()
    if a.validate_steps and a.task!="E_mix":
        p.error("The GUI verification pair uses E_mix")
    if not a.model.is_file():
        p.error(f"Missing checkpoint: {a.model}")
    torch.set_num_threads(1)
    root=RUN/"viewer"/str(time.time_ns())
    app=QtWidgets.QApplication([])
    w=Window(a,root,tasks,"fourDlead")
    w.show()
    app.exec()
    result=dict(steps=w.env.total_physical_steps,source="Qt live deterministic evaluation",task=w.task["id"])
    if a.validate_steps:
        expected=RUN/"dev/E_mix/V2_42/episode_0000/integration_states.npz"
        observed=next(root.rglob("integration_states.npz"))
        with np.load(expected) as before, np.load(observed) as after:
            np.testing.assert_array_equal(before["states"][:a.validate_steps],after["states"])
        assert (root/"viewer.png").exists()
        result["bitwise_equal"]=True
        write_json(RUN/"qt_equivalence.json",result)
    write_json(root/"viewer_summary.json",result)
    print(root)


if __name__=="__main__":
    main()
