"""Bounded strong-load scan, stopped per direction at the first failure."""
import _bootstrap
import argparse
import json

import pandas as pd

from run_phase2c5_attitude_tracking import trial
from src.model import ROOT, write_json
from src.workspace.attitude_study import RUN


def main():
    p=argparse.ArgumentParser()
    p.add_argument("direction",choices=["0","1","2","3","collect"])
    choice=p.parse_args().direction
    if choice=="collect":
        rows=sum([json.loads((RUN/f"strong_{i}.json").read_text()) for i in range(4)],[])
        pd.DataFrame(rows).to_csv(ROOT/"results/phase2c5_torque_boundary.csv",index=False)
        return
    index=int(choice)
    gate=json.loads((RUN/"torque_gate.json").read_text())
    rows=[]
    for amplitude in [1.2, 2.4, 4.8, 8.0, 12.0]:
        torque=[0.,0.]
        torque[index//2]=amplitude*(-1 if index%2 else 1)
        row=trial(RUN/"strong_torque"/f"axis{index}_{amplitude:g}","Home",gate["method"],torque)
        row.update(direction=index, amplitude_Nm=amplitude,
                   recoverable=row["success"] and not row["load_right_censored"] and not row["unload_right_censored"])
        rows.append(row)
        write_json(RUN/f"strong_{index}.json",rows)
        if not row["recoverable"]:
            break


if __name__=="__main__":
    main()
