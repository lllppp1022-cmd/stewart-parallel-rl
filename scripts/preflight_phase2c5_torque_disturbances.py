import _bootstrap
import argparse
import json

import pandas as pd

from run_phase2c5_attitude_tracking import trial
from src.model import ROOT, write_json
from src.workspace.attitude_study import RUN

LEVELS = [.15, .5, 1.2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices=["0", "1", "2", "3", "collect", "C4", "fourDlead"])
    args = parser.parse_args()
    tracking = json.loads((RUN/"tracking_gate.json").read_text())
    if not tracking["passed"]:
        raise RuntimeError("Active tracking gate failed")
    method = tracking["selected"]
    if args.worker in ("C4", "fourDlead"):
        method = args.worker
        rows = [dict(trial(RUN/"torque"/method/f"axis{i}", "Home", method,
                          [(.5 if i==0 else -.5 if i==1 else 0.),
                           (.5 if i==2 else -.5 if i==3 else 0.)]), level=2, direction=i)
                for i in range(4)]
        write_json(RUN/f"torque_{method}.json", rows)
        return
    if args.worker != "collect":
        index = int(args.worker)
        cases = []
        for level, amplitude in enumerate(LEVELS, 1):
            torque = [0., 0.]
            torque[index//2] = amplitude * (-1 if index % 2 else 1)
            cases.append(dict(trial(RUN/"torque"/f"axis{index}_level{level}", "Home", method, torque),
                              level=level, direction=index))
            write_json(RUN/f"torque_{index}.json", cases)
            if not cases[-1]["success"] or cases[-1]["load_right_censored"] or cases[-1]["unload_right_censored"]:
                break
        return
    rows = sum([json.loads((RUN/f"torque_{i}.json").read_text()) for i in range(4)], [])
    complete = [level for level in range(1,4) if sum(r["level"] == level and r["success"]
                and not r["load_right_censored"] and not r["unload_right_censored"] for r in rows) == 4]
    for candidate in ("C4", "fourDlead"):
        rows.extend(json.loads((RUN/f"torque_{candidate}.json").read_text()))
    scores = {}
    for candidate in ("C0", "C4", "fourDlead"):
        eligible = [r for r in rows if r["method"]==candidate and r["level"]==2]
        if len(eligible)==4 and all(r["success"] and not r["load_right_censored"]
                                    and not r["unload_right_censored"] for r in eligible):
            scores[candidate] = sum(r["load_iae_deg_s"]+r["unload_iae_deg_s"] for r in eligible)
    method = min(scores, key=scores.get) if scores else method
    if complete:
        for task, torque in [("Hold", (.10, -.10)), ("R3", (.10, .10)), ("Home", (.24, -.21))]:
            rows.append(dict(trial(RUN/"torque"/f"combined_{task}", task, method, torque, 2.2, 3.7),
                             level=None, direction="combined"))
    pd.DataFrame(rows).to_csv(ROOT/"results/phase2c5_torque_preflight.csv", index=False)
    write_json(RUN/"torque_gate.json", dict(passed=bool(complete), four_direction_levels=complete,
               amplitudes_Nm=LEVELS, method=method, matched_level2_scores=scores, duration_s=1.5,
               boundary_bracketed=False, maximum_means_largest_tested_recoverable_only=True))


if __name__ == "__main__":
    main()
