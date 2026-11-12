"""Exploratory transfer check with whole reference groups held out."""
import numpy as np
import pandas as pd

from audit_phase2c9_dynamics import ROOT, RUN, PLANTS, REFS
from phase2c8_protocol import read, write_json


def windows(observations, frames, indices):
    offsets = np.arange(1 - frames, 1)
    return observations[indices[:, None] + offsets].reshape(len(indices), -1)


def evaluate(learn, test):
    x, y = learn
    z, labels = test
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .01)
    x = (x - center) / scale
    z = (z - center) / scale
    w = np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * len(x) * .01, x.T @ (2 * y - 1))
    return float(np.mean((z @ w >= 0) == labels))


def main():
    folder = RUN / "preflight_v2"
    metrics = [read(folder / f"{i:02d}/metrics.json") for i in range(len(PLANTS) * len(REFS))]
    rows = []
    eligible = []
    for p, (kind, value) in enumerate(PLANTS[1:], 1):
        case = [(j * len(PLANTS), j * len(PLANTS) + p) for j in range(len(REFS))]
        safe = all(metrics[i]["success"] for pair in case for i in pair)
        effects = [max(metrics[b][k] / metrics[a][k] - 1 for k in
                       ("load_iae_deg_s", "unload_iae_deg_s", "so3_peak_deg")) for a, b in case]
        if not safe:
            rows.append(dict(kind=kind, value=value, safe=False, max_degradation=max(effects)))
            continue
        scores = {}
        for frames in (1, 5, 10):
            datasets = []
            for a, b in case:
                data = [np.load(folder / f"{i:02d}/observations.npy") for i in (a, b)]
                indices = np.arange(10, min(map(len, data)), 5)
                x = np.concatenate([windows(d, frames, indices) for d in data])
                datasets.append((x, np.repeat([0, 1], len(indices))))
            scores[frames] = [evaluate(datasets[i], datasets[1-i]) for i in (0, 1)]
            for i, score in enumerate(scores[frames]):
                rows.append(dict(kind=kind, value=value, safe=True, max_degradation=max(effects),
                                 held_out_reference=REFS[1-i][0], frames=frames, balanced_accuracy=score))
        for frames in (5, 10):
            gain = np.array(scores[frames]) - scores[1]
            if min(effects) >= .05 and min(gain) >= .05 and min(scores[frames]) >= .65:
                eligible.append(dict(kind=kind, value=value, frames=frames, gains=gain.tolist()))
    pd.DataFrame(rows).to_csv(ROOT / "results/phase2c9_history_information.csv", index=False)
    write_json(RUN / "history_gate.json", dict(eligible=eligible, two_reference_transfer_only=True,
        criterion="both references safe; at least one attitude metric degrades >=5% in each; "
                  "history balanced accuracy >=65% and gain >=5 percentage points in both held-out reference folds",
        label_used_only_offline=True, no_online_controller_or_training_conclusion=True))
    print(eligible)


if __name__ == "__main__":
    main()
