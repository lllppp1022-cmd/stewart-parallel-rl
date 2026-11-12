"""Freeze Phase 2C-6 datasets and archive provenance before running them."""
import _bootstrap
import json

from src.model import ROOT, sha, write_json
from src.rl.attitude_reward_v2 import NAMES, SCALES, WEIGHTS
from src.workspace.attitude_study import RUN as OLD

RUN = ROOT / "results/runs/phase2c6_20261107_v1"
SEEDS = (42, 123, 2026)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def config(stem):
    return read(ROOT / "configs" / f"phase2c6_{stem}.yaml")


def prepare():
    RUN.mkdir(parents=True, exist_ok=False)
    hashes = read(OLD / "history_freeze.json")
    manifest = ROOT / "results/phase2c5_artifact_manifest.json"
    hashes.update({k: v["sha256"] for k, v in read(manifest)["files"].items()})
    hashes[str(manifest.relative_to(ROOT))] = sha(manifest)
    write_json(RUN / "history_before.json", hashes)
    original = read(ROOT / "configs/phase2c5_attitude_env.yaml")
    ppo = read(ROOT / "configs/phase2c5_ppo.yaml")
    old_eval = read(ROOT / "configs/phase2c5_eval.yaml")
    final = [
        dict(id="F_roll", torque=[-.43, 0.], on=2.32, off=3.86),
        dict(id="F_pitch", torque=[0., .93], on=1.72, off=3.36),
        dict(id="F_mix", torque=[.72, -.78], on=2.26, off=3.92),
    ]
    curriculum = [
        dict(id=f"C{level}_{i}", torque=[sign*level if axis==0 else 0., sign*level if axis==1 else 0.],
             on=on, off=off)
        for level in (.8, 1., 1.2)
        for i, (axis, sign, on, off) in enumerate([(0, 1, 1.76, 3.24), (0, -1, 2.22, 3.94),
                                                   (1, 1, 2.08, 3.42), (1, -1, 1.88, 3.56)])
    ]
    curriculum += [dict(id="Cmix1", torque=[.72,.96], on=1.84, off=3.50),
                   dict(id="Cmix2", torque=[-.96,.72], on=2.18, off=3.82)]
    settings = {
        "reward_v2": dict(names=NAMES, scales=SCALES, weights=WEIGHTS, cap=100,
                          units=["deg","rad/s","mm","Nm","Nm/s"],
                          comparator="V1 plus one slew-only offline candidate; V2 fixed candidate",
                          rationale="Use millidegree-scale observed errors and retain both positive regularizers."),
        "ppo": dict(original=ppo, environment=original, smoke=512, ablation_decisions=8192,
                    curriculum_extra_decisions=8192, ceiling_per_seed=16384),
        "eval": dict(development=old_eval["tasks"][:3], final=final, regression=old_eval["tasks"][-1],
                     gains=old_eval["classical_feedback_gains"], recovery_bands_deg=[.1,.01], dwell_s=.5,
                     dev_rule="Same window IAE improves >=5% in at least two of three paired tasks, and decreases in mean, relative to V1. All succeed, mean XYZ and max native slew each <=110% of V1. Deterministic requests have measurable accepted effect.",
                     curriculum_rule="At least two fixed V2 seeds pass the same development rule before curriculum.",
                     promotion_rule="At least two seeds beat classical in both IAEs on >=2 final cases, with no protection and no >10% XYZ or slew increase on any final case."),
        "curriculum": dict(candidates=curriculum, admission="Only exact combinations passing baseline and bounded random preflight",
                           level3_regression_only=2.4, excluded_boundary=4.8,
                           continuation="V2 8192 checkpoint + 8192 curriculum decisions, reported separately"),
    }
    for key, value in settings.items():
        write_json(ROOT/"configs"/f"phase2c6_{key}.yaml",value)


if __name__ == "__main__":
    prepare()
