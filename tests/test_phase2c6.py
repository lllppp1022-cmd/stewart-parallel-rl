"""Phase 2C-6 tests exercise the current implementation and archived 1 ms evidence."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from collect_phase2c6 import completed_seeds
from evaluate_phase2c6 import recovery
from phase2c6_history_audit import check_hashes
from phase2c6_protocol import OLD, ROOT, RUN, config, read, sha
from src.rl.attitude_reward_v2 import AttitudeRewardV2Env, v2_components
from src.rl.ppo_trace import parameter_hash
from src.rl.stewart_attitude_env import CAP, DIM, FIELDS, GAMMA, HORIZON, SCALE, STRIDE, failure_penalty
from validate_phase2c6_reward import audit, columns, physical


@pytest.mark.parametrize("column,expected", [(0,1),(1,.2),(2,.1),(3,.001),(4,.000025)])
def test_reward_units(column, expected):
    inputs=[0.,np.zeros(3),0.,np.zeros(2),np.zeros(2)]
    inputs[column]=[.006,np.array([0,.003,0]),.2,np.array([.05,0]),np.array([0,2.5])][column]
    actual=v2_components(*inputs)
    assert actual[column]==pytest.approx(expected)
    assert np.count_nonzero(actual)==1


def test_reward_bound():
    assert v2_components(180.,np.ones(3),1.,np.ones(2),np.ones(2)).sum()==pytest.approx(CAP)
    for n in (0,1,299):
        assert failure_penalty(n)<-sum(CAP*GAMMA**i for i in range(n))


def test_gym_contract_and_reset(tmp_path):
    c=config("ppo")["environment"]
    env=AttitudeRewardV2Env(tmp_path/"env",c["train"],c["method"])
    try:
        check_env(env)
        first,_=env.reset(seed=77)
        sample=env.step(np.array([0.,0.],np.float32))
        assert sample[4]["physical_steps"]==20
        assert not sample[2] and not sample[3] and np.isfinite(sample[1])
        np.testing.assert_array_equal(first,env.reset(seed=77)[0])
        np.testing.assert_array_equal(sample[0],env.step(np.zeros(2))[0])
    finally:
        env.close()


@pytest.mark.parametrize("seed",[42,123,2026])
def test_old_model_reload(seed):
    path=ROOT/f"models/phase2c5/seed{seed}/final.zip"
    model=PPO.load(path,device="cpu")
    original=read(OLD/f"training/seed{seed}/summary.json")
    assert parameter_hash(model.policy)==original["final_hash"] and sha(path)==original["model_sha"]
    assert model.observation_space.shape==(52,) and model.action_space.shape==(2,)


def test_training_contract_and_causal_features():
    c=config("ppo")["environment"]
    assert c==read(ROOT/"configs/phase2c5_attitude_env.yaml")
    assert config("ppo")["original"]==read(ROOT/"configs/phase2c5_ppo.yaml")
    assert (DIM,SCALE,STRIDE,HORIZON,c["physical_dt"])==(52,.05,20,6000,.001)
    assert not any(x in " ".join(FIELDS).lower() for x in ("tau","drive","external","future"))


def test_final_task_isolation():
    env=config("ppo")["environment"]
    c=config("eval")
    signature=lambda t:(tuple(t["torque"]),t["on"],t["off"])
    assert set(map(signature,c["final"])).isdisjoint(map(signature,env["train"]+c["development"]+config("curriculum")["candidates"]))
    assert max(np.linalg.norm(t["torque"]) for t in config("curriculum")["candidates"])<=1.2+1e-9
    assert c["regression"]["torque"]==[2.4,0] and config("curriculum")["excluded_boundary"]==4.8


def test_zero_physics_and_reward_recomputation():
    gate=read(RUN/"training_gate.json")
    assert gate["passed"] and gate["zero_bitwise_steps"]==6000
    directory=RUN/"probe/T7/baseline"
    with np.load(directory/"episode_0000/integration_states.npz") as actual, np.load(OLD/"validation/frozen/integration_states.npz") as before:
        np.testing.assert_array_equal(actual["states"],before["states"])
    a,b,c=[],[],[]
    audit(directory,"zero",a,b,c)
    assert len(c)==3


@pytest.mark.parametrize("method",["xp","xm","yp","ym"])
def test_probe_sign_and_limits(method):
    root=RUN/"probe/T7"/method
    d=physical(root/"episode_0000")
    cmd=columns(d,"F_command")
    add=cmd-columns(d,"F_C0")
    assert abs(add).max()<=5+1e-8
    assert abs(np.diff(np.vstack([np.zeros((1,6)),add]),axis=0)).max()*1000<=20+1e-6
    assert abs(columns(d,"force")).max()<400
    assert abs(d[[f"accepted_{a}_Nm" for a in ("mx","my")]]).max().max()>.005
    s=read(root/"comparison.json")
    assert s["success"] and s["external_max_error"]<1e-9
    assert d.projected.any()


def test_recovery_semantics():
    t=np.arange(1001)*.001
    e=np.zeros(len(t))
    assert recovery(t,e,0.,1.,.01,True)==("not_triggered",None)
    e[:100]=.02
    assert recovery(t,e,0.,1.,.01,True)==("recovered",.1)
    assert recovery(t[:150],e[:150],0.,1.,.01,False)==("right_censored",None)
    e[-501]=.02
    assert recovery(t,e,0.,1.,.01,True)==("right_censored",None)


def test_missing_lock_scope(tmp_path):
    keep=tmp_path/"docs"/"report.docx"
    keep.parent.mkdir()
    keep.write_bytes(b"persistent")
    digest=hashlib.sha256(b"persistent").hexdigest()
    lock="docs/~$report.docx"
    result=check_hashes(tmp_path,{lock:digest,"docs/report.docx":digest})
    assert result["matched"]==1 and result["missing_word_locks"]==[lock]
    with pytest.raises(FileNotFoundError):
        check_hashes(tmp_path,{"docs/other.docx":digest})
    with pytest.raises(AssertionError):
        check_hashes(tmp_path,{"docs/report.docx":"bad"})


def test_same_first_rollout_and_optimizer_update():
    root=RUN/"training/seed42"
    assert read(root/"matched_v1_rollout.json")["samples"]==512
    with np.load(root/"rollout_001.npz") as new, np.load(OLD/"training/seed42/rollout_001.npz") as old:
        for key in ("observations","actions","episode_starts"):
            np.testing.assert_array_equal(new[key],old[key])
        assert not np.array_equal(new["rewards"],old["rewards"])
    updates=read(root/"updates.json")
    assert all(u["decisions"]==i*512 and u["physical_steps"]==i*10240
               and u["actor_delta"]>0 and u["critic_delta"]>0 for i,u in enumerate(updates,1))
    assert read(RUN/"smoke.json")["reload_equal"]


def test_curriculum_admission_exact_physics():
    gate=read(RUN/"curriculum_admission.json")
    results=pd.read_csv(ROOT/"results/phase2c6_curriculum_preflight.csv")
    for t in gate["admitted"]:
        g=results[results.task_id==t["id"]]
        assert len(g)==2 and g.success.all() and set(g.method)=={"baseline","random"}
        assert (g.external_max_error<1e-9).all()
    assert len(results)==2*gate["tested"]


def test_phase6_checkpoint_and_evaluation_integrity():
    summary=read(RUN/"training/seed42/summary.json")
    assert 42 in completed_seeds()
    assert summary["physical_steps"]==summary["decisions"]*20 and summary["updates"]==16
    path=ROOT/"models/phase2c6/reward_v2/seed42/final.zip"
    assert sha(path)==summary["model_sha"]
    rows=pd.read_csv(ROOT/"results/phase2c6_classical_vs_ppo.csv")
    assert rows.policy_unchanged.all() and (rows[rows.method=="V2_42"].model_sha256==sha(path)).all()


def test_curriculum_exact_prefix_and_bounded_integration():
    for seed in (42,123,2026):
        fixed=read(RUN/f"training/seed{seed}/summary.json")
        course=read(RUN/f"curriculum_training/seed{seed}/summary.json")
        assert fixed["final_hash"]==course["initial_hash"]
        assert course["cumulative_decisions"]<=16384 and course["decisions"]<=8192
        assert course["physical_steps"]==course["decisions"]*20
        progress=read(RUN/f"curriculum_training/seed{seed}/curriculum_progression.json")["stages"]
        assert sum(e["actual_decisions"] for e in progress)==course["decisions"]
        assert all(e["passed"] for e in progress[:-1])


def test_independent_and_curriculum_rgb_sources():
    for kind in ("curriculum_training","attitude_evaluation"):
        proof=read(RUN/f"video_{kind}.json")
        assert proof["verified"] and proof["decoded_frames"]>0
        assert all(c["all_frame_state_hashes_and_timestamps_match"] for c in proof["clips"])
    paths=[Path(c["source"]) for c in read(RUN/"video_curriculum_training.json")["clips"]]
    amplitudes=[np.linalg.norm(read(p/"summary.json")["task"]["torque"]) for p in paths]
    assert any(a<.8 for a in amplitudes) and any(np.isclose(a,1.2) for a in amplitudes)


def test_video_sources_and_history():
    proofs=read(RUN/"video_actual_v2_training.json")
    assert proofs["verified"] and all(r["all_frame_state_hashes_and_timestamps_match"] for r in proofs["clips"])
    assert read(RUN/"history_after.json")["matched"]>1000
    assert read(RUN/"qt_equivalence.json")["steps"]==6000


def test_protection_keeps_terminal_reward_and_hold_steps(tmp_path):
    env=AttitudeRewardV2Env(tmp_path/"failure",[config("ppo")["environment"]["train"][0]],"fourDlead")
    try:
        env.reset(seed=4)
        env.owner.hold("test_protection_signal",failed=True)
        _, reward, term, trunc, info=env.step(np.zeros(2))
        assert term and not trunc and info["physical_steps"]==200
        assert reward <= failure_penalty(299) and info["failure"]=="test_protection_signal"
        a,b,c=[],[],[]
        audit(env.root,"test injected protection",a,b,c)
        assert all(not r["complete"] for r in c)
    finally:
        env.close()
