"""Validate the distributed model, six terminal actors and actual physics."""
import _bootstrap
import json
import contextlib
import time
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from src.model import ROOT,load_model,sha
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation
from phase2c9b_protocol import cfg,read,RUN


def main():
    torch.set_num_threads(1)
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert (m.nq,m.nv,m.nu,m.neq)==(43,42,12,6)
    selection=read(RUN/'final_selection.json')
    for key,item in selection['models'].items():
        assert sha(item['path'])==item['sha']
        policy=PPO.load(item['path'],device='cpu')
        dim=290 if key.startswith('history_5frame') else 58
        assert policy.observation_space.shape==(dim,)
        action=policy.predict(np.zeros(dim,dtype=np.float32),deterministic=True)[0]
        assert action.shape==(2,) and np.isfinite(action).all()
    task=next(t for t in cfg('tasks')['final'] if t['id']=='Final_m21.5_6')
    runtime=ROOT/'release/runtime'/str(time.time_ns())
    runtime.mkdir(parents=True,exist_ok=False)
    with contextlib.nullcontext(str(runtime)) as tmp:
        base=StewartAdaptiveEnv(Path(tmp)/'evaluation',[task],seed=6401,version='v2_control')
        env=HistoryObservation(base,5)
        model=PPO.load(selection['models']['history_5frame/seed42']['path'],device='cpu')
        try:
            obs,_=env.reset(seed=6401)
            for _ in range(50):
                action=model.predict(obs,deterministic=True)[0]
                obs,reward,terminated,truncated,_=env.step(action)
                assert not terminated and not truncated and np.isfinite(reward)
            assert base.total_physical_steps==1000
            assert abs(base.owner.model.body_mass[base.owner.model.body('upper').id]-21.5)<1e-10
            assert base.owner.nominal.body_mass[base.owner.nominal.body('upper').id]==20
        finally:env.close()
    results=pd.read_csv(ROOT/'results/phase2c9b_generalization.csv')
    assert len(results)==72 and results.success.all()
    means=results.groupby('method')[['load_iae_deg_s','unload_iae_deg_s','so3_peak_deg']].mean()
    saved=pd.read_csv(ROOT/'results/phase2c9b_method_metrics.csv').set_index('method')
    np.testing.assert_allclose(means,saved.loc[means.index,means.columns],atol=1e-12,rtol=0)
    print(json.dumps(dict(passed=True,models=6,actual_physical_steps=1000,final_trials=72,metric_recomputation=True)))


if __name__=='__main__':main()
