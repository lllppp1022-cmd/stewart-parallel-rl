"""Deterministic prefix replay for identical-state action interventions."""
import argparse
import numpy as np
import pandas as pd
from phase2c9b_protocol import RUN,cfg,write_json
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv


def main(mass):
    task=next(t for t in cfg('tasks')['training'] if t['mass_kg']==mass and t['id'].endswith('_6'))
    actions=[(0,0),(.25,0),(-.25,0),(0,.25),(0,-.25),(.25,-.25)]
    rows=[];prefix=None;zero=None
    for i,action in enumerate(actions):
        env=StewartAdaptiveEnv(RUN/'authority'/str(mass)/str(i),[task],seed=42,version='v2_control')
        try:
            env.reset(seed=42)
            for k in range(220):env.step(np.zeros(2))
            before=env.owner.data.qpos.copy()
            if prefix is None:prefix=before
            np.testing.assert_array_equal(prefix,before)
            for _ in range(5):
                _,_,term,_,_=env.step(action);assert not term
            out=np.r_[env.owner.data.qvel,env.owner.data.actuator_force[:6]]
            if zero is None:zero=out.copy()
            rows.append(dict(action=action,max_force_difference_N=float(abs(out[-6:]-zero[-6:]).max()),
                             max_velocity_difference=float(abs(out[:-6]-zero[:-6]).max())))
        finally:env.close()
    effect=any(r['max_force_difference_N']>1e-7 and r['max_velocity_difference']>1e-9 for r in rows[1:])
    write_json(RUN/'authority'/str(mass)/'audit.json',dict(mass_kg=mass,same_prefix=True,measurable_effect=effect,results=rows))
    print(mass,effect,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mass',type=float,required=True);main(p.parse_args().mass)
