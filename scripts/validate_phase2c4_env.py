import _bootstrap,argparse,json,gc,copy
from pathlib import Path
import numpy as np,pandas as pd
from stable_baselines3.common.env_checker import check_env
from src.model import ROOT,write_json,sha
from src.rl.stewart_large_motion_env import RUN,FIXED,StewartLargeMotionResidualEnv,large_owner
from src.rl.large_motion_rewards import costs,failure_penalty,CONFIG
from src.rl.large_motion_observations import FIELDS,LargeObservation

def safe(folder):
    d=pd.read_csv(Path(folder)/'physics.csv.gz');a=d[[f'F_command_{j}' for j in range(6)]].to_numpy()-d[[f'F_C0_{j}' for j in range(6)]].to_numpy();rate=np.diff(np.vstack([np.zeros((1,6)),a]),axis=0)/.001
    return dict(passed=bool(d.failure.isna().all() and abs(a).max()<=5+1e-8 and abs(rate).max()<=20+1e-6 and d.warning_count.max()==0 and not d.native_force_saturated.any()),force_peak_N=float(abs(d[[f'force_{j}' for j in range(6)]]).to_numpy().max()),add_peak_N=float(abs(a).max()),add_slew_N_s=float(abs(rate).max()),physical_steps=len(d))

def zero():
    e=StewartLargeMotionResidualEnv(RUN/'validation/zero_env',mode='Baseline Comparison');e.reset(seed=42);b,r=large_owner(RUN/'validation/zero_frozen',FIXED);step=e.owner.step;checks=0;first=[]
    def paired():
        nonlocal checks
        step();b.step();a=e.owner
        for k in ['qpos','qvel','act','ctrl','mocap_pos','mocap_quat','qacc_warmstart','userdata']:np.testing.assert_array_equal(getattr(a.data,k),getattr(b.data,k))
        np.testing.assert_array_equal(a.data.actuator_force,b.data.actuator_force);np.testing.assert_array_equal(a.lag.state,b.lag.state);np.testing.assert_array_equal(a.residual.injection.original.previous,b.residual.injection.original.previous)
        for k,v in vars(a.residual.base.observer).items():
            if isinstance(v,(np.ndarray,float,int)) or v is None:np.testing.assert_equal(v,getattr(b.residual.base.observer,k))
        np.testing.assert_array_equal(a.ref.q,b.ref.q);assert a.failure==b.failure and a.profile.description()==b.profile.description()
        np.testing.assert_array_equal(a.recorder.states[-1],b.recorder.states[-1]);checks+=1
        if len(first)<50:first.append(a.recorder.states[-1].copy())
    e.owner.step=paired
    for _ in range(300):o,re,term,trunc,info=e.step([0.])
    assert trunc and not term and e.episode_summaries[-1]['complete'];b.recorder.close(dict(complete=True,zero_reference=True));e.reset(seed=42);e.step([0.]);np.testing.assert_array_equal(first,e.owner.recorder.states);e.close()
    write_json(RUN/'validation/zero.json',dict(passed=True,paired_physical_steps=checks,bitwise_complete_integration=True,controller_observer_governor_lag_and_external_equal=True,reset_first50_bitwise_equal=True,complete=True,zero_reward_return=e.episode_summaries[0]['return_']))

def actions():
    rows=[]
    for name in ['positive','negative','uniform','alternating']:
        e=StewartLargeMotionResidualEnv(RUN/f'validation/{name}',mode='Action Safety Validation');e.reset(seed=7401);rng=np.random.default_rng(7401)
        for i in range(300):
            a=1. if name=='positive' else -1. if name=='negative' else float(rng.uniform(-1,1)) if name=='uniform' else (-1.)**i
            _,_,t,u,_=e.step([a])
            if t:raise RuntimeError('Action precheck failed')
        e.close();folder=Path(e.episode_summaries[0]['folder']);d=pd.read_csv(folder/'physics.csv.gz');recomputed=-float(d[[f'reward_cost_{k}' for k in ['z','xy','rotation','action','slew']]].to_numpy().sum()*.001/.05)
        assert abs(recomputed-e.episode_summaries[0]['return_'])<1e-7
        rows.append(dict(pattern=name,return_=recomputed,z_rmse_mm=float(np.sqrt(np.mean((d.p_2-d.p_ref_2)**2))*1000),xy_peak_mm=float(np.hypot(d.p_0-d.p_ref_0,d.p_1-d.p_ref_1).max()*1000),accepted_residual_peak_N=float(abs(d.rl_accepted_Fz_N).max()),observation_clip_fraction=e.episode_summaries[0]['observation_clip_fraction'],**safe(folder)));del d,e;gc.collect()
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c4_reward_audit.csv',index=False);write_json(RUN/'validation/actions.json',dict(passed=all(x['passed'] and x['observation_clip_fraction']<.001 for x in rows),cases=rows));assert all(x['passed'] for x in rows)

def interface():
    e=StewartLargeMotionResidualEnv(RUN/'validation/check_env',mode='Interface Validation');check_env(e,warn=True);e.close();ev=json.loads((ROOT/'configs/phase2c4_eval.yaml').read_text(encoding='utf-8'));rows=[]
    for task in ev['final_tasks']:
        o,r=large_owner(RUN/'validation/eval_preflight'/task['id'],task)
        for _ in range(round(r.duration*1000)):
            o.step()
            if o.failure:raise RuntimeError('Final task joint preflight failed')
        o.recorder.close(dict(complete=True,source='Safety preflight only; no parameter selection'));rows.append(dict(task_id=task['id'],**safe(o.recorder.folder)));del o;gc.collect()
    # Analytic ranking intervention uses identical state, no future/load truth.
    sample=pd.read_csv(RUN/'validation/eval_preflight'/ev['final_tasks'][0]['id']/'physics.csv.gz').iloc[6500].to_dict();base=sum(costs(sample,0)[0].values());better=copy.deepcopy(sample);better['p_2']=better['p_ref_2'];worse=copy.deepcopy(sample);worse['p_0']=worse['p_ref_0']+.0005;assert sum(costs(better,0)[0].values())<=base and sum(costs(worse,0)[0].values())>base
    for n in [0,1,50,300]:assert failure_penalty(n)<-sum(100*.99**j for j in range(n))
    write_json(RUN/'validation/interface.json',dict(passed=all(x['passed'] for x in rows),gym_check_env=True,final_safety_only=rows,feature_dimensions=sum(n for _,n in FIELDS),reward_monotonic_same_other_states=True,discounted_failure_ranking_bound=True))

def collect():
    parts={s:json.loads((RUN/f'validation/{s}.json').read_text(encoding='utf-8')) for s in ['zero','actions','interface','viewer']};assert all(x['passed'] for x in parts.values());write_json(ROOT/'results/phase2c4_env_validation.json',dict(passed=True,parts=parts,shared_budget_N=5,shared_slew_N_s=20,native_N=400,physical_dt_s=.001,decision_steps=50,horizon_steps=15000,feature_dim=53,history=1,hidden_inputs_excluded=['tau','load','drive','actual force','future state'],no_test_fitting=True));print('ENV GATE PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['zero','actions','interface','collect']);a=p.parse_args();globals()[a.mode]()
