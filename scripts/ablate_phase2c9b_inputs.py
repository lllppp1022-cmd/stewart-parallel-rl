"""Frozen-policy input sensitivity on actual evaluation observations; no physics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from phase2c9b_protocol import ROOT,RUN,SEEDS,cfg,read,write_json
from src.rl.ppo_trace import parameter_hash


def main():
    torch.set_num_threads(1);selection=read(RUN/'final_selection.json');rows=[]
    for seed in SEEDS:
        model=PPO.load(selection['models'][f'history_5frame/seed{seed}']['path'],device='cpu')
        before=parameter_hash(model.policy)
        for task in cfg('tasks')['final']:
            root=RUN/f"final/{task['id']}/history_5frame/seed{seed}/{selection['common_budget']}"
            observations=np.load(root/'observations.npy')[:-1]
            samples=pd.read_csv(root/'training_samples.csv')
            actual=model.predict(observations,deterministic=True)[0]
            np.testing.assert_allclose(actual,samples[['raw_roll','raw_pitch']],rtol=1e-5,atol=1e-6)
            frames=observations.reshape(-1,5,58)
            variants={'actual_order':observations,'repeat_current':np.tile(observations[:,-58:],(1,5)),
                      'reverse_order':frames[:,::-1,:].reshape(-1,290)}
            for name,inputs in variants.items():
                action=model.predict(inputs,deterministic=True)[0]
                rows.append(dict(seed=seed,task_id=task['id'],variant=name,observations=len(inputs),
                    normalized_action_rms_change=float(np.sqrt(((action-actual)**2).mean())),
                    maximum_action_change=float(abs(action-actual).max()),new_physics_steps=0,
                    closed_loop_control_evidence=False,out_of_distribution_intervention=name!='actual_order'))
        assert before==parameter_hash(model.policy)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c9b_history_ablation.csv',index=False)
    write_json(RUN/'input_sensitivity_validation.json',dict(passed=True,frozen_model_parameters=True,
        observations_from_actual_final_trials=True,no_new_physics=True,not_a_closed_loop_causal_effect=True))


if __name__=='__main__':main()
