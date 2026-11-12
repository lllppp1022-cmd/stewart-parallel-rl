"""Train three separately seeded, bounded SB3 attitude pilots."""
import _bootstrap
import argparse
import json
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.model import ROOT, sha, write_json
from src.rl.ppo_trace import parameters, parameter_hash, optimizer_steps
from src.rl.stewart_attitude_env import GAMMA, StewartAttitudeResidualEnv
from src.workspace.attitude_study import RUN

SETTINGS = dict(n_steps=512, batch_size=64, n_epochs=10, learning_rate=.0003, gamma=GAMMA,
                gae_lambda=.95**.4, clip_range=.2, ent_coef=0., max_grad_norm=.5, target_kl=.03)


class AttitudePPO(PPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+[k for k in self.__dict__ if k.startswith("_trace_")]

    def train(self):
        env = self._trace_env
        before_actor = parameters(self.policy,"actor").clone()
        before_critic = parameters(self.policy,"critic").clone()
        before_steps = optimizer_steps(self.policy)
        physics = env.total_physical_steps
        print("UPDATING ACTOR / CRITIC", self.num_timesteps, flush=True)
        b = self.rollout_buffer
        idx = len(self._trace_updates)+1
        np.savez_compressed(env.root/f"rollout_{idx:03d}.npz", actions=b.actions, observations=b.observations,
                            advantages=b.advantages, returns=b.returns, values=b.values,
                            rewards=b.rewards, episode_starts=b.episode_starts)
        super().train()
        assert env.total_physical_steps==physics
        row = dict(update=idx, decisions=self.num_timesteps, physical_steps=physics,
                   actor_delta=float(torch.linalg.vector_norm(parameters(self.policy,"actor")-before_actor)),
                   critic_delta=float(torch.linalg.vector_norm(parameters(self.policy,"critic")-before_critic)),
                   adam_steps=optimizer_steps(self.policy)-before_steps, policy_hash=parameter_hash(self.policy),
                   advantage_mean=float(b.advantages.mean()), advantage_std=float(b.advantages.std()))
        for k,v in self.logger.name_to_value.items():
            if k.startswith("train/") and np.isscalar(v):
                row[k]=float(v) if np.isfinite(v) else None
        assert row["actor_delta"]>0 and row["critic_delta"]>0
        assert all(row.get("train/"+key) is not None for key in ("value_loss","policy_gradient_loss","approx_kl"))
        path=self._trace_models/f"update_{idx:03d}.zip"
        self.save(path)
        row["sha256"]=sha(path)
        self._trace_updates.append(row)
        write_json(env.root/"updates.json",self._trace_updates)
        self.logger.dump(self.num_timesteps)
        print("UPDATE DONE", row, flush=True)


class Capture(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.stop = None

    def _on_rollout_start(self):
        self.env.context["version"] = len(self.model._trace_updates)
        print("COLLECTING EXPERIENCE",self.model.num_timesteps,flush=True)

    def _on_step(self):
        row=self.env.records[-1]
        np.testing.assert_allclose([row["clipped_roll"],row["clipped_pitch"]],
                                   np.clip(self.locals["actions"][0],-1,1),atol=1e-7)
        np.testing.assert_array_equal([row["raw_roll"],row["raw_pitch"]],self.locals["actions"][0])
        for key in ("reward","raw_roll","raw_pitch","accepted_roll","accepted_pitch","projection_fraction",
                    "so3_deg","xyz_mm","episode_return","action_box_clipped","reward_orientation",
                    "reward_omega","reward_xyz","reward_action","reward_slew"):
            self.logger.record_mean("attitude/"+key,row[key])
        if row["terminated"] or not np.isfinite(row["reward"]):
            self.stop = "physical_protection_or_nonfinite"
            return False
        return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=[42,123,2026],required=True)
    seed=p.parse_args().seed
    assert json.loads((RUN/"training_gate.json").read_text())["passed"]
    cfg=json.loads((ROOT/"configs/phase2c5_attitude_env.yaml").read_text())
    if seed!=42:
        assert json.loads((RUN/"smoke.json").read_text())["passed"]
    else:
        write_json(ROOT/"configs/phase2c5_ppo.yaml",dict(**SETTINGS,net=[64,64],log_std_init=-1,
                   seeds=[42,123,2026],decisions_per_seed=8192,smoke_included=True,device="cpu"))
    torch.set_num_threads(1)
    env=StewartAttitudeResidualEnv(RUN/"training"/f"seed{seed}",cfg["train"],cfg["method"],
                                  record_episodes=[0,12,25] if seed==42 else [])
    models=ROOT/"models/phase2c5"/f"seed{seed}"
    models.mkdir(parents=True,exist_ok=False)
    model=AttitudePPO("MlpPolicy",env,seed=seed,device="cpu",tensorboard_log=str(env.root/"tensorboard"),
                      policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-1.),**SETTINGS)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    model._trace_env,model._trace_models,model._trace_updates=env,models,[]
    model.save(models/"initial.zip")
    initial=parameter_hash(model.policy)
    callback=Capture(env)
    start=time.perf_counter()
    # Capture the existing sample before SB3 clips it, without re-sampling.
    forward=model.policy.forward
    def traced(obs, deterministic=False):
        action,value,log_prob=forward(obs,deterministic=deterministic)
        env.context.update(raw=action[0].detach().cpu().numpy().copy(),version=len(model._trace_updates))
        return action,value,log_prob
    model.policy.forward=traced
    error=None
    try:
        if seed==42:
            model.learn(512,callback=callback)
            if callback.stop or len(model._trace_updates)!=1:
                raise RuntimeError("Smoke gate failed")
            rng=torch.get_rng_state(),np.random.get_state(),random.getstate()
            loaded=PPO.load(models/"update_001.zip",device="cpu")
            assert parameter_hash(loaded.policy)==parameter_hash(model.policy)
            np.testing.assert_array_equal(loaded.predict(env.obs,deterministic=True)[0],
                                          model.predict(env.obs,deterministic=True)[0])
            torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            write_json(RUN/"smoke.json",dict(passed=True,decisions=512,physical_steps=env.total_physical_steps,
                       update=model._trace_updates[0],reload_equal=True))
            model.learn(7680,callback=callback,reset_num_timesteps=False)
        else:
            model.learn(8192,callback=callback)
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        model.save(models/"final.zip")
        env.close()
        write_json(env.root/"summary.json",dict(seed=seed,decisions=model.num_timesteps,
                   physical_steps=env.total_physical_steps,updates=len(model._trace_updates),
                   adam_steps=sum(x["adam_steps"] for x in model._trace_updates),
                   stop=callback.stop,error=error,wall_s=time.perf_counter()-start,
                   complete_episodes=sum(x["complete"] for x in env.episodes),
                   failures=sum(bool(x["failure"]) for x in env.episodes),
                   initial_hash=initial,final_hash=parameter_hash(model.policy),model_sha=sha(models/"final.zip")))


if __name__=="__main__":
    main()
