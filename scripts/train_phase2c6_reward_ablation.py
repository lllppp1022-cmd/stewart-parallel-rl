"""Bounded one-factor PPO ablation. Phase 2C-5 trainers and weights stay read-only."""
import argparse
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from phase2c6_protocol import OLD, ROOT, RUN, SEEDS, config, read, sha, write_json
from train_phase2c5_ppo import AttitudePPO, Capture, SETTINGS
from src.rl.attitude_reward_v2 import AttitudeRewardV2Env
from src.rl.ppo_trace import parameter_hash


class TracedPPO(AttitudePPO):
    def train(self):
        b = self.rollout_buffer
        if not self._trace_updates and self._trace_matched:
            with np.load(self._trace_matched) as original:
                for key in ("actions", "observations", "episode_starts"):
                    np.testing.assert_array_equal(getattr(b,key),original[key])
            write_json(self._trace_env.root/"matched_v1_rollout.json",
                       dict(samples=512, actions=True, observations=True, episode_starts=True,
                            source=str(self._trace_matched), sha256=sha(self._trace_matched)))
        self.logger.record("attitude/gae_min",float(b.advantages.min()))
        self.logger.record("attitude/gae_max",float(b.advantages.max()))
        self.logger.record("attitude/gae_mean",float(b.advantages.mean()))
        self.logger.record("attitude/gae_std",float(b.advantages.std()))
        self.logger.record("attitude/target_std",float(b.returns.std()))
        before = len(self._trace_updates)
        super().train()
        event = self._trace_updates[-1]
        assert len(self._trace_updates)==before+1
        self.logger.record("attitude/actor_delta",event["actor_delta"])
        self.logger.record("attitude/critic_delta",event["critic_delta"])
        self.logger.dump(self.num_timesteps)


class Diagnostics(Capture):
    def _on_step(self):
        go = super()._on_step()
        env = self.env
        row = env.records[-1]
        physical_row = env.last_physical_diagnostic_row
        for prefix, keys in (("requested", ("mx","my")), ("accepted", ("mx","my"))):
            for axis in keys:
                key=f"{prefix}_{axis}_Nm"
                self.logger.record_mean("attitude/"+key,physical_row[key])
        omega = np.array([physical_row[f"twist_actual_{i}"]-physical_row[f"twist_ref_{i}"] for i in (3,4,5)])
        self.logger.record_mean("attitude/angular_velocity_rad_s",np.linalg.norm(omega))
        self.logger.record_mean("attitude/reward_total",row["reward"])
        self.logger.record_mean("attitude/box_clip_fraction",row["action_box_clipped"])
        self.logger.record_mean("attitude/protection",float(row["terminated"]))
        if row["terminated"] or row["truncated"]:
            physical=env.finished_diagnostics[-1]
            for name, value in physical.items():
                self.logger.record_mean("episode/"+name,value)
            self.logger.record_mean("episode/return",row["episode_return"])
            self.logger.record_mean("episode/length",row["decision"])
        return go


class FirstEpisodeAtEachLevel:
    """Select RGB capture once per admitted level without consuming any RNG."""
    def __init__(self, env):
        self.env = env
        self.seen = set()

    def __contains__(self, episode):
        level = round(np.linalg.norm(self.env.task["torque"]), 2)
        label = max(.5, level)
        if label in self.seen:
            return False
        self.seen.add(label)
        return True


def learn_curriculum(model, env, callback, base, admitted, expected_extra):
    stages = ((.8,2048),(1.,2048),(1.2,4096))
    assert sum(n for _,n in stages)==expected_extra
    limits={}
    for task in admitted:
        root=RUN/"curriculum_preflight"/task["id"]
        limits[task["id"]]=1.25*max(read(root/m/"comparison.json")["so3_peak_deg"] for m in ("baseline","random"))
    events=[]
    for level, budget in stages:
        env.tasks=base+[t for t in admitted if np.linalg.norm(t["torque"])<=level+1e-9]
        start_count=len(env.episodes)
        start_decision=model.num_timesteps
        model.learn(budget,callback=callback,reset_num_timesteps=False)
        pairs=list(zip(env.episodes,env.finished_diagnostics))[start_count:]
        complete=[(e,d) for e,d in pairs if e["complete"]]
        passed=(model.num_timesteps-start_decision==budget and not callback.stop and bool(complete)
                and any(abs(np.linalg.norm(e["task"]["torque"])-level)<1e-9 for e,_ in complete)
                and all(d["peak_so3_error"]<=limits.get(e["task"]["id"],.04) for e,d in complete)
                and all(not e["failure"] for e,_ in pairs))
        events.append(dict(level=level,budget=budget,actual_decisions=model.num_timesteps-start_decision,
                           allowed=[t["id"] for t in env.tasks],complete=len(complete),passed=bool(passed),
                           all_episode_peaks=[dict(task=e["task"]["id"],**d) for e,d in complete]))
        write_json(env.root/"curriculum_progression.json",dict(rule="No protection, a completed episode at the current level, and SO3 peak <=125% of paired preflight maxima (0.04 deg for base tasks).",stages=events))
        if not passed:
            callback.stop=callback.stop or "curriculum_progression_gate_failed"
            break


def train(seed, curriculum=False):
    gate = read(RUN/"training_gate.json")
    if not gate["passed"]:
        raise RuntimeError("Reward and physics gate failed")
    for key, digest in gate["configs"].items():
        assert sha(ROOT/f"configs/phase2c6_{key}.yaml")==digest
    c = config("ppo")
    for k,v in SETTINGS.items():
        assert c["original"][k]==v
    if seed!=42 or curriculum:
        development = read(RUN/"development_gate.json")
        if curriculum and len(development["passing_seeds"]) < 2:
            raise RuntimeError("Curriculum requires two passing fixed-task seeds")
        if not curriculum and 42 not in development["passing_seeds"]:
            raise RuntimeError("Seed42 did not pass the development gate")
    torch.set_num_threads(1)
    group = "curriculum_v2" if curriculum else "reward_v2"
    tasks = c["environment"]["train"]
    if curriculum:
        admission = read(RUN/"curriculum_admission.json")
        if not admission["passed"]:
            raise RuntimeError("Joint curriculum preflight did not pass")
    env=AttitudeRewardV2Env(RUN/("curriculum_training" if curriculum else "training")/f"seed{seed}",
                            tasks,c["environment"]["method"], record_episodes=[0,12,25] if seed==42 else [])
    models=ROOT/"models/phase2c6"/group/f"seed{seed}"
    models.mkdir(parents=True,exist_ok=False)
    if curriculum:
        if seed==42:
            env.record_episodes=FirstEpisodeAtEachLevel(env)
        prefix = ROOT/"models/phase2c6/reward_v2"/f"seed{seed}"/"final.zip"
        model=TracedPPO.load(prefix,env=env,device="cpu",seed=seed)
        assert model.num_timesteps==8192
        model.tensorboard_log=str(env.root/"tensorboard")
    else:
        model=TracedPPO("MlpPolicy",env,seed=seed,device="cpu",tensorboard_log=str(env.root/"tensorboard"),
                        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-1.),**SETTINGS)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    model._trace_env,model._trace_models,model._trace_updates=env,models,[]
    model._trace_matched=None if curriculum else OLD/"training"/f"seed{seed}"/"rollout_001.npz"
    initial=parameter_hash(model.policy)
    if not curriculum:
        assert initial==read(OLD/"training"/f"seed{seed}"/"summary.json")["initial_hash"]
    model.save(models/"initial.zip")
    callback=Diagnostics(env)
    start=time.perf_counter()
    forward=model.policy.forward

    def traced(obs, deterministic=False):
        action,value,log_prob=forward(obs,deterministic=deterministic)
        env.context.update(raw=action[0].detach().cpu().numpy().copy(),
                           version=len(model._trace_updates)+(16 if curriculum else 0))
        return action,value,log_prob

    model.policy.forward=traced
    error=None
    try:
        if seed==42 and not curriculum:
            model.learn(c["smoke"],callback=callback)
            if callback.stop or len(model._trace_updates)!=1:
                raise RuntimeError("Smoke gate failed")
            rng=torch.get_rng_state(),np.random.get_state(),random.getstate()
            loaded=PPO.load(models/"update_001.zip",device="cpu")
            assert parameter_hash(loaded.policy)==parameter_hash(model.policy)
            np.testing.assert_array_equal(loaded.predict(env.obs,deterministic=True)[0],
                                          model.predict(env.obs,deterministic=True)[0])
            torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            write_json(RUN/"smoke.json",dict(passed=True,decisions=c["smoke"],physical_steps=env.total_physical_steps,
                       update=model._trace_updates[0],reload_equal=True))
            model.learn(c["ablation_decisions"]-c["smoke"],callback=callback,reset_num_timesteps=False)
        elif curriculum:
            learn_curriculum(model,env,callback,tasks,admission["admitted"],c["curriculum_extra_decisions"])
        else:
            model.learn(c["ablation_decisions"],callback=callback)
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        model.save(models/"final.zip")
        env.close()
        prefix_decisions=8192 if curriculum else 0
        write_json(env.root/"summary.json",dict(seed=seed,group=group,decisions=model.num_timesteps-prefix_decisions,
                   cumulative_decisions=model.num_timesteps,physical_steps=env.total_physical_steps,
                   updates=len(model._trace_updates),adam_steps=sum(x["adam_steps"] for x in model._trace_updates),
                   stop=callback.stop,error=error,wall_s=time.perf_counter()-start,
                   complete_episodes=sum(x["complete"] for x in env.episodes),
                   failures=sum(bool(x["failure"]) for x in env.episodes),
                   initial_hash=initial,final_hash=parameter_hash(model.policy),model_sha=sha(models/"final.zip")))


def main(curriculum=False):
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    train(p.parse_args().seed,curriculum)


if __name__=="__main__":
    main()
