"""58-dimensional causal, 50 Hz PPO over the frozen 1 kHz physical owner."""
import gc

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .stewart_attitude_env import MomentPort, StewartAttitudeResidualEnv, STRIDE, SCALE, failure_penalty
from .attitude_reward_v2 import NAMES
from ..model import write_json
from ..phase1.plant import platform_state
from ..workspace.dynamic_attitude_study import make_owner

DIM=58
FIELDS=("SO3_error[3]","angular_velocity[3]","reference_angular_velocity[3]",
        "position_error[3]","linear_velocity[3]","relative_slides[6]","slide_velocity[6]",
        "known_command[6]","causal_observer[6]","base_add[6]","previous_request[2]",
        "previous_accepted[2]","reference_rotation[3]","reference_z","reference_vz",
        "reference_az","reference_angular_acceleration[3]")


class DynamicMomentPort(MomentPort):
    def __getattr__(self,name):
        return getattr(self.original,name)


def dynamic_cost(row,accepted,slew):
    e=np.array([row[f"p_{j}"]-row[f"p_ref_{j}"] for j in range(3)])*1000
    omega=np.array([row[f"twist_actual_{j}"]-row[f"twist_ref_{j}"] for j in (3,4,5)])
    costs=np.array([(row["orientation_error_deg"]/.006)**2,
                    .2*np.sum((omega/.003)**2),.1*np.sum((e/[.2,.2,.05])**2),
                    .001*np.sum((accepted/.05)**2),.000025*np.sum((slew/2.5)**2)])
    factor=min(1.,100/max(costs.sum(),1e-30))
    if not np.isfinite(costs).all():
        raise ValueError("Nonfinite physical reward")
    return costs*factor,factor


class StewartDynamicAttitudeEnv(StewartAttitudeResidualEnv):
    def __init__(self,root,tasks,record_episodes=(),source="ACTUAL DYNAMIC PPO TRAINING"):
        super().__init__(root,tasks,"Phase2C7_FF-VA-Lead",record_episodes,source)
        self.observation_space=gym.spaces.Box(-10,10,(DIM,),np.float32)
        self.finished_diagnostics=[]
        self.horizon=0

    def observe(self):
        o=self.owner
        p,r,v=platform_state(o.model,o.data)
        ref=self.reference.at(o.data.time)
        estimate=o.residual.base.diagnostics.get("equivalent_disturbance",np.zeros(6))
        z,dz,ddz=self.reference.scalar(o.data.time)
        frame=np.r_[Rotation.from_matrix(r@ref.r.T).as_rotvec()/.001,
                    v[3:]/.02,ref.twist[3:]/.02,(p-ref.p)/[.0002,.0002,.00005],v[:3]/.01,
                    (o.data.qpos[o.qadr]-self.reference.at(0).q)/.06,o.data.qvel[o.dof]/.03,
                    (o.last_command-o.home.at(0).ff)/5,np.asarray(estimate)/[20,20,20,.8,.8,.8],
                    self.port.base_add/5,self.port.request/SCALE,self.port.accepted/SCALE,
                    Rotation.from_matrix(ref.r@o.geometry.home_rotation.T).as_rotvec()/.01,
                    z/.05,dz/.03,ddz/.03,self.reference.angular_acceleration(o.data.time)/.02]
        if frame.shape!=(DIM,) or not np.isfinite(frame).all():
            raise ValueError("Invalid causal dynamic observation")
        self.clip_count+=int((abs(frame)>10).sum())
        self.obs_count+=DIM
        return np.clip(frame,-10,10).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        self.finish("reset_incomplete")
        self.episode+=1
        self.task=self.tasks[int(self.np_random.integers(len(self.tasks)))]
        self.owner,self.reference=make_owner(self.root/f"episode_{self.episode:04d}",self.task,
                                              self.episode in self.record_episodes)
        self.horizon=round(self.reference.duration*1000)
        self.owner.episode=self.episode
        injection=self.owner.residual.injection
        self.port=DynamicMomentPort(injection.original,self.owner.residual.base)
        injection.original=self.port
        self.decision=self.clip_count=self.obs_count=0
        self.episode_return=0.
        self.previous_accepted=np.zeros(2)
        self.context={}
        self.done=False
        self.last_row=None
        self.physical_diagnostics=dict(peak_so3_error=0.,load_window_IAE=0.,unload_window_IAE=0.,cap_steps=0)
        append=self.owner.recorder.append

        def log(row,state):
            accepted=self.port.accepted if not self.owner.failure else np.zeros(2)
            slew=(accepted-self.previous_accepted)*1000
            self.previous_accepted=accepted.copy()
            self.cost,factor=dynamic_cost(row,accepted,slew)
            for name,value in zip(NAMES,self.cost):
                row[f"dyn_cost_{name}"]=value
            row["reward_cap_factor"]=factor
            for j,axis in enumerate(("mx","my")):
                row[f"requested_{axis}_Nm"]=self.port.request[j]
                row[f"accepted_{axis}_Nm"]=accepted[j]
                row[f"slew_{axis}_Nm_s"]=slew[j]
            for j in range(6):
                row[f"rl_mapped_leg_{j}_N"]=self.port.leg[j]
                row[f"same_state_base_add_{j}_N"]=self.port.base_add[j]
                row[f"ff_requested_leg_{j}_N"]=self.owner.residual.injection.audit.get("ff_leg",np.zeros(6))[j]
            row.update(decision=self.decision,global_decision=self.global_decisions,
                       policy_version=self.context.get("version",-1),rl_source=self.source,
                       projected=self.port.projected,stage=self.task["stage"],task_id=self.task["id"])
            append(row,state)
            self.last_row=row
            self.last_physical_diagnostic_row=row
            self.physical_diagnostics["peak_so3_error"]=max(self.physical_diagnostics["peak_so3_error"],row["orientation_error_deg"])
            self.physical_diagnostics["cap_steps"]+=int(factor<1)
            for name,edge in (("load","on"),("unload","off")):
                if round(self.task[edge]*1000)<=row["input_step"]<round(self.task[edge]*1000)+1000:
                    self.physical_diagnostics[name+"_window_IAE"]+=row["orientation_error_deg"]*.001

        self.owner.recorder.append=log
        self.obs=self.observe()
        return self.obs.copy(),dict(task_id=self.task["id"])

    def step(self,action):
        if self.owner is None or self.done:
            raise RuntimeError("Reset before step")
        a=np.asarray(action,float)
        if a.shape!=(2,) or not np.isfinite(a).all():
            raise ValueError("Expected two finite actions")
        clipped=np.clip(a,-1,1)
        self.port.request=SCALE*clipped
        self.decision+=1
        self.global_decisions+=1
        start=self.owner.n
        total=np.zeros(5)
        accepted,projected=[],[]
        for _ in range(min(STRIDE,self.horizon-start)):
            self.owner.step()
            total+=self.cost
            accepted.append(self.port.accepted.copy())
            projected.append(self.port.projected)
            if self.owner.failure:
                break
        failure=bool(self.owner.failure)
        while failure and self.owner.data.time<self.owner.protect_end-1e-10:
            self.owner.step()
            total+=self.cost
        n=self.owner.n-start
        self.total_physical_steps+=n
        penalty=failure_penalty(int(np.ceil(max(0,self.horizon-self.owner.n)/20))) if failure else 0.
        reward=float(-total.sum()/20+penalty)
        self.done=failure or self.owner.n>=self.horizon
        self.episode_return+=reward
        self.obs=self.observe()
        raw=self.context.get("raw",a)
        row=dict(episode=self.episode,decision=self.decision,global_decision=self.global_decisions,
                 physical_steps=n,reward=reward,terminal_penalty=penalty,
                 raw_roll=raw[0],raw_pitch=raw[1],clipped_roll=clipped[0],clipped_pitch=clipped[1],
                 accepted_roll=float(np.mean(accepted,axis=0)[0]),accepted_pitch=float(np.mean(accepted,axis=0)[1]),
                 projection_fraction=float(np.mean(projected)),policy_version=self.context.get("version",-1),
                 terminated=failure,truncated=self.done and not failure,task_id=self.task["id"],
                 stage=self.task["stage"],so3_deg=self.last_row["orientation_error_deg"],
                 xyz_mm=self.last_row["position_error_mm"],episode_return=self.episode_return,
                 action_box_clipped=bool(np.any(abs(raw)>1)))
        for j,axis in enumerate(("roll","pitch")):
            row[f"actor_mean_{axis}"]=float(self.context.get("mean",a)[j])
            row[f"actor_std_{axis}"]=float(self.context.get("std",np.zeros(2))[j])
        for name,value in zip(NAMES,total):
            row[f"reward_{name}"]=-value/20
        self.records.append(row)
        if self.writer is None:
            import csv
            self.writer=csv.DictWriter(self.stream,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        info=dict(failure=self.owner.failure,physical_steps=n)
        if self.done:
            info["episode"]=dict(r=self.episode_return,l=self.decision)
            self.finish("protected_failure" if failure else "time_limit")
        return self.obs.copy(),reward,failure,self.done and not failure,info

    def finish(self,reason):
        if self.owner is None:
            return
        o=self.owner
        row=dict(episode=self.episode,task=self.task,reason=reason,physical_steps=o.n,
                 complete=o.n==self.horizon and not o.failure,failure=o.failure,
                 folder=str(o.recorder.folder),episode_return=self.episode_return,
                 clip_fraction=self.clip_count/max(1,self.obs_count),source=self.source,
                 duration_s=self.reference.duration,diagnostics=self.physical_diagnostics.copy())
        o.recorder.close(row)
        self.episodes.append(row)
        self.finished_diagnostics.append(self.physical_diagnostics.copy())
        write_json(self.root/"episodes.json",self.episodes)
        self.owner=None
        del o
        gc.collect()
