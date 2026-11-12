"""Frozen-control coordination and full-chain references for dynamic attitude tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation

from ..audit_control import scratch_reference_data, static_balance
from ..control.dynamic_reference_feedforward import DynamicReferenceFeedforward
from ..control.large_motion_feedforward_wrapper import LargeMotionFeedforwardWrapper
from ..control.phase2b2_experiment import MeasurementAdapter
from ..disturbance.recording import install_colored_capture
from ..kinematics.inverse_kinematics import inverse_kinematics
from ..kinematics.jacobian import actuator_jacobian, singularity_diagnostics
from ..kinematics.stewart_geometry import StewartGeometry
from ..model import ROOT, write_json
from ..phase1.plant import create_plant
from ..phase1.trajectory import Reference
from ..rl.stewart_residual_env import owner_config
from .execution import LargeOwner
from .geometry import ExpandedGeometryQuery, check_native_joint_limits

RUN=ROOT/"results/runs/phase2c7_20261107_v1"


def reference_key(spec):
    geometry={k:spec[k] for k in ("z_m","roll_deg","pitch_deg")}
    return hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()[:16]


class DynamicReference:
    def __init__(self, spec, build=False):
        self.spec=dict(spec)
        self.name="dynamic_"+reference_key(spec)
        self.rise_s=float(spec["rise_s"])
        self.duration=3+2*self.rise_s
        self.amplitude=float(spec["z_m"])
        self.axis=np.deg2rad([spec["roll_deg"],spec["pitch_deg"],0.])
        if not (0<=self.amplitude<=.05 and abs(self.axis).max()<=np.deg2rad(.35)
                and 4<=self.rise_s<=10 and np.isfinite(list(spec.values())).all()):
            raise ValueError("Reference outside bounded candidate set")
        model=create_plant()
        self.geometry=g=StewartGeometry(model)
        path=RUN/"references"/f"{reference_key(spec)}.npz"
        if not path.exists():
            if not build:
                raise RuntimeError(f"Missing independent reference preflight {path}")
            path.parent.mkdir(parents=True,exist_ok=True)
            query=ExpandedGeometryQuery()
            grid=np.linspace(0,1,101) if self.amplitude or self.axis.any() else np.array([0.,1.])
            qs,derivatives,forces,chains,checks=[],[],[],[],[]
            seed=None
            for value in grid:
                p=g.home_position+[0,0,self.amplitude*value]
                r=Rotation.from_rotvec(self.axis*value).as_matrix()@g.home_rotation
                sol=inverse_kinematics(g,p,r,seed)
                if not sol.success:
                    raise ValueError(f"Full chain IK rejected at {value}")
                seed=sol.chain_q
                data=scratch_reference_data(model,g,sol,p,r)
                native=check_native_joint_limits(model,data.qpos)
                gap=query.evaluate(data.qpos)
                j,_=actuator_jacobian(g,p,r,sol)
                condition=singularity_diagnostics(j)["normalized_condition"]
                check=dict(s=float(value),native_margin=native["minimum_native_margin"],
                           passive_unlimited=len(native["unlimited_hinges"]),
                           stroke_margin_m=sol.minimum_stroke_margin,closure_m=sol.max_position_error,
                           closure_rad=sol.max_rotation_error,clearance_m=gap["minimum_m"],
                           condition=condition)
                checks.append(check)
                if not(native["valid"] and gap["minimum_m"]>=.005 and condition is not None
                       and condition<=50 and sol.max_position_error<1e-4 and sol.max_rotation_error<1e-4):
                    write_json(path.with_suffix(".failed.json"),dict(spec=spec,checks=checks))
                    raise ValueError(f"Reference geometry rejected {check}")
                qs.append(sol.strokes)
                derivatives.append(j@np.r_[0,0,self.amplitude,self.axis])
                forces.append(static_balance(model,g,sol,p,r)[0])
                chains.append(sol.chain_q)
            spline=CubicHermiteSpline(grid,qs,derivatives)
            error=[]
            for i,value in enumerate((grid[:-1]+grid[1:])/2):
                sol=inverse_kinematics(g,g.home_position+[0,0,self.amplitude*value],
                                     Rotation.from_rotvec(self.axis*value).as_matrix()@g.home_rotation,chains[i])
                if not sol.success:
                    raise ValueError("Midpoint full-chain IK failed")
                error.append(float(abs(spline(value)-sol.strokes).max()))
            assert max(error)<1e-7
            dense=spline(np.linspace(0,1,10001))
            assert np.all(dense>=g.stroke_ranges[:,0]) and np.all(dense<=g.stroke_ranges[:,1])
            np.savez_compressed(path,grid=grid,q=qs,dqds=derivatives,ff=forces,chain=chains)
            write_json(path.with_suffix(".json"),dict(spec=spec,checks=checks,
                       midpoint_stroke_error_m=max(error),dense_samples=len(dense),
                       full_chain=True,partial_geometry=True))
        with np.load(path) as cache:
            self.strokes=CubicHermiteSpline(cache["grid"],cache["q"],cache["dqds"])
            self.forces=CubicSpline(cache["grid"],cache["ff"])

    def progress(self,time):
        T=self.rise_s
        def ramp(value):
            u=float(np.clip(value,0,1))
            s=10*u**3-15*u**4+6*u**5
            ds=30*u*u*(1-u)**2/T if 0<value<1 else 0.
            dds=(60*u-180*u*u+120*u**3)/T**2 if 0<value<1 else 0.
            return np.array([s,ds,dds])
        return ramp((time-1)/T)-ramp((time-2-T)/T)

    def scalar(self,time):
        return self.progress(time)*self.amplitude

    def angular_acceleration(self,time):
        return self.axis*self.progress(time)[2]

    def at(self,time):
        s,ds,_=self.progress(time)
        p=self.geometry.home_position+[0,0,self.amplitude*s]
        r=Rotation.from_rotvec(self.axis*s).as_matrix()@self.geometry.home_rotation
        return Reference(self.strokes(s),self.strokes(s,1)*ds,p,r,
                         np.r_[0,0,self.amplitude*ds,self.axis*ds],self.forces(s))


def make_owner(folder, task, record=False):
    reference=DynamicReference(task["reference"])
    q=ExpandedGeometryQuery()
    workspace=json.loads((ROOT/"configs/phase2c3a_large_workspace.json").read_text(encoding="utf-8"))
    parameters=json.loads((ROOT/"results/phase2c3b_feedforward_parameters.json").read_text(encoding="utf-8"))
    cfg=owner_config(folder,dict(id=task["id"],tau_s=.005,Fz_N=0.,
                                on_step=round(task["on"]*1000),off_step=round(task["off"]*1000)),record)
    cfg["disturbance"]["wrench"]=[0,0,0,*task["torque"],0]
    cfg.update(stage="phase2c7",max_sim_seconds=reference.duration,max_wall_seconds=360,
               controller="C3",diagnostic=dict(kind="four_d_lead"))
    o=LargeOwner(cfg,reference,q,workspace)
    ff=DynamicReferenceFeedforward(parameters["kv_N_s_m"],parameters["ka_kg"])
    o.residual=LargeMotionFeedforwardWrapper(o.residual,reference,ff,"Phase2C7_FF-VA-Lead")
    o.controller=MeasurementAdapter(o.residual)
    o.learning_reference=reference
    o.config.update(trajectory=reference.name,dynamic_task=task,ff_parameters=parameters)
    write_json(Path(folder)/"config.json",o.config)
    if record:
        install_colored_capture(o.recorder,o.model)
    return o,reference
