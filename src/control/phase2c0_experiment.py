"""Optional synthetic plant dynamics and isolated causal diagnostics."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from .phase2b3_experiment import policy,ScaledReference
from .phase2b2_experiment import ResidualOwner,actual_force_map,MeasurementAdapter
from .task_residual_4d import Task4Controller
from .residual_action import ResidualController
from ..dynamics.actuator_lag import ActuatorLag,initialization_description
from ..dynamics.execution_hook import PLANT_STEP
from ..kinematics.jacobian import singularity_diagnostics
from ..model import write_json

class PregeneratedNoise:
    def __init__(self,seed,steps=6400):self.samples=np.random.default_rng(seed).normal(size=(steps,12));self.index=0
    def normal(self,size):
        if size!=12 or self.index>=len(self.samples):raise ValueError('Frozen noise budget exhausted')
        row=self.samples[self.index].copy();self.index+=1;return row

class PhysicalOwner(ResidualOwner):
    def __init__(self,config,*channels):
        measurement=config.get('measurement',{});clean=dict(config,use_residual=False,action=dict(kind='zero',space='leg',frequency=20));clean.pop('measurement',None);super().__init__(clean,*channels)
        self.residual=Task4Controller(self.controller.base,policy(config.get('yaw_policy',dict(kind='zero4'))));self.controller=MeasurementAdapter(self.residual)
        if config.get('reference_scale',1.)!=1:self.active=ScaledReference(self.active,config['reference_scale'])
        self.config=dict(self.config,**{k:config[k] for k in ['measurement','actuator_dynamics','diagnostic'] if k in config});dynamics=config.get('actuator_dynamics',dict(enabled=False,tau_s=0.));initial=self.home.at(0).ff.copy()
        self.lag=ActuatorLag(dynamics.get('tau_s',0.),initial,dynamics.get('enabled',False));self.shadow_base=ActuatorLag(self.lag.tau,initial,self.lag.enabled);self.shadow_c0=ActuatorLag(self.lag.tau,initial,self.lag.enabled);self.last_command=initial.copy();self.last_drive=initial.copy();self.command_history=[];self.shadow_base_drive=initial.copy();self.shadow_c0_drive=initial.copy()
        if config.get('diagnostic'):
            from .phase2c0_actions import make_action
            core=self.residual.base;action=make_action(config['diagnostic'],core,self);wrapper=ResidualController(core,action,allow_privileged=action.privileged);wrapper.name=action.name;self.residual=wrapper;self.controller=MeasurementAdapter(wrapper)
        # Only p/R/v sensor channels; chain q/dq remain ideal, same boundary as B2.
        if measurement:
            self.controller=MeasurementAdapter(self.residual,measurement)
            if any(measurement.get(k,0) for k in ['position_std_m','rotation_std_rad','velocity_std']):self.controller.rng=PregeneratedNoise(measurement.get('seed',42))
        append=self.recorder.append
        def log(row,state):
            gain=self.model.actuator_gainprm[:6,0];force=np.array([row[f'force_{j}'] for j in range(6)])
            for key,value in [('F_command',self.last_command),('F_drive',self.last_drive),('F_drive_previous',self.previous_drive),('shadow_base_drive',self.shadow_base_drive),('shadow_c0_drive',self.shadow_c0_drive)]:
                for j,x in enumerate(value):row[f'{key}_{j}']=float(x)
            for key,value in [('observer_memory_wrench',self.memory_wrench),('known_command_wrench',self.command_wrench),('nominal_drive_wrench',self.drive_wrench)]:
                for j,x in enumerate(value):row[f'{key}_{j}']=float(x)
            row['observer_memory_applicable']=self.memory_applicable
            # Historical f_cmd retains COMMAND semantics; native ctrl is DRIVE here.
            for j,x in enumerate(self.last_command):row[f'f_cmd_{j}']=float(x)
            for key,value in [('gain_only_wrench_error',self.true_map.T@(force-self.last_drive)),('plant_total_wrench_error',self.true_map.T@(force-self.last_command))]:
                for j,x in enumerate(value):row[f'{key}_{j}']=float(x)
            adapter=getattr(self.residual,'adapter',None)
            for j,x in enumerate(adapter.target if adapter else np.zeros(4)):row[f'task4_target_{j}']=float(x)
            for key,drive in [('lagged_residual_wrench',self.shadow_base_drive),('lagged_additional_wrench',self.shadow_c0_drive)]:
                for j,x in enumerate(self.true_map.T@(force-np.clip(gain*drive,-400,400))):row[f'{key}_{j}']=float(x)
            row['actuator_update_count']=self.lag.steps;row['actuator_command_step']=self.n-1;row['measurement_boundary']='PARTIAL OBSERVATION CHANNEL ROBUSTNESS'
            append(row,state)
        self.recorder.append=log
        # Reuse inherited append chain; its physical force fields used instantaneous
        # counterfactuals. Replace them at the final writer with lagged shadows.
        # Actual rows are emitted by Recorder.append; no writer mutation is needed:
        # new lag-shadow fields below are canonical, old W_physical_* are explicitly
        # labelled instantaneous-command counterfactual only in the report.
        write_json(Path(config['folder'])/'actuator_dynamics.json',dict(**dynamics,dt_s=.001,initialization=initialization_description(initial),order='bounded command -> exact-ZOH lag -> first6 native ctrl -> actual gain/native clip -> original mj_step',semantics='SYNTHETIC FIRST-ORDER FORCE RESPONSE; no identified hydraulic model',hidden_from_controller=['tau','internal_drive','actual_force'],known_observer_input='previous bounded controller command, NOT drive/actual force'))
        write_json(Path(config['folder'])/'measurement.json',dict(spec=measurement,label='SYNTHETIC SENSOR NOISE / PARTIAL OBSERVATION CHANNEL ROBUSTNESS',seed=measurement.get('seed',42),rng='numpy.default_rng PCG64, pre-generated6400x12, no global RNG',q_dq='ideal full chain feedback retained'))
        write_json(Path(config['folder'])/'config.json',self.config)
    def apply_actuator_command(self,command):
        self.last_command=np.asarray(command).copy();self.previous_drive=self.lag.state.copy();self.last_drive=self.lag.advance(command)
        core=self.residual.base;self.command_wrench=core.jacobian.T@self.last_command;self.drive_wrench=core.jacobian.T@self.last_drive;self.memory_applicable=core.name=='C3' and core.observer.previous_drive is not None and not self.failure;self.memory_wrench=np.zeros(6)
        if self.memory_applicable:
            o=core.observer;self.memory_wrench=o.previous_drive+o.equilibrium+o.stiffness@core.diagnostics['position_coordinates']+o.damping@o.previous_velocity
        trace=self.residual.trace;fb=trace.get('F_base',command);c0=trace.get('F_C0',command);self.shadow_base_drive=self.shadow_base.advance(fb);self.shadow_c0_drive=self.shadow_c0.advance(c0)
        self.command_history.append(self.last_command.copy());return self.last_drive.copy()
    def snapshot(self):
        s=super().snapshot()
        if hasattr(self,'last_command'):s=replace(s,data=dict(s.data,f_cmd=tuple(self.last_command),actuator_drive=tuple(self.last_drive)))
        return s
    def step(self):
        self.true_map=actual_force_map(self.model,self.data);health=singularity_diagnostics(self.true_map)
        if not self.failure and (health['normalized_condition'] is None or health['normalized_condition']>50):self.hold('actual_jacobian_condition',failed=True)
        PLANT_STEP(self)
