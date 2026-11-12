"""Reuse the established eight-plot Qt viewer with admitted mass/task models."""
import argparse
import time
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from PySide6 import QtWidgets
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,write_json
from src.rl.stewart_adaptive_env import StewartAdaptiveEnv,HistoryObservation
from src.visualization.enhanced.style import leg_mapping,rgba
from view_phase2c7_policy import Window


class DisplayEnvironment:
    def __init__(self,env,base):
        self.env=env;self.base=base

    def __getattr__(self,name):
        return getattr(self.base,name)

    def step(self,action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class IntegralPolicy:
    def __init__(self,gains):
        self.gains=gains;self.integral=np.zeros(2)

    def predict(self,obs,deterministic=True):
        g=self.gains;error=obs[:2]*.001;velocity=(obs[3:5]-obs[6:8])*.02
        pd=-g['kp']*error-g['kd']*velocity
        proposed=np.clip(np.exp(-g['leak']*.02)*self.integral-g['ki']*.02*error,-g['imax'],g['imax'])
        self.integral=np.where(abs(pd+proposed)<=.05,proposed,np.exp(-g['leak']*.02)*self.integral)
        return np.clip((pd+self.integral)/.05,-1,1),None


class MassWindow(Window):
    def __init__(self,args,root,tasks):
        self.initialized=False
        super().__init__(args,root,tasks)
        self.setWindowTitle('Stewart Phase2C9B | LIVE FROZEN POLICY EVALUATION')
        for i in range(self.task_select.count()):
            task=tasks[self.task_select.itemData(i)]
            self.task_select.setItemText(i,f"{task['id']} | plant {task['mass_kg']} kg | nominal 20 kg | Mx/My {task['torque']}")
        self.seed_select=QtWidgets.QComboBox()
        for seed in SEEDS:self.seed_select.addItem(str(seed),seed)
        self.seed_select.setCurrentIndex(self.seed_select.findData(args.seed))
        seed_row=QtWidgets.QHBoxLayout();seed_row.addWidget(QtWidgets.QLabel('Training seed (select, then Reset new episode):'))
        seed_row.addWidget(self.seed_select);self.layout().insertLayout(2,seed_row)

    def tick(self):
        super().tick()
        self.status.setText(self.status.text()+f' | ACTIVE {self.active_method} seed{self.active_seed} | plant {self.task["mass_kg"]} kg / nominal 20 kg')

    def reset(self):
        if not self.initialized:
            self.control_select.addItems(['Adaptive classic','Single-frame PPO','Five-frame PPO'])
            self.control_select.setCurrentIndex(3 if self.args.method=='single_frame' else 4)
            self.initialized=True
        if self.env:self.env.close()
        if self.renderer:self.renderer.close()
        self.task=self.tasks[self.task_select.currentData()]
        method=('baseline','classical50','adaptive','single_frame','history_5frame')[self.control_select.currentIndex()]
        seed=self.seed_select.currentData() if hasattr(self,'seed_select') else self.args.seed
        selection=read(RUN/'final_selection.json');budget=selection['common_budget'] if method in METHODS else 0
        actual_seed=seed if method in METHODS else 0
        self.active_method=method;self.active_seed=actual_seed
        proof=RUN/'final'/self.task['id']/method/f'seed{actual_seed}'/str(budget)/'metrics.json'
        assert read(proof)['success'],'Only admitted model/task pairs can be displayed'
        self.runs+=1
        base=StewartAdaptiveEnv(self.root/f'run{self.runs}',[self.task],seed=actual_seed,version='v2_control',source='PHASE2C9B LIVE DETERMINISTIC EVALUATION')
        environment=HistoryObservation(base,5) if method=='history_5frame' else base
        self.obs,_=environment.reset(seed=6401);self.env=DisplayEnvironment(environment,base);self.owner=base.owner
        base.context['version']=budget//512 if method in METHODS else -1
        self.mode=0 if method=='baseline' else 2
        if method in METHODS:self.policy=PPO.load(selection['models'][f'{method}/seed{seed}']['path'],device='cpu')
        elif method=='adaptive':self.policy=IntegralPolicy(read(RUN/'adaptive_selection.json')['gains'])
        elif method=='classical50':self.policy=IntegralPolicy(dict(kp=25.,kd=1.,ki=0.,leak=1.,imax=.025))
        self.renderer=mujoco.Renderer(self.owner.model,width=640,height=480)
        self.colors={gid:rgba(r['leg']-1) for r in leg_mapping(self.owner.model) for gid in r['geoms']}
        self.rows=[];self.lines=[];self.finished=self.paused=False;self.proof=proof
        for plot in self.graphs:plot.clear()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS,default='history_5frame')
    parser.add_argument('--seed',choices=SEEDS,type=int,default=42);parser.add_argument('--task',default='Final_m21.5_6')
    parser.add_argument('--validate-steps',type=int,choices=[1000]);args=parser.parse_args();args.model=None
    torch.set_num_threads(1);tasks={t['id']:t for t in cfg('tasks')['final']};root=RUN/'viewer'/str(time.time_ns())
    app=QtWidgets.QApplication([]);w=MassWindow(args,root,tasks);w.show();app.exec()
    result=dict(window_closed=w.closed,physical_steps=w.env.total_physical_steps,method=args.method)
    if args.validate_steps:
        with np.load(next(root.rglob('integration_states.npz'))) as x,np.load(read(w.proof)['folder']+'/integration_states.npz') as y:
            np.testing.assert_array_equal(x['states'],y['states'][:len(x['states'])])
        result['headless_bitwise_equal']=True
    write_json(root/'viewer_validation.json',result)


if __name__=='__main__':main()
