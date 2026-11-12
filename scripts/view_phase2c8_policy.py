"""Reuse the established Qt live renderer with frozen Phase8 model admission."""
import argparse
import time
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from PySide6 import QtWidgets,QtCore
from phase2c8_protocol import ROOT,RUN,SEEDS,cfg,read,write_json,sha
from src.rl.phase2c8_env import Phase2C8Env
from src.visualization.enhanced.style import leg_mapping,rgba
from view_phase2c7_policy import Window

class Phase8Window(Window):
    def __init__(self,args,root,tasks,selection):
        self.model_select=None;self.selection=selection;self.selected_key=f'{args.method}/seed{args.seed}';self.step_checks=[]
        super().__init__(args,root,tasks)
        self.setWindowTitle('Stewart Phase 2C-8 | FROZEN POLICY | LIVE DYNAMIC EVALUATION')
        self.model_select=QtWidgets.QComboBox()
        for key,item in selection['models'].items():self.model_select.addItem(f'{key} | {selection["budget"]} decisions',key)
        self.model_select.setCurrentIndex(self.model_select.findData(self.selected_key));self.layout().insertWidget(2,self.model_select)
        b=QtWidgets.QPushButton('Single decision / 20 physical steps');b.clicked.connect(self.single);self.layout().addWidget(b)
        self.layout().addWidget(QtWidgets.QLabel('Select model/task/controller then Reset. Final models only; unchanged load/force/safety limits. Camera: drag and wheel.'))
    def reset(self):
        if self.model_select:self.selected_key=self.model_select.currentData()
        item=self.selection['models'][self.selected_key];assert sha(item['path'])==item['sha'];self.policy=PPO.load(item['path'],device='cpu')
        method,seed_text=self.selected_key.split('/');task=self.tasks[self.task_select.currentData()];mode=self.control_select.currentIndex()
        trial_method=method if mode==2 else 'baseline' if mode==0 else 'classical50';trial_seed=int(seed_text[4:]) if mode==2 else 0;n=self.selection['budget'] if mode==2 else 0
        proof=RUN/'final'/task['id']/trial_method/f'seed{trial_seed}'/str(n)/'metrics.json'
        if not proof.exists() or not read(proof)['success']:raise ValueError('Selected policy/task has not passed frozen physical evaluation')
        if self.env:self.env.close()
        if self.renderer:self.renderer.close()
        self.runs+=1;self.task=task;self.mode=mode
        self.env=Phase2C8Env(self.root/f'run{self.runs}',[task],seed=6401,version=method if mode==2 else 'v2_control',source='Qt LIVE FROZEN PHASE2C8 EVALUATION')
        self.obs,_=self.env.reset(seed=6401);self.env.context['version']=self.selection['budget']//512 if mode==2 else -1;self.owner=self.env.owner
        self.renderer=mujoco.Renderer(self.owner.model,width=640,height=480);self.colors={gid:rgba(r['leg']-1) for r in leg_mapping(self.owner.model) for gid in r['geoms']}
        self.rows=[];self.lines=[];self.finished=self.paused=False
        for plot in self.graphs:plot.clear()
        self.proof=proof
    def single(self):
        n=self.owner.n;self.paused=False;self.tick();self.paused=True;self.step_checks.append([n,self.owner.n])

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['v2_control','v3','v3_return_norm'],default='v3');p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--task',default='T8_mix');p.add_argument('--validate-steps',type=int,choices=[1000]);a=p.parse_args();torch.set_num_threads(1)
    selection=read(RUN/'final_selection.json');a.model=selection['models'][f'{a.method}/seed{a.seed}']['path'];tasks={t['id']:t for t in cfg('evaluation')['final']};assert a.task in tasks
    root=RUN/'viewer'/str(time.time_ns());app=QtWidgets.QApplication([]);w=Phase8Window(a,root,tasks,selection);w.show()
    if a.validate_steps:
        w.timer.stop();w.paused=True
        def check():
            n=w.owner.n;w.tick();assert w.owner.n==n;w.single();assert w.owner.n==n+20;w.paused=False;w.timer.start(1)
        QtCore.QTimer.singleShot(200,check)
    app.exec();result=dict(physical_steps=w.env.total_physical_steps,window_closed=w.closed,single_step=w.step_checks,source='LIVE DETERMINISTIC EVALUATION')
    if a.validate_steps:
        from pathlib import Path
        with np.load(Path(read(w.proof)['folder'])/'integration_states.npz') as x,np.load(next(root.rglob('integration_states.npz'))) as y:np.testing.assert_array_equal(x['states'][:len(y['states'])],y['states'])
        result['bitwise_equal']=True
    write_json(root/'viewer_validation.json',result);print(root,flush=True)
if __name__=='__main__':main()
