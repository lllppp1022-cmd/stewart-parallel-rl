"""Phase 7 Qt live owner, restricted to already evaluated final tasks/policies."""
import argparse
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from PySide6 import QtWidgets,QtCore
from phase2c7v_common import ROOT,RUN,SEEDS,read,write_json,checkpoint
from view_phase2c7_policy import Window

class FrozenWindow(Window):
    def __init__(self,args,root,tasks):
        self.seed_select=None;self.step_checks=[]
        super().__init__(args,root,tasks)
        self.setWindowTitle('Stewart 7V | FROZEN DETERMINISTIC EVALUATION | REAL DYNAMICS')
        self.seed_select=QtWidgets.QComboBox()
        for s in SEEDS:self.seed_select.addItem(f'Frozen final seed {s} | 32768 decisions',s)
        self.seed_select.setCurrentIndex(self.seed_select.findData(args.seed))
        self.layout().insertWidget(2,self.seed_select)
        b=QtWidgets.QPushButton('Single decision: 20 physical steps (pauses playback)');b.clicked.connect(self.single_step);self.layout().addWidget(b)
        self.layout().addWidget(QtWidgets.QLabel('Task/controller/seed changes take effect on Reset. Drag image to orbit; wheel to zoom.\nNo training. No editable force, load or safety limits. Baseline, classical50 and all final PPO policies were already evaluated on these four tasks.'))
    def reset(self):
        if self.seed_select is not None:
            self.policy=PPO.load(checkpoint(self.seed_select.currentData(),32768),device='cpu');self.policy.policy.set_training_mode(False)
        super().reset()
        if self.runs==1 and self.args.classical:self.mode=1;self.control_select.setCurrentIndex(1)
    def single_step(self):
        before=self.owner.n;self.paused=False;self.tick();self.paused=True
        self.step_checks.append(dict(before=before,after=self.owner.n))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['F_roll','F_pitch','F_mix','F_30'],default='F_roll');p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--classical',action='store_true');p.add_argument('--validate-steps',type=int,choices=[1000]);a=p.parse_args();a.model=checkpoint(a.seed,32768)
    tasks={t['id']:t for t in read(ROOT/'configs/phase2c7_eval.yaml')['final']}
    torch.set_num_threads(1);root=RUN/'viewer'/str(time.time_ns());app=QtWidgets.QApplication([]);w=FrozenWindow(a,root,tasks);w.show()
    if a.validate_steps:
        w.timer.stop();w.paused=True
        def exercise():
            n=w.owner.n;w.tick();assert w.owner.n==n
            w.single_step();assert w.owner.n==n+20
            w.paused=False;w.timer.start(1)
        QtCore.QTimer.singleShot(300,exercise)
    app.exec()
    result=dict(task=a.task,seed=a.seed,classical=a.classical,physical_steps=w.env.total_physical_steps,single_step=w.step_checks,source='LIVE FROZEN DETERMINISTIC EVALUATION',window_closed=w.closed)
    if a.validate_steps:
        method='classical50' if a.classical else f'seed{a.seed}'
        baseline=RUN/'comparison'/a.task/method/'final/episode_0000/integration_states.npz'
        if not baseline.exists():raise FileNotFoundError('Need completed headless comparison first: '+str(baseline))
        with np.load(baseline) as x,np.load(next(root.rglob('integration_states.npz'))) as y:
            np.testing.assert_array_equal(x['states'][:len(y['states'])],y['states'])
        result['bitwise_equal_headless']=True
    write_json(root/'viewer_validation.json',result);print(root,flush=True)
if __name__=='__main__':main()
