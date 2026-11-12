"""Single-owner Qt live dynamics with eight reference/actual plots."""
import argparse
import time
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
from PySide6 import QtCore,QtGui,QtWidgets
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from phase2c7_protocol import ROOT,RUN,read,write_json
from view_phase2c5_policy import CameraImage
from render_phase2c7_videos import series,COLORS
from src.rl.stewart_dynamic_attitude_env import StewartDynamicAttitudeEnv
from src.phase1.plant import platform_state
from src.visualization.enhanced.style import leg_mapping,rgba


class Window(QtWidgets.QWidget):
    def __init__(self,args,root,tasks):
        super().__init__();self.args=args;self.root=root;self.tasks=tasks
        self.policy=PPO.load(args.model,device='cpu') if args.model else None
        if self.policy:assert self.policy.observation_space.shape==(58,)
        self.env=self.renderer=None;self.runs=0;self.closed=False
        self.setWindowTitle('Stewart Phase 2C-7 | LIVE DYNAMIC EVALUATION')
        layout=QtWidgets.QVBoxLayout(self);self.status=QtWidgets.QLabel();layout.addWidget(self.status)
        select=QtWidgets.QHBoxLayout();layout.addLayout(select)
        self.task_select=QtWidgets.QComboBox()
        for key,t in tasks.items():self.task_select.addItem(f'{key}: Z {t["reference"]["z_m"]*1000:g} mm | Mx/My {t["torque"]}',key)
        self.task_select.setCurrentIndex(self.task_select.findData(args.task))
        self.control_select=QtWidgets.QComboBox();self.control_select.addItems(['Frozen combined baseline','Classical 50 Hz'])
        if self.policy:self.control_select.addItem('PPO deterministic');self.control_select.setCurrentIndex(2)
        select.addWidget(self.task_select);select.addWidget(self.control_select)
        select.addWidget(QtWidgets.QLabel('Select validated task/controller, then Reset'))
        body=QtWidgets.QHBoxLayout();layout.addLayout(body)
        self.image=CameraImage();self.image.setMinimumSize(650,520);body.addWidget(self.image,2)
        plots=QtWidgets.QVBoxLayout();body.addLayout(plots,2)
        self.graphs=[pg.PlotWidget() for _ in range(8)]
        for plot in self.graphs:plots.addWidget(plot)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        for label,fn in [('Pause/resume',self.pause),('Reset new episode',self.reset),('Close and archive',self.close)]:
            b=QtWidgets.QPushButton(label);b.clicked.connect(fn);controls.addWidget(b)
        self.reset();self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(20)
        self.resize(1500,1050)

    def pause(self):self.paused=not self.paused

    def reset(self):
        if self.env:self.env.close()
        if self.renderer:self.renderer.close()
        self.runs+=1;self.task=self.tasks[self.task_select.currentData()];self.mode=self.control_select.currentIndex()
        self.env=StewartDynamicAttitudeEnv(self.root/f'run{self.runs}',[self.task],source='Qt LIVE DETERMINISTIC EVALUATION')
        self.obs,_=self.env.reset(seed=6401);self.owner=self.env.owner
        self.renderer=mujoco.Renderer(self.owner.model,width=640,height=480)
        self.colors={gid:rgba(row['leg']-1) for row in leg_mapping(self.owner.model) for gid in row['geoms']}
        self.rows=[];self.lines=[];self.finished=self.paused=False
        for plot in self.graphs:plot.clear()

    def tick(self):
        if not self.paused and not self.finished:
            action=np.zeros(2)
            if self.mode==2:action=self.policy.predict(self.obs,deterministic=True)[0]
            elif self.mode==1:
                _,r,v=platform_state(self.owner.model,self.owner.data);ref=self.env.reference.at(self.owner.data.time)
                action=(-25*Rotation.from_matrix(r@ref.r.T).as_rotvec()[:2]-(v[3:5]-ref.twist[3:5]))/.05
            self.obs,_,term,trunc,_=self.env.step(action);self.rows.append(self.env.last_row.copy())
            table=pd.DataFrame(self.rows)
            for k,(title,values) in enumerate(series(table)):
                if len(self.lines)<=k:
                    self.graphs[k].setTitle(title)
                    self.lines.append([self.graphs[k].plot(pen=COLORS[j%6]) for j in range(values.shape[1])])
                for j,line in enumerate(self.lines[k]):line.setData(table.time_s.to_numpy(),values[:,j])
            self.finished=term or trunc or bool(self.args.validate_steps and self.owner.n>=self.args.validate_steps)
        self.renderer.update_scene(self.owner.data,camera=self.image.cam)
        for geom in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid in self.colors:geom.rgba[:]=self.colors[geom.objid];geom.matid=-1
        rgb=self.renderer.render().copy();h,w,_=rgb.shape
        image=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
        self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        status=self.owner.failure or ('FINISHED: reset for a new physical episode' if self.finished else 'PAUSED' if self.paused else 'RUNNING')
        self.status.setText(f'DYNAMIC ACTUATOR SIMULATION | {self.task["id"]} | t={self.owner.data.time:.3f}s | {status} | PARTIAL GEOMETRY COVERAGE')
        if self.finished and self.args.validate_steps:self.grab().save(str(self.root/'viewer.png'));self.close()

    def closeEvent(self,event):
        if not self.closed:
            self.timer.stop();self.renderer.close();self.env.close();self.closed=True
        event.accept()


def main():
    tasks={t['id']:t for t in read(RUN/'admission.json')['admitted']}
    p=argparse.ArgumentParser();p.add_argument('--task',choices=tasks,default='C2');p.add_argument('--model',type=Path)
    p.add_argument('--validate-steps',type=int,choices=[1000,15000]);a=p.parse_args()
    torch.set_num_threads(1);root=RUN/'viewer'/str(time.time_ns());app=QtWidgets.QApplication([])
    w=Window(a,root,tasks);w.show();app.exec()
    result=dict(task=a.task,physical_steps=w.env.total_physical_steps,source='LIVE DYNAMIC EVALUATION')
    if a.validate_steps and not a.model:
        before=RUN/'preflight'/a.task/'baseline/episode_0000/integration_states.npz'
        observed=next(root.rglob('integration_states.npz'))
        with np.load(before) as x,np.load(observed) as y:np.testing.assert_array_equal(x['states'][:a.validate_steps],y['states'])
        result['bitwise_equal']=True;write_json(RUN/'qt_equivalence.json',result)
    write_json(root/'viewer_summary.json',result);print(root)


if __name__=='__main__':main()
