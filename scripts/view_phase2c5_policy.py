"""Windows Qt viewer. The timer owns both physical integration and display."""
import _bootstrap
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pyqtgraph as pg
import torch
from PySide6 import QtCore, QtGui, QtWidgets
from stable_baselines3 import PPO

from src.model import ROOT, write_json
from src.rendering import camera
from src.rl.stewart_attitude_env import StewartAttitudeResidualEnv
from src.visualization.enhanced.style import leg_mapping, rgba
from src.workspace.attitude_study import RUN


class CameraImage(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.cam=camera()
        self.drag=None

    def mousePressEvent(self,event):
        self.drag=event.position()

    def mouseMoveEvent(self,event):
        if self.drag is not None:
            delta=event.position()-self.drag
            if event.buttons() & QtCore.Qt.MouseButton.RightButton:
                self.cam.lookat[0]-=delta.x()*.001
                self.cam.lookat[2]+=delta.y()*.001
            else:
                self.cam.azimuth-=delta.x()*.45
                self.cam.elevation=float(np.clip(self.cam.elevation+delta.y()*.35,-89,89))
            self.drag=event.position()

    def mouseReleaseEvent(self,event):
        self.drag=None

    def wheelEvent(self,event):
        self.cam.distance=float(np.clip(self.cam.distance*np.exp(-event.angleDelta().y()/1200),.25,8.))


class Window(QtWidgets.QWidget):
    def __init__(self,args,root,tasks,method):
        super().__init__()
        self.args,self.root,self.tasks,self.method=args,root,tasks,method
        self.model=PPO.load(args.model,device="cpu") if args.model else None
        self.env=None
        self.renderer=None
        self.runs=0
        self.paused=False
        self.started=time.perf_counter()
        self.setWindowTitle("Stewart attitude | live closed-loop evaluation")
        main=QtWidgets.QVBoxLayout(self)
        self.status=QtWidgets.QLabel()
        main.addWidget(self.status)
        selectors=QtWidgets.QHBoxLayout()
        main.addLayout(selectors)
        self.task_select=QtWidgets.QComboBox()
        for key,task in tasks.items():
            self.task_select.addItem(f"{key} | Mx/My {task['torque'][0]:g}, {task['torque'][1]:g} Nm",key)
        self.task_select.setCurrentIndex(self.task_select.findData(args.task))
        self.control_select=QtWidgets.QComboBox()
        self.control_select.addItem(method,False)
        if self.model:
            self.control_select.addItem("PPO checkpoint",True)
            self.control_select.setCurrentIndex(1)
        selectors.addWidget(self.task_select)
        selectors.addWidget(self.control_select)
        selectors.addWidget(QtWidgets.QLabel("Select task/controller, then Reset"))
        body=QtWidgets.QHBoxLayout()
        main.addLayout(body)
        self.image=CameraImage()
        self.image.setMinimumSize(650,530)
        body.addWidget(self.image,2)
        graphs=QtWidgets.QVBoxLayout()
        body.addLayout(graphs,1)
        self.graphs=[]
        for title in ("Roll/Pitch actual & reference (deg)", "XYZ tracking error (mm)",
                      "Six slide strokes (mm)", "Six actuator forces (N)", "Mx/My request & accepted (Nm)"):
            plot=pg.PlotWidget(title=title)
            graphs.addWidget(plot)
            self.graphs.append(plot)
        controls=QtWidgets.QHBoxLayout()
        main.addLayout(controls)
        for text,handler in [("Pause / resume",self.pause),("Reset new episode",self.reset),("Stop / archive",self.close)]:
            button=QtWidgets.QPushButton(text)
            controls.addWidget(button)
            button.clicked.connect(handler)
        self.reset()
        self.timer=QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)
        self.resize(1430,930)

    def pause(self):
        self.paused=not self.paused

    def reset(self):
        if self.env:
            self.env.close()
        if self.renderer:
            self.renderer.close()
        self.runs+=1
        self.task=self.tasks[self.task_select.currentData()]
        self.use_model=bool(self.control_select.currentData())
        self.env=StewartAttitudeResidualEnv(self.root/f"run{self.runs}",[self.task],self.method,source="Qt live evaluation")
        self.obs,_=self.env.reset(seed=42)
        self.owner=self.env.owner
        self.renderer=mujoco.Renderer(self.owner.model,width=640,height=480)
        self.colors={gid:rgba(row["leg"]-1) for row in leg_mapping(self.owner.model) for gid in row["geoms"]}
        self.series=[]
        self.lines=[]
        self.finished=False
        self.paused=False
        names=[["Roll","Roll ref","Pitch","Pitch ref"],["X","Y","Z"],
               [f"Leg {j}" for j in range(1,7)],[f"Leg {j}" for j in range(1,7)],
               ["Mx req","My req","Mx accepted","My accepted"]]
        for plot,labels in zip(self.graphs,names):
            plot.clear()
            if plot.plotItem.legend is None:
                plot.addLegend(colCount=2)
            self.lines.append([plot.plot(pen=pg.intColor(i,len(labels)),name=label) for i,label in enumerate(labels)])

    def tick(self):
        if not self.paused and not self.finished:
            action=self.model.predict(self.obs,deterministic=True)[0] if self.use_model else np.zeros(2)
            self.obs,_,term,trunc,_=self.env.step(action)
            d=self.env.last_row
            self.series.append([d["time_s"],d["rpy_0"],d["rpy_ref_0"],d["rpy_1"],d["rpy_ref_1"],
                *[(d[f"p_{i}"]-d[f"p_ref_{i}"])*1000 for i in range(3)],
                *[d[f"q_{i}"]*1000 for i in range(6)],*[d[f"force_{i}"] for i in range(6)],
                *[d[f"{kind}_{axis}_Nm"] for kind in ("requested","accepted") for axis in ("mx","my")]])
            data=np.array(self.series)
            i=1
            for group in self.lines:
                for line in group:
                    line.setData(data[:,0],data[:,i])
                    i+=1
            self.finished=term or trunc or bool(self.args.validate_steps and self.owner.n>=self.args.validate_steps)
        o=self.owner
        self.renderer.update_scene(o.data,camera=self.image.cam)
        for geom in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid in self.colors:
                geom.rgba[:]=self.colors[geom.objid]
                geom.matid=-1
        rgb=self.renderer.render().copy()
        h,w,_=rgb.shape
        image=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
        self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        status=o.failure or ("DISPLAY ONLY" if self.finished else "PAUSED" if self.paused else "RUNNING")
        self.status.setText(f"{self.task['id']} | {'PPO' if self.use_model else self.method} | t={o.data.time:.3f} s | {status} | Partial geometry")
        if self.finished and self.args.validate_steps:
            self.grab().save(str(self.root/"viewer.png"))
            self.close()
        elif time.perf_counter()-self.started>600:
            self.close()

    def closeEvent(self,event):
        self.timer.stop()
        self.renderer.close()
        self.env.close()
        event.accept()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["R1","R2","R3","T0","T1","T2","T3","T4","T5","T6","T7",
                                      "E_roll","E_pitch","E_mix","E_strong_regression"],default="R3")
    p.add_argument("--model",type=Path)
    p.add_argument("--controller",choices=["C0","C4","fourDlead"])
    p.add_argument("--validate-steps",type=int,choices=[1000,2000,6000])
    a=p.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/"configs/phase2c5_attitude_env.yaml").read_text())
    method=a.controller or config["method"]
    if a.task.startswith("R"):
        if a.model:
            p.error("The checkpoint is admitted only on the preflighted Home disturbance tasks")
    elif method!=config["method"]:
        p.error("Disturbance viewer uses the matching preflighted baseline")
    tasks={key:dict(id=key,reference=key,torque=[0.,0.],on=2.,off=3.5) for key in ["R1","R2","R3"]} if not a.model else {}
    if method==config["method"]:
        final=json.loads((ROOT/"configs/phase2c5_eval.yaml").read_text())["tasks"]
        tasks.update({t["id"]:t for t in config["train"]+final})
    if a.task not in tasks:
        p.error("Requested task has no matching preflight evidence")
    root=RUN/"viewer"/str(time.time_ns())
    app=QtWidgets.QApplication([])
    window=Window(a,root,tasks,method)
    window.show()
    app.exec()
    write_json(root/"viewer_summary.json",dict(steps=window.env.total_physical_steps,source="live Qt",task=window.task["id"]))
    print(root,flush=True)


if __name__=="__main__":
    main()
