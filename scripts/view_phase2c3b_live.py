"""Single-owner LIVE MuJoCo RGB/Qt viewer; no IPC or GUI access to physics state.
All integration and rendering run sequentially in the Qt main thread. Camera
controls only change MjvCamera; pause stops the simulation clock explicitly.
"""
import _bootstrap,argparse,time,json
from pathlib import Path
import numpy as np,mujoco
from PySide6 import QtWidgets,QtCore,QtGui
import pyqtgraph as pg
from src.model import ROOT,write_json
from src.workspace.feedforward_study import RUN,make_owner
from src.visualization.enhanced.style import LEG_COLORS,leg_mapping,rgba
from src.rendering import camera

class LiveWindow(QtWidgets.QWidget):
    def __init__(self,owner,reference,folder,limit=None):
        super().__init__();self.owner=owner;self.reference=reference;self.folder=folder;self.limit=limit;self.paused=False;self.closed=False;self.finished=False;self.started=time.perf_counter();self.times=[];self.z=[];self.zr=[];self.forces=[];self.ff=[]
        self.setWindowTitle('LARGE-MOTION DYNAMIC FEEDFORWARD CONTROL STUDY | LIVE PHYSICS');layout=QtWidgets.QVBoxLayout(self);self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status);row=QtWidgets.QHBoxLayout();layout.addLayout(row);self.image=QtWidgets.QLabel();self.image.setMinimumSize(640,480);row.addWidget(self.image,2);plots=QtWidgets.QVBoxLayout();row.addLayout(plots,1);self.pz=pg.PlotWidget(title='Actual / reference Z from Home (mm)');plots.addWidget(self.pz);self.pf=pg.PlotWidget(title='Six actual leg forces (N)');plots.addWidget(self.pf);self.pp=pg.PlotWidget(title='Requested FF and applied total-add Fz (N)');plots.addWidget(self.pp)
        self.ps=pg.PlotWidget(title='Six relative slide strokes (mm), actual solid / reference dashed');plots.addWidget(self.ps);self.strokes=[];self.stroke_refs=[];self.slines=[self.ps.plot(pen=pg.mkPen(c,width=1)) for c in LEG_COLORS];self.srefs=[self.ps.plot(pen=pg.mkPen(c,width=1,style=QtCore.Qt.PenStyle.DashLine)) for c in LEG_COLORS]
        self.zline=self.pz.plot(pen=pg.mkPen('#33d5a0',width=2));self.rline=self.pz.plot(pen=pg.mkPen('#ffc857',width=1,style=QtCore.Qt.PenStyle.DashLine));self.flines=[self.pf.plot(pen=pg.mkPen(c,width=1)) for c in LEG_COLORS];self.ffline=self.pp.plot(pen='#ffc857');self.apline=self.pp.plot(pen='#33d5a0');buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
        for label,callback in [('Pause / Resume',self.toggle),('Camera left',lambda:self.rotate(-30)),('Camera right',lambda:self.rotate(30)),('Quit',self.close)]:b=QtWidgets.QPushButton(label);b.clicked.connect(callback);buttons.addWidget(b)
        self.cam=camera();owner.model.vis.global_.offwidth=960;owner.model.vis.global_.offheight=720;self.renderer=mujoco.Renderer(owner.model,width=960,height=720);self.colors={gid:rgba(row['leg']-1) for row in leg_mapping(owner.model) for gid in row['geoms']};self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(1);self.resize(1400,850)
    def toggle(self):self.paused=not self.paused
    def rotate(self,value):self.cam.azimuth+=value;self.display()
    def display(self):
        o=self.owner;self.renderer.update_scene(o.data,camera=self.cam)
        for g in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.colors:g.rgba[:]=self.colors[g.objid];g.matid=-1
        rgb=self.renderer.render().copy();h,w,_=rgb.shape;q=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy();self.image.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.image.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))
        row=o.recorder.states[-1] if o.recorder.states else None;p=o.data.site_xpos[o.site_id];inj=getattr(o.residual,'injection',None);a=inj.audit if inj else {};self.status.setText(f'LIVE ACTUAL PHYSICS | {o.controller.name} | physical t={o.data.time:.3f}s, steps={o.n} | Z actual={(p[2]-o.geometry.home_position[2])*1000:.3f}mm, ref={(o.ref.p[2]-o.geometry.home_position[2])*1000:.3f}mm | FF target={a.get("ff_wrench",np.zeros(6))[2]:+.4f}N | mesh gap={o.expanded_gap["minimum_m"]*1000:.2f}mm | {o.failure or ("COMPLETE - DISPLAY ONLY" if self.finished else "PAUSED CLOCK" if self.paused else "RUNNING")} | geometric coverage remains partial')
    def tick(self):
        try:
            if not self.paused and not self.finished:
                for _ in range(20):
                    self.owner.step()
                    if self.limit and self.owner.n>=self.limit:self.finished=True;break
                    if self.owner.failure and self.owner.data.time>=self.owner.protect_end-1e-10:self.finished=True;break
                    if self.owner.data.time>=self.reference.duration-1e-10:self.finished=True;break
                o=self.owner;inj=getattr(o.residual,'injection',None);a=inj.audit if inj else {};self.times.append(float(o.data.time));self.z.append(float((o.data.site_xpos[o.site_id,2]-o.geometry.home_position[2])*1000));self.zr.append(float((o.ref.p[2]-o.geometry.home_position[2])*1000));self.forces.append(o.data.actuator_force[:6].copy());self.ff.append([float(a.get('ff_wrench',np.zeros(6))[2]),float(a.get('realized_wrench',np.zeros(6))[2])]);self.zline.setData(self.times,self.z);self.rline.setData(self.times,self.zr)
                for j,line in enumerate(self.flines):line.setData(self.times,np.array(self.forces)[:,j])
                self.strokes.append(o.data.qpos[o.qadr].copy()*1000);self.stroke_refs.append(o.ref.q.copy()*1000)
                for j,line in enumerate(self.slines):line.setData(self.times,np.array(self.strokes)[:,j]);self.srefs[j].setData(self.times,np.array(self.stroke_refs)[:,j])
                self.ffline.setData(self.times,np.array(self.ff)[:,0]);self.apline.setData(self.times,np.array(self.ff)[:,1])
            self.display()
            if self.limit and self.finished:self.grab().save(str(self.folder/'live_gui.png'));self.close()
            if time.perf_counter()-self.started>300:self.close()
        except Exception as exc:self.owner.event('viewer_exception',error=repr(exc));self.close();raise
    def closeEvent(self,event):
        if self.closed:event.accept();return
        self.closed=True;self.timer.stop();self.renderer.close();o=self.owner;o.recorder.close(dict(physics_steps=o.n,complete=bool(o.data.time>=self.reference.duration-1e-10 and not o.failure),failure=o.failure,terminal_reason='validation_budget' if self.limit else 'viewer_closed',live_physics=True,no_gui_qpos_or_ctrl_writes=True));event.accept()

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['C4','fourDlead','FF-V','FF-VA','FF-VA-Lead'],default='FF-VA');p.add_argument('--rise',type=float,choices=[4.,6.,8.,10.],default=6.);p.add_argument('--validate-steps',type=int);p.add_argument('--compare-headless',action='store_true');a=p.parse_args()
    if a.validate_steps is not None and not 100<=a.validate_steps<=6000:p.error('Validation budget must be100..6000 steps')
    folder=RUN/'live_viewer'/str(time.time_ns());off=None
    if a.compare_headless:
        if not a.validate_steps:p.error('compare-headless requires a finite validation budget')
        o,r=make_owner(folder/'off',a.method,.05,a.rise)
        for _ in range(a.validate_steps):o.step()
        off=np.array(o.recorder.states);o.recorder.close(dict(physics_steps=o.n,complete=False,terminal_reason='viewer_equivalence_diagnostic'));del o
    o,r=make_owner(folder/'on',a.method,.05,a.rise);app=QtWidgets.QApplication([]);w=LiveWindow(o,r,folder/'on',a.validate_steps);w.show();app.exec()
    if off is not None:
        with np.load(folder/'on/integration_states.npz') as z:equal=bool(np.array_equal(off,z['states']))
        write_json(RUN/'live_viewer_validation.json',dict(passed=equal,bitwise_integration_equal=equal,physics_steps_each=a.validate_steps,root=str(folder),single_physics_owner_thread=True,windows_Qt_MuJoCo_renderer=True,native_glfw_viewer=False));assert equal
    print('LIVE VIEWER ARCHIVED',folder,flush=True)
if __name__=='__main__':main()
