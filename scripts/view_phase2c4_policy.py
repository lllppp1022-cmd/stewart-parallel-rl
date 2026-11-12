"""Single-thread Qt live evaluation, never an archived-state animation."""
import _bootstrap,argparse,time,json
from pathlib import Path
import numpy as np,mujoco,pyqtgraph as pg,torch
from PySide6 import QtCore,QtGui,QtWidgets
from src.model import ROOT,write_json
from src.rl.stewart_large_motion_env import RUN,FIXED,StewartLargeMotionResidualEnv,large_owner
from src.rendering import camera
from src.visualization.enhanced.style import leg_mapping,rgba

class Window(QtWidgets.QWidget):
    def __init__(self,args,folder):
        super().__init__();self.args=args;self.folder=folder;self.env=None;self.model=None;self.renderer=None;self.paused=False;self.finished=False;self.runs=0;self.started=time.perf_counter();self.setWindowTitle('CHECKPOINT EVALUATION | LIVE LARGE-MOTION PHYSICS');layout=QtWidgets.QVBoxLayout(self);self.status=QtWidgets.QLabel();layout.addWidget(self.status);row=QtWidgets.QHBoxLayout();layout.addLayout(row);self.image=QtWidgets.QLabel();self.image.setMinimumSize(800,600);row.addWidget(self.image,2);right=QtWidgets.QVBoxLayout();row.addLayout(right);self.plots=[pg.PlotWidget(title=s) for s in ['Z actual/reference (mm)','XY error (mm) / SO3 (deg)','Six actual leg forces (N)','PPO requested / accepted Fz (N)']]
        for p in self.plots:right.addWidget(p)
        buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
        for label,cb in [('Pause / Resume',self.pause),('Reset new episode',self.reset),('Stop / archive',self.close)]:b=QtWidgets.QPushButton(label);b.clicked.connect(cb);buttons.addWidget(b)
        if args.method=='PPO':
            from stable_baselines3 import PPO
            self.model=PPO.load(args.model,device='cpu')
        self.reset();self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(1);self.resize(1450,950)
    def pause(self):self.paused=not self.paused
    def reset(self):
        if self.env:self.env.close()
        elif getattr(self,'owner',None):self.owner.recorder.close(dict(source='Baseline Comparison reset',complete=self.owner.n>=15000,failure=self.owner.failure))
        if self.renderer:self.renderer.close()
        self.runs+=1;folder=self.folder/f'episode_reset{self.runs}';self.finished=False;self.times=[];self.values=[]
        if self.args.method=='C4':self.owner,self.reference=large_owner(folder,FIXED,method='C4')
        else:
            self.env=StewartLargeMotionResidualEnv(folder,mode='Checkpoint Evaluation');self.obs,_=self.env.reset(seed=42);self.owner=self.env.owner;self.reference=self.env.reference
        self.owner.model.vis.global_.offwidth=960;self.owner.model.vis.global_.offheight=720;self.renderer=mujoco.Renderer(self.owner.model,width=960,height=720);self.cam=camera();self.colors={g:rgba(x['leg']-1) for x in leg_mapping(self.owner.model) for g in x['geoms']};self.lines=[]
        for p,n in zip(self.plots,[2,2,6,2]):p.clear();self.lines.append([p.plot(pen=pg.intColor(i,n)) for i in range(n)])
    def tick(self):
        if not self.paused and not self.finished:
            if self.env:
                action=np.zeros(1)
                if self.model:
                    with torch.no_grad():distribution=self.model.policy.get_distribution(torch.as_tensor(self.obs.reshape(1,-1),device='cpu'));action=distribution.distribution.mean[0].cpu().numpy();self.env.policy_context=dict(raw_action=float(action[0]),actor_mean=float(action[0]),actor_std=float(distribution.distribution.stddev[0,0]),policy_version=int(self.model.num_timesteps//256))
                self.obs,_,term,trunc,_=self.env.step(action);self.finished=term or trunc
            else:
                for _ in range(50):
                    self.owner.step()
                    if self.owner.failure or self.owner.n>=15000:self.finished=True;break
            o=self.owner;r=self.reference.at(o.data.time);p=o.data.site_xpos[o.site_id];a=getattr(o.residual,'injection',None);audit=a.audit if a else {};effect=audit.get('probe_effect_command',np.zeros(6));applied=float((o.residual.base.jacobian.T@effect)[2]);self.times.append(float(o.data.time));from scipy.spatial.transform import Rotation
            angle=np.rad2deg(np.linalg.norm(Rotation.from_matrix(o.data.site_xmat[o.site_id].reshape(3,3)@r.r.T).as_rotvec()));self.values.append([float((p[2]-o.geometry.home_position[2])*1000),float((r.p[2]-o.geometry.home_position[2])*1000),float(np.linalg.norm(p[:2]-r.p[:2])*1000),angle,*o.data.actuator_force[:6],float(audit.get('probe_wrench',np.zeros(6))[2]),applied]);v=np.asarray(self.values)
            for lines,indices in zip(self.lines,[[0,1],[2,3],list(range(4,10)),[10,11]]):
                for line,i in zip(lines,indices):line.setData(self.times,v[:,i])
            if self.args.validate_steps and o.n>=self.args.validate_steps:self.finished=True
        o=self.owner;self.renderer.update_scene(o.data,camera=self.cam)
        for g in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.colors:g.rgba[:]=self.colors[g.objid];g.matid=-1
        rgb=self.renderer.render().copy();h,w,_=rgb.shape;im=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy();self.image.setPixmap(QtGui.QPixmap.fromImage(im).scaled(self.image.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio));state='VALIDATION BUDGET / INCOMPLETE' if self.finished and o.n<15000 else 'COMPLETE / DISPLAY ONLY' if self.finished else 'PAUSED' if self.paused else 'RUNNING';self.status.setText(f'CHECKPOINT EVALUATION | {self.args.method} | physical t={o.data.time:.3f}s | {o.failure or state} | PARTIAL GEOMETRIC COVERAGE | deterministic policy')
        if self.args.validate_steps and self.finished:self.grab().save(str(self.folder/'live_gui.png'));self.close()
        if time.perf_counter()-self.started>600:self.close()
    def closeEvent(self,event):
        self.timer.stop();self.renderer.close()
        if self.env:self.env.close()
        else:self.owner.recorder.close(dict(source='Baseline Comparison',complete=self.owner.n>=15000,failure=self.owner.failure))
        event.accept()

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['C4','FF-VA','PPO'],default='FF-VA');p.add_argument('--model');p.add_argument('--validate-steps',type=int);p.add_argument('--compare-headless',action='store_true');a=p.parse_args()
    if a.method=='PPO' and not a.model:p.error('PPO requires --model')
    root=RUN/'viewer'/str(time.time_ns());off=None
    if a.compare_headless:
        if a.method!='FF-VA' or not a.validate_steps or a.validate_steps%50:p.error('Equivalence requires FF-VA and finite multiple50')
        e=StewartLargeMotionResidualEnv(root/'off',mode='Viewer Equivalence');e.reset(seed=42)
        for _ in range(a.validate_steps//50):e.step([0.])
        off=np.array(e.owner.recorder.states);e.close()
    app=QtWidgets.QApplication([]);w=Window(a,root/'on');w.show();app.exec()
    if off is not None:
        with np.load(root/'on/episode_reset1/episode_0000/integration_states.npz') as z:same=bool(np.array_equal(off,z['states']))
        write_json(RUN/'validation/viewer.json',dict(passed=same,bitwise_equal=same,physics_steps_each=a.validate_steps,root=str(root),single_owner_thread=True));assert same
    print('VIEWER ARCHIVED',root)
if __name__=='__main__':main()
