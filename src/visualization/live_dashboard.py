"""Qt WINDOW B: commands and immutable snapshots only; never owns live MjData."""
import time,json,csv
from pathlib import Path
from queue import Empty
import numpy as np
from PySide6 import QtCore,QtGui,QtWidgets
import pyqtgraph as pg
from .telemetry import SnapshotBuffer,key_command

COLORS=['#48c8ff','#ffad42','#7fdd77','#ff6e91','#b9a0ff','#efdf62']

class RobotReplayWindow(QtWidgets.QLabel):
    def __init__(self):
        super().__init__();self.setWindowTitle('Stewart Replay Robot');self.resize(740,600);self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);self.setStyleSheet('background:#101d29;color:white')
    def show_frame(self,frame,origin):
        a=np.ascontiguousarray(frame);q=QtGui.QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QtGui.QImage.Format.Format_RGB888).copy();self.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio));self.setWindowTitle('Stewart Replay Robot | '+origin)

class Dashboard(QtWidgets.QMainWindow):
    def __init__(self,session=None,replay_folder=None):
        super().__init__();self.session=session;self.replay=None;self.replay_robot=None;self.buffer=SnapshotBuffer();self.latest=None;self.closing=False;self.final_close=False;self.pending_restore=False;self.replay_playing=False;self.last_wall=time.perf_counter();self.playhead=0.
        self.setWindowTitle('Stewart Phase1.5 Control Panel');screen=QtWidgets.QApplication.primaryScreen().availableGeometry();width=max(620,int(screen.width()*.42));self.setGeometry(screen.right()-width,screen.top()+20,width,max(700,screen.height()-40));self.setMinimumSize(570,650)
        pg.setConfigOptions(background='#101d29',foreground='#e8f0f8',antialias=False)
        central=QtWidgets.QWidget();self.setCentralWidget(central);layout=QtWidgets.QVBoxLayout(central);layout.setContentsMargins(7,5,7,5);layout.setSpacing(3)
        self.banner=QtWidgets.QLabel('DYNAMIC CONTROL DEMONSTRATION | PARTIAL PROXY QUERY');self.banner.setStyleSheet('background:#164052;color:white;padding:5px;font-weight:bold');layout.addWidget(self.banner)
        self.metrics=QtWidgets.QLabel('Waiting for physical owner...');self.metrics.setStyleSheet('font:11px Consolas');self.metrics.setMinimumHeight(56);layout.addWidget(self.metrics)
        controls=QtWidgets.QGridLayout();layout.addLayout(controls);self.sliders={};self.slider_labels={}
        for row,(key,title,scale) in enumerate([('z_mm','Z target delta (mm)',100),('roll_deg','Roll target (deg)',1000),('pitch_deg','Pitch target (deg)',1000)]):
            controls.addWidget(QtWidgets.QLabel(title),row,0);slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);slider.setRange(-400,400);slider.setValue(0);slider.setObjectName(key);label=QtWidgets.QLabel('0.000');label.setMinimumWidth(48);controls.addWidget(slider,row,1);controls.addWidget(label,row,2);self.sliders[key]=(slider,scale);self.slider_labels[key]=label
            slider.valueChanged.connect(lambda value,k=key,s=scale:self.slider_changed(k,value/s));slider.sliderReleased.connect(self.submit_sliders)
        self.debounce=QtCore.QTimer(self);self.debounce.setSingleShot(True);self.debounce.setInterval(220);self.debounce.timeout.connect(self.submit_sliders)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row);row.addWidget(QtWidgets.QLabel('Next controller:'));self.controller=QtWidgets.QComboBox();self.controller.addItems(['C0','C1','C2']);self.controller.setCurrentText(session.base.get('controller','C0') if session else 'C0');row.addWidget(self.controller);self.controller.currentTextChanged.connect(lambda _:self.send('controller'))
        for name,kind in [('Home (R)','home'),('Hold (Space)','hold'),('Pause (P)','pause'),('Reset','reset')]:
            b=QtWidgets.QPushButton(name);b.clicked.connect(lambda checked=False,k=kind:self.action(k));row.addWidget(b)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row);self.record=QtWidgets.QPushButton('Record');self.record.setCheckable(True);self.record.setChecked(session.base.get('record',False) if session else False);self.record.clicked.connect(lambda checked:self.send('record',{'enabled':checked}));row.addWidget(self.record)
        for name in ['X (disabled)','Y (disabled)','Yaw (disabled)']:
            b=QtWidgets.QPushButton(name);b.setEnabled(False);b.setToolTip('Workspace/dynamics not independently qualified');row.addWidget(b)
        b=QtWidgets.QPushButton('Camera (V)');b.clicked.connect(lambda:self.send('camera'));row.addWidget(b);b=QtWidgets.QPushButton('Quit');b.clicked.connect(self.close);row.addWidget(b)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row);self.preset=QtWidgets.QComboBox();self.preset.addItems(['home','z','roll','pitch','z_roll','roll_pitch','periodic']);row.addWidget(self.preset);b=QtWidgets.QPushButton('Run preset');b.clicked.connect(lambda:self.send('preset',{'name':self.preset.currentText()}));row.addWidget(b)
        self.window=QtWidgets.QComboBox();self.window.addItems(['5','10','20']);self.window.setCurrentText('10');row.addWidget(QtWidgets.QLabel('Seconds'));row.addWidget(self.window);self.leg=QtWidgets.QComboBox();self.leg.addItems(['All legs']+[f'Leg {i}' for i in range(1,7)]);row.addWidget(self.leg)
        self.show_ref=QtWidgets.QCheckBox('Ref dashed');self.show_ref.setChecked(True);self.show_actual=QtWidgets.QCheckBox('Actual solid');self.show_actual.setChecked(True);self.show_error=QtWidgets.QCheckBox('Leg error');row.addWidget(self.show_ref);row.addWidget(self.show_actual);row.addWidget(self.show_error)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row);row.addWidget(QtWidgets.QLabel('Position plot:'));self.position_axis=QtWidgets.QComboBox();self.position_axis.addItems(['XYZ world','X','Y','Z']);row.addWidget(self.position_axis);row.addWidget(QtWidgets.QLabel('Keys affect Control Panel only. Reset = NEW EPISODE.'))
        tabs=QtWidgets.QTabWidget();layout.addWidget(tabs,1);self.graph=pg.GraphicsLayoutWidget();tabs.addTab(self.graph,'Live curves');self.plots=[];self.curves={}
        for index,(key,title,units,count) in enumerate([('position','Platform XYZ: world coordinates','mm',3),('angles','Roll / Pitch / Yaw','deg',3),('legs','Relative slide stroke (actuatorpos)','mm',6),('force','Six actual forces | native limit +/-400 N','N',6)]):
            plot=self.graph.addPlot(row=index,col=0,title=title);plot.showGrid(x=True,y=True,alpha=.2);plot.setLabel('left',units);plot.setLabel('bottom','Simulation time','s');plot.addLegend(offset=(4,4),colCount=3);self.plots.append(plot)
            for j in range(count):
                name=(['X','Y','Z'][j] if key=='position' else ['Roll','Pitch','Yaw'][j] if key=='angles' else f'Leg{j+1}');color=COLORS[j]
                actual=plot.plot(pen=pg.mkPen(color,width=1.8),name=name);ref=plot.plot(pen=pg.mkPen(color,width=1,style=QtCore.Qt.PenStyle.DashLine));self.curves[(key,j)]=(ref,actual)
        passive=QtWidgets.QWidget();p_layout=QtWidgets.QVBoxLayout(passive);self.passive_leg=QtWidgets.QComboBox();self.passive_leg.addItems([f'Leg {i}' for i in range(1,7)]);p_layout.addWidget(self.passive_leg);self.passive_text=QtWidgets.QLabel('');p_layout.addWidget(self.passive_text);self.passive_plot=pg.PlotWidget(title='Five passive revolute joints, selected leg only');self.passive_plot.setLabel('left','Angle','deg');self.passive_plot.setLabel('bottom','Simulation time','s');p_layout.addWidget(self.passive_plot);self.passive_curves=[self.passive_plot.plot(pen=pg.mkPen(COLORS[j],width=2)) for j in range(5)];tabs.addTab(passive,'Passive joints')
        self.hover=QtWidgets.QLabel('Focus this panel: W/S Z, A/D Roll, I/K Pitch, R Home, Space Hold, P Pause, V Camera, Q/Esc Quit');self.hover.setWordWrap(True);self.hover.setStyleSheet('font-size:10px');layout.addWidget(self.hover)
        self.message=QtWidgets.QLabel('');self.message.setWordWrap(True);layout.addWidget(self.message)
        bottom=QtWidgets.QHBoxLayout();layout.addLayout(bottom)
        for title,callback in [('Open Replay',self.open_replay_dialog),('Play/Pause',self.toggle_replay),('Frame >',self.next_frame),('Export CSV',self.export_csv),('Screenshot',self.screenshot)]:
            b=QtWidgets.QPushButton(title);b.clicked.connect(callback);bottom.addWidget(b)
        self.speed=QtWidgets.QComboBox();self.speed.addItems(['0.25','0.5','1','2','4']);self.speed.setCurrentText('1');bottom.addWidget(self.speed)
        self.proxies=[pg.SignalProxy(plot.scene().sigMouseMoved,rateLimit=15,slot=lambda args,p=plot:self.hover_at(p,args)) for plot in self.plots]
        QtWidgets.QApplication.instance().installEventFilter(self);self.timer=QtCore.QTimer(self);self.timer.setInterval(50);self.timer.timeout.connect(self.refresh);self.timer.start()
        if replay_folder:self.load_replay(replay_folder)
    def slider_changed(self,key,value):self.slider_labels[key].setText(f'{value:.3f}');self.debounce.start()
    def submit_sliders(self):
        self.debounce.stop();self.send('target',{k:s.value()/scale for k,(s,scale) in self.sliders.items()},'slider')
    def send(self,kind,payload=None,source='panel'):
        if self.replay or not self.session:return
        ok=self.session.send(kind,payload,source)
        if not ok:self.message.setText('Command queue full: command rejected, no unsafe state change')
    def action(self,kind):
        if kind=='reset':
            if self.session and not self.replay:self.session.request_reset(self.controller.currentText());self.message.setText('Reset: old episode archiving; new process/episode will start')
            return
        if kind in ['home','hold']:self.pending_restore=True
        self.send(kind)
    def eventFilter(self,obj,event):
        if event.type()!=QtCore.QEvent.Type.KeyPress or not self.isActiveWindow():return super().eventFilter(obj,event)
        key=event.key();name='SPACE' if key==QtCore.Qt.Key.Key_Space else 'ESCAPE' if key==QtCore.Qt.Key.Key_Escape else event.text().upper()
        if self.replay:
            if key in (QtCore.Qt.Key.Key_Right,):self.next_frame();return True
            if name=='P':self.toggle_replay();return True
            return super().eventFilter(obj,event)
        cmd=key_command(name,True,event.isAutoRepeat())
        if cmd:
            kind,payload=cmd
            if kind=='quit':self.close()
            elif kind=='home':self.pending_restore=True;self.send(kind,payload,'keyboard:'+name)
            elif kind=='hold':self.pending_restore=True;self.send(kind,payload,'keyboard:'+name)
            else:self.send(kind,payload,'keyboard:'+name)
            return True
        return super().eventFilter(obj,event)
    def refresh(self):
        now=time.perf_counter();delta=now-self.last_wall;self.last_wall=now
        if self.replay:
            if self.replay_playing:
                self.playhead+=delta*float(self.speed.currentText());index=max(0,np.searchsorted(self.replay.times,self.playhead,side='right')-1);self.replay.at(index)
                if self.playhead>=self.replay.times[-1]:self.replay_playing=False
            self.latest=self.replay.at(self.replay.index);self.buffer.rows.clear()
            for snap in self.replay.history(20):self.buffer.append(snap)
            self.robot_frame()
        elif self.session:
            alive=self.session.poll()
            for _ in range(64):
                try:snap=self.session.telemetry.get_nowait()
                except Empty:break
                self.latest=snap;self.buffer.append(snap)
            if not alive and self.closing:self.final_close=True;self.close();return
            if not alive and self.session.base.get('close_on_complete'):self.final_close=True;self.close();return
        if self.latest:self.draw()
    def draw(self):
        s=self.latest;d=s.data;failure=bool(d['failure']);self.banner.setText(f'{s.mode} | {s.status} | PARTIAL PROXY QUERY');self.banner.setStyleSheet('color:white;padding:5px;font-weight:bold;background:'+('#922d3e' if failure else '#8d6520' if s.status=='PAUSED' else '#164052'))
        self.metrics.setText(f'Episode {s.episode}  sim {s.time_s:.3f}s  step {s.step}  ACTIVE {d["controller"]}\nZ actual {d["p"][2]*1000:.4f} mm  R/P {d["rpy"][0]:.4f}/{d["rpy"][1]:.4f} deg | pose {d["pose_error_mm"]:.4f} mm, {d["angle_error_deg"]:.5f} deg\nLeg err {d["leg_error_mm"]:.4f} mm | force {d["force_peak_N"]:.2f} N /400 | weld {d["weld_t"]:.2e} m {d["weld_r"]:.2e} rad | proxy {d["proxy_gap"]*1000:.2f} mm')
        playback=(' | Replay '+('PLAYING' if self.replay_playing else 'PAUSED')+f' {self.speed.currentText()}x') if self.replay else ''
        self.message.setText(str(d.get('failure') or d.get('message',''))+playback+' | Reward/Actor/Policy: N/A (no training)')
        if 'accepted_target' in d and not self.debounce.isActive() and not any(slider.isSliderDown() for slider,_ in self.sliders.values()):
            displayed_target=d.get('pending_target') or d['accepted_target']
            for key,(slider,scale) in self.sliders.items():slider.blockSignals(True);slider.setValue(round(displayed_target[key]*scale));slider.blockSignals(False);self.slider_labels[key].setText(f'{displayed_target[key]:.3f}')
            self.pending_restore=False
        samples=self.buffer.recent(float(self.window.currentText()),s.time_s)
        if not samples:return
        t=np.array([x.time_s for x in samples]);arrays=lambda key:np.asarray([x.data[key] for x in samples])
        for pi,(key,actual,ref,scale) in enumerate([('position','p','p_ref',1000),('angles','rpy','rpy_ref',1),('legs','q','q_ref',1000),('force','force','f_cmd',1)]):
            av=arrays(actual)*scale;rv=arrays(ref)*scale
            visible_values=[]
            for j in range(av.shape[1]):
                rc,ac=self.curves[(key,j)];selected=key!='legs' or self.leg.currentIndex()==0 or self.leg.currentIndex()==j+1
                if key=='position':selected=self.position_axis.currentIndex()==0 or self.position_axis.currentIndex()==j+1
                ac.setData(t,av[:,j]-rv[:,j] if key=='legs' and self.show_error.isChecked() else av[:,j]);rc.setData(t,rv[:,j]);ac.setVisible(selected and self.show_actual.isChecked());rc.setVisible(selected and self.show_ref.isChecked() and not (key=='legs' and self.show_error.isChecked()))
                if ac.isVisible():visible_values.append(av[:,j]-rv[:,j] if key=='legs' and self.show_error.isChecked() else av[:,j])
                if rc.isVisible():visible_values.append(rv[:,j])
            if visible_values:
                lo=min(float(v.min()) for v in visible_values);hi=max(float(v.max()) for v in visible_values);padding=max((hi-lo)*.15,[.02,.005,.01,.3][pi]);self.plots[pi].setYRange(lo-padding,hi+padding,padding=0)
        self.plots[2].setTitle('Slide tracking error: actual - reference' if self.show_error.isChecked() else 'Relative slide stroke (actuatorpos)')
        for plot in self.plots:plot.setXRange(max(0,s.time_s-float(self.window.currentText())),max(.05,s.time_s),padding=0)
        leg=self.passive_leg.currentIndex();a=np.asarray([x.data['passive'][leg] for x in samples])*180/np.pi
        for j,curve in enumerate(self.passive_curves):curve.setData(t,a[:,j])
        self.passive_text.setText(f'Leg {leg+1}, five passive angles (deg): '+', '.join(f'{x:.3f}' for x in a[-1]))
    def hover_at(self,plot,args):
        if not self.latest:return
        point=plot.vb.mapSceneToView(args[0]);samples=list(self.buffer.rows)
        if samples:
            snap=min(samples,key=lambda x:abs(x.time_s-point.x()));self.hover.setText(f'Hover t={snap.time_s:.4f}s | q(mm)='+str(np.round(np.array(snap.data['q'])*1000,4))+' | RPY(deg)='+str(np.round(snap.data['rpy'],4)))
    def open_replay_dialog(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'Select archived episode/run',str(Path(self.session.folder) if self.session else Path.cwd()))
        if path:self.load_replay(path)
    def load_replay(self,path):
        if self.session and self.session.owner and self.session.owner.is_alive():self.message.setText('Stop Live Control before loading Replay');return
        from .replay import ReplaySource
        try:self.replay=ReplaySource(path)
        except Exception as exc:self.message.setText(str(exc));return
        self.buffer=SnapshotBuffer();self.playhead=float(self.replay.times[0]);self.replay_robot=RobotReplayWindow();self.replay_robot.show();self.replay_playing=False;self.record.setEnabled(False)
        self.controller.setEnabled(False)
        for slider,_ in self.sliders.values():slider.setEnabled(False)
    def robot_frame(self):
        try:self.replay_robot.show_frame(self.replay.image(self.latest),self.replay.origin)
        except Exception as exc:self.message.setText('Replay image error: '+str(exc))
    def toggle_replay(self):
        if self.replay:
            if self.playhead>=self.replay.times[-1]:self.playhead=float(self.replay.times[0]);self.replay.at(0)
            self.replay_playing=not self.replay_playing
    def next_frame(self):
        if self.replay:self.replay_playing=False;self.replay.at(min(self.replay.index+1,len(self.replay.times)-1));self.playhead=float(self.replay.times[self.replay.index])
    def export_csv(self):
        path=QtWidgets.QFileDialog.getSaveFileName(self,'Export recorded physical data','','CSV (*.csv)')[0]
        if not path:return
        if Path(path).exists():self.message.setText('Refusing to overwrite existing export');return
        import pandas as pd
        if self.replay:self.replay.df.to_csv(path,index=False)
        elif self.session:
            folder=Path(self.session.current['folder']);file=folder/'physics.csv.gz'
            if file.exists():
                try:pd.read_csv(file).to_csv(path,index=False)
                except Exception:self.message.setText('Episode journal is still open; export after stop/replay')
    def screenshot(self):
        folder=Path(self.session.current['folder']) if self.session else self.replay.folder;self.grab().save(str(folder/f'dashboard_{time.time_ns()}.png'))
    def closeEvent(self,event):
        if self.final_close or not self.session:
            self.timer.stop();QtWidgets.QApplication.instance().removeEventFilter(self)
            if self.replay:self.replay.close()
            if self.replay_robot:self.replay_robot.close()
            event.accept();return
        self.closing=True;self.session.request_stop();self.message.setText('Archiving episode and closing renderer...');event.ignore()
