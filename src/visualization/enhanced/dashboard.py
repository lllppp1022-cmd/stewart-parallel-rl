"""Display-only transforms applied after frozen Dashboard status/controls."""
import json,time
from pathlib import Path
import numpy as np
from PySide6 import QtCore,QtGui,QtWidgets
import pyqtgraph as pg
from ..live_dashboard import Dashboard
from .style import LEG_COLORS,MODES,series
from .events import read_events,visible_events,latest_event,EventTail

class EnhancedDashboard(Dashboard):
    def __init__(self,*a,**kw):
        self.ready=False;self.events=[];self.last_read=0.;self.home_override=None;self.event_lines=[];self.event_tail=EventTail()
        super().__init__(*a,**kw);self.setWindowTitle('Stewart Phase 1.5.1 Enhanced Control')
        layout=self.centralWidget().layout();row=QtWidgets.QHBoxLayout();self.position_mode=QtWidgets.QComboBox();self.position_mode.addItems(MODES);self.position_mode.setCurrentIndex(1);self.leg_mode=QtWidgets.QComboBox();self.leg_mode.addItems(MODES);self.leg_mode.setCurrentIndex(1);self.angle_unit=QtWidgets.QComboBox();self.angle_unit.addItems(['deg','mdeg'])
        for label,widget in [('Position',self.position_mode),('Slides',self.leg_mode),('Angles',self.angle_unit)]:row.addWidget(QtWidgets.QLabel(label));row.addWidget(widget)
        self.labels=QtWidgets.QCheckBox('Leg labels (L)');self.labels.setChecked(True);self.labels.toggled.connect(lambda enabled:self.send('visual_labels',{'enabled':enabled}));row.addWidget(self.labels);layout.insertLayout(3,row)
        cards=QtWidgets.QGridLayout();self.cards=[]
        for i,color in enumerate(LEG_COLORS):
            b=QtWidgets.QPushButton(f'Leg{i+1}\nWaiting...');b.setStyleSheet(f'border:2px solid {color};font:11px Consolas;padding:2px');b.clicked.connect(lambda checked=False,j=i:self.leg.setCurrentIndex(j+1));cards.addWidget(b,i//3,i%3);self.cards.append(b)
            pix=QtGui.QPixmap(14,14);pix.fill(QtGui.QColor(color));self.leg.setItemIcon(i+1,QtGui.QIcon(pix))
        layout.insertLayout(4,cards);self.baseline_label=QtWidgets.QLabel('Home baseline: waiting');layout.insertWidget(5,self.baseline_label);self.event_label=QtWidgets.QLabel('Events: waiting');self.event_label.setWordWrap(True);layout.insertWidget(6,self.event_label)
        self.show_error.hide()
        for (key,j),(ref,actual) in self.curves.items():
            actual.setPen(pg.mkPen(LEG_COLORS[j],width=1.8));ref.setPen(pg.mkPen(LEG_COLORS[j],width=1,style=QtCore.Qt.PenStyle.DashLine))
        for plot in self.plots:plot.getAxis('left').enableAutoSIPrefix(False);plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.ready=True
    def load_replay(self,path):
        super().load_replay(path)
        from .streaming import enable_segmented_replay
        enable_segmented_replay(self.replay)
        file=Path(path)/'home_baseline.json';self.home_override=json.loads(file.read_text(encoding='utf-8')) if file.exists() else None
    def eventFilter(self,obj,event):
        if self.ready and event.type()==QtCore.QEvent.Type.KeyPress and self.isActiveWindow() and event.key()==QtCore.Qt.Key.Key_L:
            if not event.isAutoRepeat():self.labels.toggle()
            return True
        return super().eventFilter(obj,event)
    def draw(self):
        super().draw()
        if not self.ready:return
        s=self.latest;d=s.data;home=d.get('home_baseline',self.home_override);samples=self.buffer.recent(float(self.window.currentText()),s.time_s)
        if not samples:return
        self.baseline_label.setText('Fixed Home: '+(home['source'] if home else 'UNKNOWN — Delta unavailable; no first-visible-point substitution'))
        t=np.array([x.time_s for x in samples]);get=lambda k:np.asarray([x.data[k] for x in samples])
        for pi,(key,actual,ref) in enumerate([('position','p','p_ref'),('angles','rpy','rpy_ref'),('legs','q','q_ref'),('force','force','f_cmd')]):
            mode=self.position_mode.currentText() if key=='position' else self.leg_mode.currentText();values=[]
            try:av,rv,unit=series(get(actual),get(ref),key,mode,home,self.angle_unit.currentText())
            except ValueError:
                for j in range(3 if key=='position' else 6):
                    for curve in self.curves[(key,j)]:curve.clear()
                self.plots[pi].setTitle(key+' | HOME BASELINE UNKNOWN');continue
            for j in range(av.shape[1]):
                rc,ac=self.curves[(key,j)];selected=(key not in ('legs','force') or self.leg.currentIndex() in (0,j+1)) and (key!='position' or self.position_axis.currentIndex() in (0,j+1))
                ac.setData(t,av[:,j]);rc.setData(t,rv[:,j]);ac.setVisible(selected and self.show_actual.isChecked());rc.setVisible(selected and self.show_ref.isChecked() and (key not in ('position','legs') or mode!='Tracking Error'))
                if ac.isVisible():values.append(av[:,j])
                if rc.isVisible():values.append(rv[:,j])
            if values:
                lo=min(float(v.min()) for v in values);hi=max(float(v.max()) for v in values);pad=max((hi-lo)*.15,.002 if unit=='deg' else .02);self.plots[pi].setYRange(lo-pad,hi+pad,padding=0)
            title={'position':'Platform position','angles':'Roll / Pitch / Yaw (1 deg = 1000 mdeg)','legs':'Moving-joint stroke','force':'Native actual force / command, +/-400 N'}[key];self.plots[pi].setTitle(title+(' | '+mode if key in ('position','legs') else ''));self.plots[pi].setLabel('left',unit)
        delta=(np.asarray(d['q'])-np.asarray(home['slide_m']))*1000 if home else [None]*6
        for i,b in enumerate(self.cards):
            dx=f'{delta[i]:+.4f}' if delta[i] is not None else 'UNKNOWN';b.setText(f'Leg{i+1}  Delta {dx} mm\nErr {(d["q"][i]-d["q_ref"][i])*1000:+.4f} mm | {d["force"][i]:.2f} N')
        folder=self.replay.folder if self.replay else Path(self.session.current['folder'])
        if time.perf_counter()-self.last_read>.2:
            self.events=self.event_tail.read(folder) if self.session and self.session.current.get('stream_logs') else read_events(folder)
            self.last_read=time.perf_counter()
        e=latest_event(self.events,s.time_s);self.event_label.setText(f'SIM {e["time_s"]:.3f}s | {e["label"]} | command {e.get("command_id","unknown")}' if e else 'No recorded event at this time')
        for plot,line in self.event_lines:plot.removeItem(line)
        self.event_lines=[];key={'plan_accepted','target_rejected','rejected','hold','protect_hold','motion_completed','pause_state','episode_started'}
        for e in [e for e in visible_events(self.events,s.time_s,float(self.window.currentText())) if e['type'] in key][-16:]:
            for plot in self.plots:
                line=pg.InfiniteLine(e['time_s'],pen=pg.mkPen('#f58282' if e['type'] in ('target_rejected','protect_hold','rejected') else '#888888',width=1,style=QtCore.Qt.PenStyle.DotLine));plot.addItem(line);self.event_lines.append((plot,line))
    def hover_at(self,plot,args):
        super().hover_at(plot,args)
        if not self.ready or not self.events:return
        t=plot.vb.mapSceneToView(args[0]).x();visible=visible_events(self.events,self.latest.time_s,float(self.window.currentText()))
        if visible:
            e=min(visible,key=lambda e:abs(e['time_s']-t))
            if abs(e['time_s']-t)<.12:
                same=[x for x in self.events if x.get('command_id')==e.get('command_id') and x['time_s']<=self.latest.time_s+1e-10];before=next((x.get('target_before') for x in same if x.get('target_before') is not None),'unknown');after=next((x.get('target_after') for x in reversed(same) if x.get('target_after') is not None),'unknown');self.hover.setText(f'{e["time_s"]:.3f}s {e["label"]} | before {before} | after {after} | preflight {e.get("preflight","not recorded")}')
    def robot_frame(self):
        if not self.ready:return super().robot_frame()
        try:
            rgb=self.replay.image(self.latest)
            if self.labels.isChecked():
                from .overlay import ProjectionOverlay
                if not hasattr(self,'projection'):self.projection=ProjectionOverlay()
                d=self.latest.data;rgb,_=self.projection.apply(rgb,d['qpos'],d['qvel'],d['ctrl'],self.latest.time_s)
            self.replay_robot.show_frame(rgb,self.replay.origin+' + projected leg labels')
        except Exception as exc:self.message.setText('Replay image error: '+str(exc))
