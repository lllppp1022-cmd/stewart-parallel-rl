"""Read-only Qt analysis of actual historical data. No environment or optimizer."""
import argparse
import os
os.environ['PYQTGRAPH_QT_LIB']='PySide6'
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtWidgets,QtCore
from phase2c7v_common import ROOT,RUN,SEEDS,NAMES,read,write_json,stage

class Analysis(QtWidgets.QWidget):
    def __init__(self):
        super().__init__();self.setWindowTitle('Stewart | HISTORICAL TRAINING ANALYSIS | READ ONLY')
        self.data=pd.read_csv(ROOT/'results/phase2c7v_reward_components.csv');self.ep=pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv');self.up=pd.read_csv(RUN/'updates.csv')
        self.ev=pd.read_csv(ROOT/'results/phase2c7v_checkpoint_evaluation.csv')
        self.up['stage']=self.up.decisions.map(stage)
        self.inventory=read(RUN/'checkpoint_inventory.json');layout=QtWidgets.QVBoxLayout(self);bar=QtWidgets.QHBoxLayout();layout.addLayout(bar)
        self.seed=QtWidgets.QComboBox();self.seed.addItems([str(s) for s in SEEDS]);bar.addWidget(self.seed)
        self.stage=QtWidgets.QComboBox();self.stage.addItems(['All stages','A','B','C']);bar.addWidget(self.stage)
        self.metric_select=QtWidgets.QComboBox();self.metric_select.addItems(['Complete episode return','Immediate reward','Five costs','Cost fractions','Actor / Critic delta','Value loss','Explained variance','KL / clip fraction','Actor standard deviation','Value targets / advantages','Fixed Dev_mix return','Fixed Dev_mix load/unload IAE']);bar.addWidget(self.metric_select)
        self.raw=QtWidgets.QCheckBox('Raw data');self.raw.setChecked(True);bar.addWidget(self.raw)
        self.trend=QtWidgets.QCheckBox('Display trend');self.trend.setChecked(True);bar.addWidget(self.trend)
        self.checkpoint=QtWidgets.QComboBox();layout.addWidget(self.checkpoint)
        self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.plot=pg.PlotWidget();self.plot.addLegend();self.plot.setLabel('bottom','Actual PPO Decisions');layout.addWidget(self.plot)
        self.note=QtWidgets.QLabel('Dots are actual measurements. Trends are display only. Curriculum distributions and episode durations differ. x = incomplete fragment, excluded from complete-return averages.\nCheckpoint selection only displays file/hash; this analysis window never runs or trains a policy.');self.note.setWordWrap(True);layout.addWidget(self.note)
        self.seed.currentIndexChanged.connect(self.fill);self.checkpoint.currentIndexChanged.connect(self.describe)
        for w in (self.stage,self.metric_select):w.currentIndexChanged.connect(self.refresh)
        self.raw.toggled.connect(self.refresh);self.trend.toggled.connect(self.refresh)
        self.fill();self.resize(1400,850)
    def fill(self):
        self.checkpoint.blockSignals(True);self.checkpoint.clear()
        for r in self.inventory:
            if r['seed']==int(self.seed.currentText()):self.checkpoint.addItem(f'{r["decisions"]} decisions | {r["path"].split(chr(92))[-1]}',r)
        self.checkpoint.blockSignals(False);self.describe();self.refresh()
    def describe(self):
        r=self.checkpoint.currentData()
        if r:self.info.setText(f'EXISTING FROZEN CHECKPOINT: {r["decisions"]} decisions\n{r["path"]}\nSHA256: {r["sha256"]}')
    def select(self,table):
        g=table[table.seed==int(self.seed.currentText())]
        return g if self.stage.currentIndex()==0 else g[g.stage==self.stage.currentText()]
    def refresh(self):
        self.plot.clear();self.plot.addLegend();mode=self.metric_select.currentIndex();g=self.select(self.data);e=self.select(self.ep);u=self.select(self.up);v=self.select(self.ev)
        def line(x,y,name,index=0,window=100):
            x=np.asarray(x);y=np.asarray(y,dtype=float);color=pg.intColor(index,hues=8)
            if self.raw.isChecked():self.plot.plot(x,y,pen=pg.mkPen(color,width=1),symbol='o' if len(x)<150 else None,symbolSize=4,name=name+' raw')
            if self.trend.isChecked() and len(y)>1:self.plot.plot(x,pd.Series(y).rolling(window,min_periods=1).mean().to_numpy(),pen=pg.mkPen(color,width=3),name=name+' trend')
        if mode==0:
            for i,stage in enumerate(('A','B','C')):
                a=e[e.complete&(e.stage==stage)];line(a.end_decision,a.episode_return,stage, i,3)
            a=e[~e.complete]
            if self.raw.isChecked():self.plot.plot(a.end_decision.to_numpy(),a.episode_return.to_numpy(),pen=None,symbol='x',symbolPen='r',name='Incomplete fragment')
        elif mode==1:line(g.global_decision,g.reward,'Environment reward')
        elif mode in (2,3):
            for i,k in enumerate(NAMES):line(g.global_decision,g[('cost_' if mode==2 else 'fraction_')+k],k,i)
        elif mode>=10:
            for i,k in enumerate(['evaluation_return'] if mode==10 else ['load_iae_deg_s','unload_iae_deg_s']):line(v.decisions,v[k],k,i,1)
        else:
            keys={4:['actor_delta','critic_delta'],5:['train/value_loss'],6:['train/explained_variance'],7:['train/approx_kl','train/clip_fraction'],8:['train/std'],9:['value_target_mean','value_target_std','advantage_mean','advantage_std']}[mode]
            for i,k in enumerate(keys):line(u.decisions,u[k],k,i,3)
        for n in (4096,8192):self.plot.addItem(pg.InfiniteLine(n,pen=pg.mkPen('gray',style=QtCore.Qt.PenStyle.DashLine)))
        self.plot.setTitle(self.metric_select.currentText()+' | seed '+self.seed.currentText()+' | '+self.stage.currentText());self.plot.showGrid(x=True,y=True,alpha=.2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');a=p.parse_args()
    app=QtWidgets.QApplication([])
    w=Analysis()
    w.show()
    if a.validate:
        def check():
            combinations=0
            for s in range(3):
                w.seed.setCurrentIndex(s)
                for stage in range(4):
                    w.stage.setCurrentIndex(stage)
                    for m in range(w.metric_select.count()):w.metric_select.setCurrentIndex(m);w.refresh();combinations+=1
            w.stage.setCurrentIndex(0);w.seed.setCurrentIndex(0);w.metric_select.setCurrentIndex(0);app.processEvents();w.grab().save(str(RUN/'analysis_gui.png'))
            write_json(RUN/'analysis_gui_validation.json',dict(combinations=combinations,seeds=3,checkpoints_per_seed=w.checkpoint.count(),read_only=True));w.close()
        QtCore.QTimer.singleShot(1000,check)
    app.exec()
if __name__=='__main__':main()
