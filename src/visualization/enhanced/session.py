from pathlib import Path
import time,multiprocessing as mp
from ..session import Session
from ...model import ROOT
from .owner import owner_main
from .mirror import mirror_main
class EnhancedSession(Session):
    def __init__(self,config):
        # Same lifecycle as frozen Session, with opt-in process targets and distinct output root.
        self.context=mp.get_context('spawn');self.base=dict(config);self.folder=ROOT/'results/runs'/f'phase1_5_1_session_{time.time_ns()}';self.folder.mkdir(parents=True);self.episode=-1;self.history=[];self.owner=None;self.viewer=None;self.commands=None;self.telemetry=None;self.pending_reset=None;self.dropped_commands=0
    def start(self,controller=None):
        if self.owner is not None and self.owner.is_alive():raise RuntimeError('Archive old episode before Reset')
        self.episode+=1;self.commands=self.context.Queue(64);self.telemetry=self.context.Queue(64);self.mirror=self.context.Queue(8);self.stop=self.context.Event();self.mirror_closed=self.context.Event()
        for q in [self.commands,self.telemetry,self.mirror]:q.cancel_join_thread()
        c=dict(self.base,controller=controller or self.base.get('controller','C0'),episode=self.episode,folder=str(self.folder/f'episode_{self.episode:03}'));self.current=c
        self.owner=self.context.Process(target=owner_main,args=(c,self.commands,self.telemetry,self.mirror if c.get('native_viewer',True) else None,self.stop,self.mirror_closed));self.owner.start();self.viewer=None
        if c.get('native_viewer',True):self.viewer=self.context.Process(target=mirror_main,args=(c,self.mirror,self.stop,self.mirror_closed));self.viewer.start()
        self.history.append(dict(episode=self.episode,folder=c['folder'],controller=c['controller'],owner_pid=self.owner.pid,mirror_pid=self.viewer.pid if self.viewer else None));self.save();return c
