"""Bounded spawn-process lifetime; reset always creates a new episode/process."""
from pathlib import Path
import time,json,multiprocessing as mp
from queue import Full
from ..model import ROOT,write_json
from .owner import owner_main
from .native_mirror import mirror_main

class Session:
    def __init__(self,base_config):
        self.context=mp.get_context('spawn');self.base=dict(base_config);self.folder=ROOT/'results/runs'/f'phase1_5_session_{time.time_ns()}';self.folder.mkdir(parents=True);self.episode=-1;self.history=[];self.owner=None;self.viewer=None;self.commands=None;self.telemetry=None;self.pending_reset=None;self.dropped_commands=0
    def start(self,controller=None):
        if self.owner is not None and self.owner.is_alive():raise RuntimeError('Old episode must archive before Reset')
        self.episode+=1;self.commands=self.context.Queue(64);self.telemetry=self.context.Queue(64);self.mirror=self.context.Queue(8);self.stop=self.context.Event();self.mirror_closed=self.context.Event()
        for q in [self.commands,self.telemetry,self.mirror]:q.cancel_join_thread()
        config=dict(self.base,controller=controller or self.base.get('controller','C0'),episode=self.episode,folder=str(self.folder/f'episode_{self.episode:03}'))
        self.current=config;self.owner=self.context.Process(target=owner_main,args=(config,self.commands,self.telemetry,self.mirror if config.get('native_viewer',True) else None,self.stop,self.mirror_closed),name='StewartPhysicalOwner');self.owner.start();self.viewer=None
        if config.get('native_viewer',True):self.viewer=self.context.Process(target=mirror_main,args=(config,self.mirror,self.stop,self.mirror_closed),name='StewartReadOnlyNativeMirror');self.viewer.start()
        self.history.append(dict(episode=self.episode,folder=config['folder'],controller=config['controller'],condition=config.get('condition','nominal'),owner_pid=self.owner.pid,mirror_pid=self.viewer.pid if self.viewer else None));self.save();return config
    def send(self,kind,payload=None,source='panel'):
        try:self.commands.put_nowait(dict(kind=kind,payload=payload or {},source=source));return True
        except Full:self.dropped_commands+=1;return False
    def request_reset(self,controller):
        self.pending_reset=controller
        if self.owner and self.owner.is_alive():self.send('reset',source='explicit-new-episode')
    def request_stop(self):
        self.pending_reset=None
        if self.owner:self.stop.set()
    def poll(self):
        if self.owner is None:return False
        if not self.owner.is_alive():
            self.stop.set()
            if self.viewer is not None and self.viewer.is_alive():return True
            self.history[-1].update(owner_exitcode=self.owner.exitcode,mirror_exitcode=self.viewer.exitcode if self.viewer else None,dropped_commands=self.dropped_commands);self.save()
            if self.pending_reset:
                controller=self.pending_reset;self.pending_reset=None;self.start(controller);return True
            return False
        if self.viewer is not None and not self.viewer.is_alive() and self.viewer.exitcode is not None:self.mirror_closed.set()
        return True
    def save(self):write_json(self.folder/'session.json',dict(episodes=self.history,reset_semantics='New owner process and new MjData only; no live qpos teleport'))
