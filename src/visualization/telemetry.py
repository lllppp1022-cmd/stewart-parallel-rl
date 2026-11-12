"""Plain copy-only telemetry; no MuJoCo, GUI or learning imports."""
from dataclasses import dataclass,field
from collections import deque
from queue import Full,Empty
import numpy as np

MODES={'DYNAMIC CONTROL DEMONSTRATION','ACTUAL STOCHASTIC TRAINING','DETERMINISTIC POLICY EVALUATION','RECORDED STATE REPLAY'}
@dataclass(frozen=True)
class TrainingFields:
    decision_step:int|None=None
    policy_version:str|None=None
    actor_mean:tuple|None=None
    sampled_action:tuple|None=None
    applied_action:tuple|None=None
    last_completed_reward:float|None=None
    reward_after_completed_step:bool=False
    terminal_reason:str|None=None
    def validate(self):
        if self.last_completed_reward is not None and not self.reward_after_completed_step:raise ValueError('Reward is publishable only AFTER a completed environment step')
@dataclass(frozen=True)
class Snapshot:
    episode:int
    step:int
    time_s:float
    mode:str
    status:str
    data:dict
    training:TrainingFields=field(default_factory=TrainingFields)
    def __post_init__(self):
        if self.mode not in MODES:raise ValueError('Unknown source mode')
        self.training.validate()

class SnapshotBuffer:
    def __init__(self,capacity=2200):self.rows=deque(maxlen=capacity);self.episode=None
    def append(self,snapshot):
        if self.episode!=snapshot.episode:self.rows.clear();self.episode=snapshot.episode
        if not self.rows or snapshot.step!=self.rows[-1].step:self.rows.append(snapshot)
    def recent(self,seconds,at_time=None):
        if not self.rows:return []
        t=self.rows[-1].time_s if at_time is None else at_time
        return [s for s in self.rows if t-seconds<=s.time_s<=t+1e-10]

def publish_latest(queue,value):
    if queue is None:return
    try:queue.put_nowait(value)
    except Full:
        try:queue.get_nowait()
        except Empty:pass
        try:queue.put_nowait(value)
        except Full:pass

KEYS={'W':('increment',{'z_mm':.25}),'S':('increment',{'z_mm':-.25}),'A':('increment',{'roll_deg':.05}),'D':('increment',{'roll_deg':-.05}),'I':('increment',{'pitch_deg':.05}),'K':('increment',{'pitch_deg':-.05}),'R':('home',{}),'SPACE':('hold',{}),'P':('pause',{}),'V':('camera',{}),'Q':('quit',{}),'ESCAPE':('quit',{})}
def key_command(name,control_panel_focused=True,auto_repeat=False):
    if not control_panel_focused or auto_repeat:return None
    return KEYS.get(name.upper())
