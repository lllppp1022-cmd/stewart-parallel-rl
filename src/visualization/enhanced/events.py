"""Read real event journal; never infer commands from motion curves."""
import json
from pathlib import Path
KEY_TYPES={'input','target_queued','target_rejected','rejected','plan_accepted','preset_started','motion_started','motion_completed','motion_aborted','target_superseded','target_no_motion','hold','protect_hold','pause_state','auto_resume','episode_started'}
def description(e):
    typ=e['type'];kind=e.get('kind','');payload=e.get('payload',{})
    if typ=='input':
        if kind=='increment':return 'SUBMITTED '+', '.join(f'{k} {v:+g}' for k,v in payload.items())
        return 'SUBMITTED '+kind+' '+(str(payload) if payload else '')
    return {'target_queued':'QUEUED (not preflighted)','plan_accepted':'PREFLIGHT PASSED','motion_started':'EXECUTING','motion_completed':'COMPLETED','motion_aborted':'ABORTED','target_superseded':'SUPERSEDED (not independently executed)','target_rejected':'REJECTED','rejected':'REJECTED','hold':'Hold activated','protect_hold':'Protective hold','pause_state':'Pause' if e.get('paused') else 'Resume','auto_resume':'Resume (scheduled)','episode_started':'Episode reset / initial Home','target_no_motion':'UNCHANGED target / no new motion','preset_started':'PREFLIGHTED PRESET'}.get(typ,typ)
def read_events(folder):
    p=Path(folder)/'events.jsonl'
    if not p.exists():return []
    lines=p.read_text(encoding='utf-8').splitlines();out=[]
    for i,line in enumerate(lines):
        try:e=json.loads(line)
        except json.JSONDecodeError:continue # concurrently written final line
        if e.get('type') in KEY_TYPES:out.append(dict(e,journal_index=i,label=description(e)))
    return out
def visible_events(events,t,window=10):return [e for e in events if t-window<=e.get('time_s',0)<=t+1e-10]
def latest_event(events,t):
    candidates=[e for e in events if e.get('time_s',0)<=t+1e-10 and e['type'] not in ('target_queued','target_superseded')]
    return candidates[-1] if candidates else None

class EventTail:
    """Incremental live reader with a bounded display cache; disk journal stays complete."""
    def __init__(self,limit=512):
        from collections import deque
        self.rows=deque(maxlen=limit);self.folder=None;self.offset=0;self.index=0
    def read(self,folder):
        folder=Path(folder)
        if folder!=self.folder:self.folder=folder;self.offset=0;self.index=0;self.rows.clear()
        p=folder/'events.jsonl'
        if not p.exists():return list(self.rows)
        with p.open('rb') as stream:
            stream.seek(self.offset)
            for _ in range(1024):
                start=stream.tell();line=stream.readline()
                if not line.endswith(b'\n'):self.offset=start;break
                self.offset=stream.tell()
                try:e=json.loads(line)
                except (ValueError,UnicodeError):continue
                if e.get('type') in KEY_TYPES:self.rows.append(dict(e,journal_index=self.index,label=description(e)))
                self.index+=1
        return list(self.rows)
