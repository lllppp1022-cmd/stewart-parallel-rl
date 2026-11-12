"""Full physics journal + optional actual RGB; graphics errors cannot add steps."""
from pathlib import Path
import csv,gzip,json,time,hashlib
import numpy as np,mujoco,imageio.v2 as imageio
from ..model import write_json
from ..rendering import camera

class Recorder:
    def __init__(self,folder,enabled=False):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False);self.enabled=enabled;self.stream=gzip.open(self.folder/'physics.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1);self.csv=None;self.events=(self.folder/'events.jsonl').open('w',encoding='utf-8');self.frames=(self.folder/'frames.jsonl').open('w',encoding='utf-8');self.writer=None;self.renderer=None;self.nframes=0;self.states=[];self.times=[];self.errors=[];self.render_failures=[]
    def event(self,event):self.events.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');self.events.flush()
    def append(self,row,state):
        if self.csv is None:self.csv=csv.DictWriter(self.stream,fieldnames=list(row));self.csv.writeheader()
        self.csv.writerow(row);self.states.append(state.copy());self.times.append(row['time_s'])
        if len(self.times)%500==0:self.stream.flush()
    def capture(self,model,data,snapshot,state_index):
        if not self.enabled:return
        try:
            if self.renderer is None:
                model.vis.global_.offwidth=1216;model.vis.global_.offheight=912;self.renderer=mujoco.Renderer(model,width=1216,height=912);self.writer=imageio.get_writer(self.folder/'actual_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
            self.renderer.update_scene(data,camera=camera());rgb=self.renderer.render().copy();self.writer.append_data(rgb)
            item=dict(frame=self.nframes,state_index=state_index,step=snapshot.step,time_s=snapshot.time_s,episode=snapshot.episode,controller=snapshot.data['controller'],status=snapshot.status,qpos_sha256=hashlib.sha256(data.qpos.tobytes()).hexdigest(),source='ACTUAL EXECUTION CAPTURE',source_mode=snapshot.mode)
            self.frames.write(json.dumps(item)+'\n');self.frames.flush();self.nframes+=1
        except Exception as exc:
            self.render_failures.append(dict(time_s=float(data.time),error=repr(exc)));self.enabled=False;self.event(dict(type='recording_error',time_s=float(data.time),reason=repr(exc)))
    def close(self,summary):
        self.stream.close();self.events.close();self.frames.close()
        if self.writer:
            try:self.writer.close()
            except Exception as exc:self.render_failures.append(dict(stage='encoder_close',error=repr(exc)))
        if self.renderer:
            try:self.renderer.close()
            except Exception as exc:self.render_failures.append(dict(stage='renderer_close',error=repr(exc)))
        np.savez_compressed(self.folder/'integration_states.npz',states=np.asarray(self.states),time=np.asarray(self.times),state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION))
        write_json(self.folder/'summary.json',dict(**summary,video_frames=self.nframes,recording_errors=self.render_failures,raw_rows=len(self.times),state_spec='mjSTATE_INTEGRATION',source_mode='DYNAMIC CONTROL DEMONSTRATION'))
