"""Bounded-memory live journal. RGB always comes from the actual owner MjData."""
import csv,json,hashlib
import numpy as np,mujoco,imageio.v2 as imageio
from ..recording import Recorder
from ...model import write_json
from ...rendering import camera

class StreamingRecorder(Recorder):
    @classmethod
    def adopt(cls,old,chunk_steps=2000,video_seconds=60):
        obj=cls.__new__(cls);obj.__dict__.update(old.__dict__)
        obj.chunk_steps=int(chunk_steps);obj.video_frames=max(1,round(video_seconds*25))
        obj.total_rows=0;obj.chunk_count=0;obj.peak_buffered_states=0;obj.segment=-1;obj.segment_frame=0;obj.closed=False
        (obj.folder/'states').mkdir();(obj.folder/'videos').mkdir()
        obj.chunks=(obj.folder/'state_chunks.jsonl').open('w',encoding='utf-8')
        return obj
    def append(self,row,state):
        if self.csv is None:self.csv=csv.DictWriter(self.stream,fieldnames=list(row));self.csv.writeheader()
        self.csv.writerow(row);self.states.append(state.copy());self.times.append(row['time_s']);self.total_rows+=1
        self.peak_buffered_states=max(self.peak_buffered_states,len(self.states))
        if len(self.states)>=self.chunk_steps:self.flush_chunk()
        if self.total_rows%500==0:self.stream.flush()
    def flush_chunk(self):
        if not self.states:return
        name=f'states/state_{self.chunk_count:06}.npz';first=self.total_rows-len(self.states)
        np.savez_compressed(self.folder/name,states=np.asarray(self.states),time=np.asarray(self.times),state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),first_state_index=first)
        self.chunks.write(json.dumps(dict(file=name,first_state_index=first,rows=len(self.states),first_time_s=self.times[0],last_time_s=self.times[-1]))+'\n');self.chunks.flush()
        self.states.clear();self.times.clear();self.chunk_count+=1
    def finish_video(self):
        if self.writer is not None:
            writer=self.writer;self.writer=None
            try:writer.close()
            except Exception as exc:self.render_failures.append(dict(stage='encoder_close',error=repr(exc)))
    def capture(self,model,data,snapshot,state_index):
        if not self.enabled:return
        try:
            if self.renderer is None:
                model.vis.global_.offwidth=1216;model.vis.global_.offheight=912;self.renderer=mujoco.Renderer(model,width=1216,height=912)
            if self.writer is None or self.segment_frame>=self.video_frames:
                self.finish_video();self.segment+=1;self.segment_frame=0;self.video_file=f'videos/actual_rgb_{self.segment:06}.mp4'
                self.writer=imageio.get_writer(self.folder/self.video_file,fps=25,codec='libx264',quality=8,macro_block_size=16)
            self.renderer.update_scene(data,camera=camera());self.writer.append_data(self.renderer.render().copy())
            item=dict(frame=self.nframes,video_file=self.video_file,segment_frame=self.segment_frame,state_index=state_index,step=snapshot.step,time_s=snapshot.time_s,episode=snapshot.episode,controller=snapshot.data['controller'],status=snapshot.status,qpos_sha256=hashlib.sha256(data.qpos.tobytes()).hexdigest(),source='ACTUAL EXECUTION CAPTURE',source_mode=snapshot.mode)
            self.frames.write(json.dumps(item)+'\n');self.frames.flush();self.nframes+=1;self.segment_frame+=1
        except Exception as exc:
            self.render_failures.append(dict(time_s=float(data.time),error=repr(exc)));self.enabled=False;self.finish_video();self.event(dict(type='recording_error',time_s=float(data.time),reason=repr(exc)))
    def close(self,summary):
        if self.closed:return
        self.closed=True
        try:self.flush_chunk()
        finally:
            self.finish_video()
            if self.renderer is not None:
                try:self.renderer.close()
                except Exception as exc:self.render_failures.append(dict(stage='renderer_close',error=repr(exc)))
            for f in (self.stream,self.events,self.frames,self.chunks):f.close()
        write_json(self.folder/'summary.json',dict(**summary,video_frames=self.nframes,recording_errors=self.render_failures,raw_rows=self.total_rows,state_spec='mjSTATE_INTEGRATION',source_mode='DYNAMIC CONTROL DEMONSTRATION',storage='chunked_npz',state_chunks=self.chunk_count,peak_buffered_states=self.peak_buffered_states))

def iter_state_chunks(folder):
    from pathlib import Path
    folder=Path(folder)
    with (folder/'state_chunks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                with np.load(folder/json.loads(line)['file']) as z:yield z['states'],z['time']

class SegmentReader:
    def __init__(self,folder,frames):self.folder=folder;self.frames=frames;self.name=None;self.reader=None
    def get_data(self,index):
        item=self.frames[index]
        if item['video_file']!=self.name:
            self.close();self.name=item['video_file'];self.reader=imageio.get_reader(self.folder/self.name)
        return self.reader.get_data(item['segment_frame'])
    def close(self):
        if self.reader is not None:self.reader.close();self.reader=None

def enable_segmented_replay(replay):
    file=replay.folder/'frames.jsonl'
    if not file.exists():return
    with file.open(encoding='utf-8') as stream:rows=[json.loads(x) for x in stream if x.strip()]
    if rows and 'video_file' in rows[0]:
        replay.frame_rows=rows;replay.reader=SegmentReader(replay.folder,rows);replay.times=np.array([r['time_s'] for r in rows]);replay.origin='ACTUAL EXECUTION CAPTURE (SEGMENTED)'
