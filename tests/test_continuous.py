import importlib.util,json,sys,threading,time
from pathlib import Path
import numpy as np,pytest
from src.model import ROOT
from src.visualization.enhanced.owner import EnhancedOwner
from src.visualization.enhanced.runtime import budget_reason
from src.visualization.enhanced.streaming import iter_state_chunks,StreamingRecorder,enable_segmented_replay
from src.visualization.enhanced.events import EventTail

def cli():
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('continuous_entry',ROOT/'scripts/teleop_stewart_enhanced.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_default_no_deadline():
    args=cli().parse_args([])
    assert args.max_sim_seconds is None and args.max_wall_seconds is None
    assert budget_reason({},10**15,10**15) is None
    assert budget_reason({'max_sim_seconds':1},1000,0)=='budget_complete'
    assert budget_reason({'max_wall_seconds':1},0,1)=='wall_budget_complete'

@pytest.mark.parametrize('args',[['--max-sim-seconds','nan'],['--max-wall-seconds','inf'],['--max-sim-seconds','0'],['--state-chunk-steps','0'],['--headless']])
def test_invalid_cli(args):
    with pytest.raises(SystemExit):cli().parse_args(args)

def make(folder,**kw):
    return EnhancedOwner(dict(folder=str(folder),condition='nominal',record=False,pace=False,stream_logs=True,state_chunk_steps=64,max_sim_seconds=None,max_wall_seconds=20,**kw))

@pytest.mark.parametrize('channel,reason',[('stop','user_quit'),('mirror_closed','native_viewer_closed')])
def test_real_owner_window_stop(tmp_path,channel,reason):
    owner=make(tmp_path/'live');flag=threading.Event();setattr(owner,channel,flag)
    original=owner.step
    def step():
        original()
        if owner.n==135:flag.set()
    owner.step=step
    assert owner.run()==reason
    summary=json.loads((tmp_path/'live/summary.json').read_text())
    assert summary['raw_rows']==135 and summary['state_chunks']==3 and summary['peak_buffered_states']==64

def test_chunked_physics_bitwise_equal(tmp_path):
    base=dict(condition='nominal',record=False,pace=False,max_sim_seconds=2.05,max_wall_seconds=30,events=[dict(at_step=900,kind='hold')])
    old=EnhancedOwner(dict(base,folder=str(tmp_path/'old')));old.run()
    new=EnhancedOwner(dict(base,folder=str(tmp_path/'new'),stream_logs=True,state_chunk_steps=64));new.run()
    with np.load(tmp_path/'old/integration_states.npz') as z:expected=z['states'];times=z['time']
    chunks=list(iter_state_chunks(tmp_path/'new'))
    np.testing.assert_array_equal(np.concatenate([x[0] for x in chunks]),expected)
    np.testing.assert_array_equal(np.concatenate([x[1] for x in chunks]),times)
    import pandas as pd
    a=pd.read_csv(tmp_path/'old/physics.csv.gz');b=pd.read_csv(tmp_path/'new/physics.csv.gz')
    assert a.equals(b)

def test_pause_and_failure_keep_exit_responsive(tmp_path):
    for mode in ('paused','failed'):
        owner=make(tmp_path/mode);owner.stop=threading.Event()
        if mode=='paused':owner.paused=True
        else:owner.failure='test_latched';owner.protect_end=0
        thread=threading.Thread(target=lambda:(time.sleep(.05),owner.stop.set()));thread.start();owner.run();thread.join()
        assert owner.n==0 and owner.reason=='user_quit'

def test_live_event_tail_bounded_and_partial_line(tmp_path):
    file=tmp_path/'events.jsonl'
    file.write_text(''.join(json.dumps(dict(type='hold',time_s=i))+'\n' for i in range(2000)),encoding='utf-8')
    tail=EventTail(32);tail.read(tmp_path);rows=tail.read(tmp_path)
    assert len(rows)==32 and rows[-1]['time_s']==1999
    with file.open('a') as f:f.write('{"type":"hold"')
    assert tail.read(tmp_path)==rows
    with file.open('a') as f:f.write(',"time_s":2000}\n')
    assert tail.read(tmp_path)[-1]['time_s']==2000

def test_actual_rgb_segments(tmp_path):
    owner=make(tmp_path/'rgb');owner.recorder.enabled=True;owner.recorder.video_frames=2
    for _ in range(200):owner.step()
    owner.recorder.close(dict(test_only=True))
    import imageio.v2 as imageio
    rows=[json.loads(x) for x in (tmp_path/'rgb/frames.jsonl').read_text().splitlines()]
    assert len(rows)==5 and len(set(r['video_file'] for r in rows))==3
    assert owner.recorder.render_failures==[]
    from types import SimpleNamespace
    replay=SimpleNamespace(folder=tmp_path/'rgb');enable_segmented_replay(replay)
    for row in rows:
        frame=replay.reader.get_data(row['frame']);assert frame.shape[:2]==(912,1216)
        assert row['step']==row['state_index']+1
    replay.reader.close()

def test_recording_error_keeps_physics_alive(tmp_path,monkeypatch):
    owner=make(tmp_path/'error');owner.recorder.enabled=True
    import src.visualization.enhanced.streaming as module
    def fail(*a,**kw):raise RuntimeError('test renderer failure')
    monkeypatch.setattr(module.mujoco,'Renderer',fail)
    for _ in range(80):owner.step()
    assert owner.n==80 and not owner.recorder.enabled and owner.recorder.render_failures
    owner.recorder.close(dict(test_only=True))
