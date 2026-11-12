import ast,json,hashlib
from pathlib import Path
import numpy as np,pytest,mujoco
from scipy.spatial.transform import Rotation
from src.model import ROOT,ids
from src.visualization.telemetry import key_command,Snapshot,SnapshotBuffer,TrainingFields
from src.visualization.motion import validate_target,target_spec,SmoothPlan
from src.visualization.owner import PhysicsOwner
from src.visualization.replay import ReplaySource

@pytest.fixture
def owner(tmp_path):
    x=PhysicsOwner(dict(folder=str(tmp_path/'episode'),pace=False,max_sim_seconds=.01,max_wall_seconds=10,condition='nominal',controller='C0'))
    yield x
    if not x.recorder.stream.closed:x.recorder.close(dict(test_only=True))

@pytest.mark.parametrize('key,payload',[('W',{'z_mm':.25}),('S',{'z_mm':-.25}),('A',{'roll_deg':.05}),('D',{'roll_deg':-.05}),('I',{'pitch_deg':.05}),('K',{'pitch_deg':-.05})])
def test_keys(key,payload):assert key_command(key)==('increment',payload)
def test_focus_and_repeat():
    assert key_command('W',False) is None;assert key_command('W',auto_repeat=True) is None
def test_slider_target_limits():
    assert validate_target(target_spec(1,.1,.1))==target_spec(1,.1,.1)
    for v in [target_spec(5),target_spec(4,.4),target_spec(float('nan')),dict(z_mm=1)]:
        with pytest.raises(ValueError):validate_target(v)
def test_target_rejection_preserves_reference(owner):
    old=owner.ref;owner.command(dict(kind='target',payload={'z_mm':900}));assert owner.pending is None;assert owner.ref is old;assert owner.failure is None
def test_smooth_full_chain_and_stroke(owner):
    p=SmoothPlan(owner.nominal,owner.geometry,owner.ref,target_spec(1,.1,0),owner.query)
    assert np.array_equal(p.at(0).q,owner.ref.q);assert max(abs(p.at(0).dq))<1e-12;assert max(abs(p.at(2).dq))<1e-10
    assert p.audit['minimum_covered_proxy_m']>=.005;assert p.audit['stroke_min_m']>=-.1;assert p.audit['stroke_max_m']<=.35
    np.testing.assert_allclose(p.at(2).p,p.p1,atol=1e-12)
def test_ik_rejection(owner,monkeypatch):
    import src.visualization.motion as mod
    class Bad:success=False
    monkeypatch.setattr(mod,'inverse_kinematics',lambda *a:Bad())
    owner.command(dict(kind='target',payload={'z_mm':1}));owner.maybe_plan();assert owner.failure.startswith('reference_preflight:');assert owner.protect_end==pytest.approx(.2)
def test_indices_state_and_force(owner):
    owner.step();s=owner.snapshot();qa,dof=ids(owner.model)
    np.testing.assert_array_equal(s.data['q'],owner.data.qpos[qa]);np.testing.assert_allclose(s.data['actuatorpos'],s.data['q'],atol=1e-14)
    np.testing.assert_array_equal(s.data['force'],owner.data.actuator_force[:6]);np.testing.assert_array_equal(s.data['f_cmd'],owner.data.ctrl[:6]);assert max(abs(owner.data.ctrl[:6]))<=400
def test_angles_from_actual_rotation(owner):
    from src.phase1.plant import platform_state
    owner.step();_,r,_=platform_state(owner.model,owner.data);np.testing.assert_allclose(owner.snapshot().data['rpy'],Rotation.from_matrix(r).as_euler('xyz',degrees=True))
def test_hold_and_home_no_teleport(owner):
    owner.step();q=owner.data.qpos.copy();owner.command(dict(kind='hold'));np.testing.assert_array_equal(q,owner.data.qpos);np.testing.assert_array_equal(owner.hold_ref.q,q[owner.qadr])
    owner.command(dict(kind='home'));np.testing.assert_array_equal(q,owner.data.qpos)
def test_pause_freezes_physics(owner):
    owner.command(dict(kind='pause'));owner.config['max_wall_seconds']=.03;owner.run();assert owner.n==0;assert owner.data.time==0
def test_controller_requires_new_episode(owner):
    owner.command(dict(kind='controller',payload={'name':'C2'}));assert owner.controller.name=='C0'
def test_failure_latched(owner):
    owner.hold('realistic_test_latch',True);q=owner.ref;owner.command(dict(kind='target',payload={'z_mm':1}));assert owner.pending is None;assert owner.ref is q;assert owner.failure
def test_no_rl_reward_before_step():
    with pytest.raises(ValueError):Snapshot(0,0,0,'ACTUAL STOCHASTIC TRAINING','RUNNING',{},TrainingFields(last_completed_reward=1))
    assert TrainingFields().last_completed_reward is None
def test_buffer_no_future_and_episode_clear():
    b=SnapshotBuffer(3)
    for i in range(5):b.append(Snapshot(0,i,i,'RECORDED STATE REPLAY','RUNNING',{}))
    assert len(b.rows)==3;assert all(s.time_s<=3 for s in b.recent(10,3));b.append(Snapshot(1,0,0,'RECORDED STATE REPLAY','HOLDING',{}));assert len(b.rows)==1
def test_display_modules_never_step():
    for name in ['live_dashboard.py','native_mirror.py','replay.py']:
        tree=ast.parse((ROOT/'src/visualization'/name).read_text());assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='mj_step' for x in ast.walk(tree))
def test_step_record_replay_exact(owner,monkeypatch):
    for _ in range(40):owner.step()
    owner.recorder.close(dict(test_only=True));r=ReplaySource(owner.config['folder']);s=r.at(0);np.testing.assert_allclose(s.data['qpos'],np.load(Path(owner.config['folder'])/'integration_states.npz')['states'][0,1:44],atol=1e-14);assert s.mode=='RECORDED STATE REPLAY';r.close()
def test_original_history_unchanged():
    old=json.loads((ROOT/'results/phase1_5_frozen_history.json').read_text(encoding='utf-8'))
    changed=[p for p,m in old['files'].items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=m['sha256']];assert changed==[]

def test_input_burst_bounded_pending_target(owner):
    for k in range(100):owner.command(dict(kind='target',payload=target_spec((k%10)/10,.05,0)))
    assert owner.n==0;assert owner.pending==target_spec(.9,.05,0);owner.maybe_plan();assert owner.active is not None

def test_recording_error_does_not_stop_physics(owner,monkeypatch):
    owner.recorder.enabled=True
    def broken(*a,**kw):raise RuntimeError('deliberate render exception')
    monkeypatch.setattr(mujoco,'Renderer',broken)
    for _ in range(80):owner.step()
    assert owner.n==80;assert owner.recorder.render_failures;assert owner.failure is None;assert not owner.recorder.enabled

def test_final_telemetry_error_still_archives(owner,monkeypatch):
    original=owner.publish
    def publish():
        if owner.n>=10:raise RuntimeError('deliberate display exception')
        return original()
    monkeypatch.setattr(owner,'publish',publish);owner.run();assert (Path(owner.config['folder'])/'summary.json').exists();assert owner.recorder.stream.closed

def test_force_command_and_actual_distinct_under_gain_mismatch(tmp_path):
    x=PhysicsOwner(dict(folder=str(tmp_path/'gain'),controller='C0',condition='asymmetric_gain',diagnostic_asymmetric=True,pace=False,max_sim_seconds=.01,max_wall_seconds=3))
    x.step();s=x.snapshot();assert not np.array_equal(s.data['f_cmd'],s.data['force']);np.testing.assert_allclose(s.data['force'],np.asarray(s.data['f_cmd'])*x.model.actuator_gainprm[:6,0]);x.recorder.close(dict(test_only=True))

def test_replay_history_causal_cached_and_image_no_steps(owner,monkeypatch):
    for _ in range(80):owner.step()
    owner.recorder.close(dict(test_only=True));r=ReplaySource(owner.config['folder']);s=r.at(1);a=r.history(.02);b=r.history(.02)
    assert a and all(x.time_s<=s.time_s+1e-10 and x.time_s>=s.time_s-.02-1e-10 for x in a);assert all(x is y for x,y in zip(a,b))
    def forbidden(*a,**kw):raise AssertionError('Replay must never integrate')
    monkeypatch.setattr(mujoco,'mj_step',forbidden);assert r.image(s).shape==(720,960,3);r.close()

def test_delivered_videos_causal_and_physical():
    audit=json.loads((ROOT/'results/phase1_5_video_frame_audit.json').read_text(encoding='utf-8'))
    for item in audit.values():
        assert item['all_frames_decoded'];assert item['resolution']==[1920,1080];assert item['fps']==25;assert item['no_future_curves'];assert item['physical_qpos_range_max']>0;assert item['changed_decoded_frame_pairs']>0
