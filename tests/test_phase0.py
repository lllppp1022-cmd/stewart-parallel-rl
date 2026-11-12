import json,subprocess,ast,sys
import numpy as np,pandas as pd,mujoco,pytest
from scipy.spatial.transform import Rotation
from src.model import *
from src.kinematics.stewart_geometry import StewartGeometry,valid_pose
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.kinematics.jacobian import actuator_jacobian,finite_difference_jacobian,singularity_diagnostics,chord_jacobian
from src.audit_control import AuditController,advance,scratch_reference_data

@pytest.fixture(scope='module')
def model():return load_model()
@pytest.fixture(scope='module')
def geometry(model):return StewartGeometry(model)
@pytest.fixture(scope='module')
def home(geometry):return inverse_kinematics(geometry,geometry.home_position,geometry.home_rotation)

def test_model_native_dimensions(model):assert (model.nq,model.nv,model.nu,model.nbody,model.neq)==(43,42,12,45,6)
def test_joint_mapping_and_force_transmission(model):
    q,v=ids(model);assert len(set(q))==len(set(v))==6
    assert all(model.joint(f'joint_lu_{i}').type[0]==2 for i in range(1,7))
    assert np.all(model.actuator_gear[:6,0]==1) and np.all(model.actuator_gainprm[:6,0]==1) and np.all(model.actuator_biasprm[:6]==0)
def test_license_and_upstream_unchanged():
    assert 'Copyright (c) 2026 vr-cuongvu' in (UPSTREAM/'LICENSE').read_text()
    assert subprocess.check_output(['git','status','--porcelain'],cwd=UPSTREAM,text=True)==''
def test_closed_chain_initial_residual(model):
    d=mujoco.MjData(model);mujoco.mj_forward(model,d);assert max(site_gap(model,d))<1e-6 and max(abs(equality_residual(d)))<1e-6
    assert np.all(model.eq_type==int(mujoco.mjtEq.mjEQ_WELD))
def test_geometry_does_not_substitute_readme(geometry):assert np.max(abs(geometry.base_points-geometry.readme_base))>.1
def test_ik_home_and_travel(geometry,home):assert home.success and home.max_position_error<1e-9 and np.max(abs(home.strokes))<1e-6 and home.minimum_stroke_margin>.09
def test_ik_unreachable_rejected(geometry,home):assert not inverse_kinematics(geometry,geometry.home_position+[0,0,10],geometry.home_rotation,home.chain_q).success
@pytest.mark.parametrize('p,r',[(np.full(3,np.nan),np.eye(3)),(np.zeros(2),np.eye(3)),(np.zeros(3),np.zeros((3,3)))])
def test_bad_pose_rejected(p,r):
    with pytest.raises(ValueError):valid_pose(p,r)
def test_independent_fk_matches_mujoco(model,geometry,home):
    d=scratch_reference_data(model,geometry,home,geometry.home_position,geometry.home_rotation)
    for i in range(6):assert np.allclose(geometry.forward_leg(i,home.chain_q[i])[:3,3],d.body(f'cardan1_{i+13}').xpos,atol=1e-12)
def test_actual_stroke_jacobian_fd(geometry,home):
    j,_=actuator_jacobian(geometry,geometry.home_position,geometry.home_rotation,home);fd=finite_difference_jacobian(geometry,geometry.home_position,geometry.home_rotation,home,1e-5);assert np.max(abs(j-fd))<1e-7
def test_chord_jacobian_fd(geometry):
    p=geometry.home_position;r=geometry.home_rotation;j=chord_jacobian(geometry,p,r);eps=1e-6;fd=[]
    for k in range(6):
        dp=np.eye(3)[k]*eps if k<3 else np.zeros(3);rv=np.eye(3)[k-3]*eps if k>=3 else np.zeros(3)
        fd.append((geometry.attachment_chord_lengths(p+dp,Rotation.from_rotvec(rv).as_matrix()@r)-geometry.attachment_chord_lengths(p-dp,Rotation.from_rotvec(-rv).as_matrix()@r))/(2*eps))
    assert np.max(abs(j-np.array(fd).T))<1e-7
def test_singularity_and_nonsingularity(geometry,home):
    j,_=actuator_jacobian(geometry,geometry.home_position,geometry.home_rotation,home);assert not singularity_diagnostics(j)['near_singular'];assert singularity_diagnostics(np.zeros((6,6)))['near_singular']
def test_controller_never_uses_disturbance_channels(model):
    c=AuditController(model);d=mujoco.MjData(model);mujoco.mj_forward(model,d);u,_=c.command(d,0);assert np.all(u[6:]==0) and np.max(abs(u))<400 and np.min(u[:6])>50
def test_scratch_ik_does_not_modify_live(model,geometry,home):
    d=mujoco.MjData(model);q=d.qpos.copy();scratch_reference_data(model,geometry,home,geometry.home_position,geometry.home_rotation);assert np.array_equal(q,d.qpos) and d.time==0
def test_real_single_leg_force_changes_motion(model):
    histories=[]
    for force in (0,2):
        d=mujoco.MjData(model);c=AuditController(model);mujoco.mj_forward(model,d)
        for k in range(200):
            u,_=c.command(d,d.time+.002);u[0]+=force;d.ctrl[:]=u;mujoco.mj_step(model,d)
        histories.append(d.qpos.copy())
    assert np.linalg.norm(histories[1]-histories[0])>1e-6
def test_zero_input_is_finite_but_not_home(model):
    d=mujoco.MjData(model)
    for _ in range(1000):mujoco.mj_step(model,d)
    mujoco.mj_forward(model,d);assert np.all(np.isfinite(d.qpos)) and d.body('upper').xpos[2]<1.02 and all(w.number==0 for w in d.warning)
def test_one_millisecond_home_real_dynamics():
    m=load_model();m.opt.timestep=.001;d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=AuditController(m);rows=[advance(m,d,c,k) for k in range(1,501)]
    assert rows[-1]['position_error_m']<1e-5 and max(r['max_site_angle_rad'] for r in rows)<1e-4 and d.time==pytest.approx(.5)
def test_no_live_qpos_assignments_in_executor():
    from src import audit_control
    import inspect
    tree=ast.parse(inspect.getsource(audit_control.advance))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            assert all('qpos' not in ast.unparse(t) for t in targets)
    assert 'mujoco.mj_step' in inspect.getsource(audit_control.advance)
def test_full_raw_logs_and_frozen_failures():
    summary=pd.read_csv(ROOT/'results/phase0_dynamics_check.csv');assert len(summary)==9 and summary.steps.sum()==22500 and summary.numerical_stability.all()
    motion=summary[summary.scenario.isin(['home','z_up','z_down','roll_pos','roll_neg','pitch_pos','pitch_neg'])];assert len(motion)==7 and (motion.stable_small_motion==False).all()
    for row in summary.to_dict('records'):
        df=pd.read_csv(ROOT/row['log']);assert len(df)==2500 and len(df.columns)>150 and np.all(np.isfinite(df.select_dtypes('number')))
        assert df['all_ctrl_7'].eq(0).all() and df['all_ctrl_12'].eq(0).all()
def test_video_states_and_source_metadata():
    table=pd.read_csv(ROOT/'results/phase0_video_frames.csv');assert len(table)==450 and table.groupby('scenario').size().eq(50).all()
    assert (ROOT/'assets/videos/phase0_simulation.mp4').stat().st_size>10000
def test_isolated_environment_no_learning_packages():
    import importlib.util
    assert 'stewart_rl' in sys.executable and importlib.util.find_spec('torch') is None and importlib.util.find_spec('stable_baselines3') is None
def test_timestep_sensitivity_did_not_relax_gate():
    table=pd.read_csv(ROOT/'results/phase0_timestep_sensitivity.csv');assert len(table)==14 and table.passed_unchanged_criteria.all() and table.steps.sum()==105000
    assert table.max_weld_angle_rad.max()<1e-4
def test_actual_video_frame_state_hashes():
    validation=json.loads((ROOT/'results/phase0_video_validation.json').read_text(encoding='utf-8'));assert validation['state_log_frame_matches']==450 and validation['frames']==450 and validation['duration_s']==45
