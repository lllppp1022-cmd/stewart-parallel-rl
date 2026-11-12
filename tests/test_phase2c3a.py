import json,inspect
import numpy as np,pytest
from src.model import ROOT,sha
from src.workspace.geometry import ExpandedGeometryQuery,inspect_pose,check_native_joint_limits,exclusion
from src.workspace.reference import LargeReference
from src.workspace.execution import LargeOwner
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.jacobian import finite_difference_jacobian,actuator_jacobian
from src.rl.stewart_residual_env import owner_config
from src.control.phase2c0_experiment import PhysicalOwner

RUN=ROOT/'results/runs/phase2c3a_20261103_v1'
def config():return json.loads((ROOT/'configs/phase2c3a_large_workspace.json').read_text(encoding='utf-8'))

@pytest.mark.parametrize('amp',[.03,.05])
def test_large_reference_full_chain_and_units(amp):
    m=create_plant();q=ExpandedGeometryQuery();r=LargeReference(m,amp,10.,RUN/'references',q)
    assert r.at(11.5).p[2]-r.geometry.home_position[2]==pytest.approx(amp)
    for t in [0.,r.duration]:
        assert abs(r.scalar(t)[0])<1e-14 and abs(r.scalar(t)[1])<1e-14 and abs(r.scalar(t)[2])<1e-14
    assert r.audit['max_off_grid_stroke_error_m']<1e-7 and r.audit['max_static_force_N']<400
    with pytest.raises(ValueError):LargeReference(m,.1,10,RUN/'references',q)

def test_query_independent_and_missing_limits_explicit():
    m=create_plant();g=StewartGeometry(m);query=ExpandedGeometryQuery();row,sol=inspect_pose(m,g,query,.05)
    assert row['simulation_geometric_candidate'] and row['COLLISION_COVERAGE_INCOMPLETE'] and row['passive_limit_margin_rad'] is None and row['unlimited_passive_count']==30
    j,_=actuator_jacobian(g,g.home_position+[0,0,.05],g.home_rotation,sol);fd=finite_difference_jacobian(g,g.home_position+[0,0,.05],g.home_rotation,sol);np.testing.assert_allclose(j,fd,atol=1e-4)
    assert exclusion('legbot_1','legbot_2') is None and exclusion('cardan2_1','legup_4') is None and exclusion('legbot_1','legup_1')
    qp=np.zeros(m.nq);qp[-4]=1;before=qp.copy();query.evaluate(qp);np.testing.assert_array_equal(qp,before)
    qp[int(m.joint('joint_lu_1').qposadr[0])]=.4;assert not check_native_joint_limits(m,qp)['valid']
    with pytest.raises(ValueError):query.evaluate(np.full(m.nq,np.nan))

def test_independent_configuration_preserves_physics_and_old_envelope():
    from src.phase1.trajectory import protocol
    c=config();old=protocol();assert old['actual_z_limit_m']==.005 and c['actual_z_bounds_m']==[-.002,.052]
    for k in ['weld_translation_limit_m','weld_angle_limit_rad','actual_xy_limit_m','actual_roll_pitch_limit_deg','actual_yaw_limit_deg','jacobian_condition_limit']:assert c[k]==old[k]
    m=create_plant();np.testing.assert_array_equal(m.actuator_forcerange[:6],np.tile([-400,400],(6,1)))

@pytest.mark.parametrize('method',['C0','C4','fourDlead'])
def test_zero_trajectory_matches_frozen_owner(tmp_path,method):
    task=dict(id='zero_equivalence',tau_s=0.,Fz_N=0.,on_step=1200,off_step=1800);q=ExpandedGeometryQuery();r=LargeReference(create_plant(),0.,10.,RUN/'references',q);a=owner_config(tmp_path/'new',task);b=owner_config(tmp_path/'old',task)
    for c in [a,b]:
        if method=='C0':c.update(controller='C0',yaw_policy=dict(kind='zero4'))
        if method=='fourDlead':c['diagnostic']=dict(kind='four_d_lead')
    new=LargeOwner(a,r,q,config());old=PhysicalOwner(b)
    try:
        for _ in range(300):
            old.step();new.step();np.testing.assert_array_equal(old.data.qpos,new.data.qpos);np.testing.assert_array_equal(old.data.qvel,new.data.qvel);np.testing.assert_array_equal(old.data.ctrl,new.data.ctrl)
        assert not new.failure
    finally:new.recorder.close(dict(test='home_equivalence'));old.recorder.close(dict(test='home_equivalence'))

def test_step_uses_frozen_integration_no_live_qpos_write():
    import src.dynamics.execution_hook as h
    assert 'mj_step' in inspect.getsource(h.instrumented_step.__globals__['DisturbanceOwner'].step)
    text=inspect.getsource(LargeOwner.step);assert '.qpos[' not in text and 'mj_forward(' not in text

def test_failure_hold_records_real_states_without_teleport(tmp_path):
    task=dict(id='unit_hold',tau_s=0.,Fz_N=0.,on_step=1200,off_step=1800);q=ExpandedGeometryQuery();r=LargeReference(create_plant(),0.,10.,RUN/'references',q);o=LargeOwner(owner_config(tmp_path/'hold',task),r,q,config())
    o.step();o.hold('UNIT_INJECTED_FAILURE_NOT_A_RESEARCH_RESULT',failed=True)
    while o.data.time<o.protect_end-1e-10:o.step()
    assert o.n==201 and o.failure and len(o.recorder.states)==201
    o.recorder.close(dict(physics_steps=o.n,complete=False,failure=o.failure))
    event=(tmp_path/'hold/events.jsonl').read_text(encoding='utf-8');assert 'actual_qpos' in event and 'protect_hold' in event
