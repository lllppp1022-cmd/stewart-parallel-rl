import json,copy,inspect
import numpy as np,pytest
from src.model import ROOT
from src.control.dynamic_reference_feedforward import DynamicReferenceFeedforward
from src.control.large_motion_feedforward_wrapper import LargeMotionFeedforwardWrapper,ReferenceBudgetInjection
from src.workspace.feedforward_study import RUN,StudyReference,make_owner
from src.workspace.geometry import ExpandedGeometryQuery
from src.phase1.plant import create_plant

@pytest.mark.parametrize('T',[4.,6.,8.,10.])
def test_reference_derivatives_units_and_stillness(T):
    r=StudyReference(create_plant(),.05,T,RUN/'references',ExpandedGeometryQuery());ff=DynamicReferenceFeedforward(570.,50.)
    assert r.scalar(1+T)[0]==pytest.approx(.05)
    for t in [0.,.5,1+T+.5,r.duration]:np.testing.assert_array_equal(ff.wrench(r,t)[0],np.zeros(6))
    for t in [1+T*.3,1+T*.7,2+T*1.3]:
        h=1e-4;z,v,a=r.scalar(t);np.testing.assert_allclose((r.scalar(t+h)[0]-r.scalar(t-h)[0])/(2*h),v,atol=1e-9);np.testing.assert_allclose((r.scalar(t+h)[1]-r.scalar(t-h)[1])/(2*h),a,atol=1e-8)
    assert ff.wrench(r,1+T*.25)[0][2]>0 and ff.wrench(r,2+T*1.25)[0][2]<0

def test_mapping_and_no_duplicate_gravity():
    r=StudyReference(create_plant(),.05,6.,RUN/'references',ExpandedGeometryQuery());j=np.eye(6)+np.ones((6,6))*.02;ff=DynamicReferenceFeedforward(570.453,50.185);f,w,p=ff.leg_force(r,3.,j);np.testing.assert_allclose(j.T@f,w,atol=1e-12);assert w[0]==w[1]==w[3]==w[4]==w[5]==0
    assert p['velocity_force_N']==pytest.approx(570.453*r.scalar(3)[1]);assert p['acceleration_force_N']==pytest.approx(50.185*r.scalar(3)[2])
    with pytest.raises(ValueError):DynamicReferenceFeedforward(np.nan,20)
    with pytest.raises(ValueError):ff.leg_force(r,3.,np.zeros((4,4)))

@pytest.mark.parametrize('baseline',['C4','fourDlead'])
def test_zero_feedforward_bitwise_frozen_baseline_through_motion(tmp_path,baseline):
    a,r=make_owner(tmp_path/'wrapped',baseline,.05,4.);b,_=make_owner(tmp_path/'frozen',baseline,.05,4.);a.residual=LargeMotionFeedforwardWrapper(a.residual,r,DynamicReferenceFeedforward(0,0),'zero-FF')
    from src.control.phase2b2_experiment import MeasurementAdapter
    a.controller=MeasurementAdapter(a.residual)
    try:
        for _ in range(2200):
            a.step();b.step();np.testing.assert_array_equal(a.data.qpos,b.data.qpos);np.testing.assert_array_equal(a.data.qvel,b.data.qvel);np.testing.assert_array_equal(a.data.ctrl,b.data.ctrl)
    finally:a.recorder.close(dict(test='zero_feedforward'));b.recorder.close(dict(test='zero_feedforward'))

def test_one_shared_budget_under_competing_requests(tmp_path):
    o,_=make_owner(tmp_path/'o','FF-VA',.05,4.)
    try:
        prev=np.zeros(6)
        for _ in range(1600):
            o.residual.injection.probe_N=20.;o.step();total=o.residual.injection.audit['actual_total_add'];assert max(abs(total))<=5+1e-10 and max(abs(total-prev))<=.0200000001;prev=total.copy();assert max(abs(o.data.ctrl[:6]))<=400
        assert o.residual.injection.original is not o.residual.injection.counter
    finally:o.recorder.close(dict(test='budget_stress_not_a_research_task'))

def test_wrapper_has_no_hidden_physical_input():
    text=inspect.getsource(ReferenceBudgetInjection)+inspect.getsource(DynamicReferenceFeedforward)
    for forbidden in ['.lag','.profile','actuator_gainprm','F_drive','external_ctrl']:assert forbidden not in text
    cfg=json.loads((ROOT/'configs/phase2c3b_feedforward_study.json').read_text(encoding='utf-8'));old=json.loads((ROOT/'configs/phase2c3a_large_workspace.json').read_text(encoding='utf-8'));assert cfg['workspace']==old and not cfg['no_RL_training']==False
