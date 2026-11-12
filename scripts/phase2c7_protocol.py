"""Phase 2C-7 protocol and independent artifact roots."""
import _bootstrap
import hashlib
import json
from pathlib import Path

from src.model import ROOT, sha, write_json

RUN=ROOT/"results/runs/phase2c7_20261107_v1"
SEEDS=(42,123,2026)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config(name):
    return read(ROOT/f"configs/phase2c7_{name}.yaml")


def trajectory(z=.05, roll=0., pitch=0., rise=6.):
    return dict(z_m=z,roll_deg=roll,pitch_deg=pitch,rise_s=rise)


def task(name, ref, torque, on, off, stage):
    return dict(id=name,reference=ref,torque=torque,on=on,off=off,stage=stage)


def prepare():
    RUN.mkdir(parents=True,exist_ok=False)
    # Hash current persistent sources and old artifacts, without modifying old manifests.
    folders=("src","configs","models","docs","assets","third_party","scripts","tests")
    paths=[p for name in folders for p in (ROOT/name).rglob("*") if p.is_file()
           and "__pycache__" not in p.parts and p.suffix!=".pyc" and not p.name.startswith("~$")
           and "phase2c7" not in str(p.relative_to(ROOT)).lower()]
    old=read(ROOT/"results/runs/phase2c6_20261107_v1/history_before.json")
    paths += [ROOT/name for name in old if (ROOT/name).is_file()]
    paths += [p for p in (ROOT/"results").glob("*2c6*") if p.is_file()]
    paths += [p for p in (ROOT/"results/runs/phase2c6_20261107_v1").rglob("*") if p.is_file()]
    hashes={}
    for path in sorted(set(paths)):
        with path.open("rb") as stream:
            hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(stream,"sha256").hexdigest()
    write_json(RUN/"history_before.json",hashes)
    # This finite list is frozen before physics outcomes and policy learning.
    home=[task(f"A{i}",trajectory(0),torque,on,off,"A")
          for i,(torque,on,off) in enumerate([
              ([.5,0],3.,5.),([0,-.8],6.2,8.2),([1.,0],9.2,11.),
              ([0,1.2],7.1,8.6),([.72,-.96],4.5,6.3)])]
    z=[task(f"B{i}",trajectory(amp,rise=rise),torque,on,off,"B")
       for i,(amp,rise,torque,on,off) in enumerate([
           (.03,6.,[.5,0],3.,5.),(.05,6.,[0,-.8],6.2,8.2),
           (.05,6.,[1.,0],9.2,11.),(.05,6.,[0,1.2],7.1,8.6),
           (.05,6.,[.72,-.96],4.5,6.3),(.03,5.,[-.5,0],8.5,10.)])]
    combined=[task(f"C{i}",trajectory(.05,r,p),torque,on,off,"C")
              for i,(r,p,torque,on,off) in enumerate([
                  (.1,0,[.5,0],3.,5.),(0,-.1,[0,-.8],6.2,8.2),
                  (.1,-.1,[.6,.8],9.2,11.),(-.1,.1,[.72,-.96],7.1,8.6),
                  (-.25,0,[-.8,0],4.5,6.3),(0,.25,[0,1.],9.,10.5),
                  (.25,-.25,[.72,.96],6.3,8.1)])]
    final=[
        task("F_roll",trajectory(.05,.1,0,5.5),[-.65,0],3.55,5.35,"final"),
        task("F_pitch",trajectory(.05,0,-.1,6.5),[0,.9],10.,11.8,"final"),
        task("F_mix",trajectory(.05,-.1,.1,6.5),[.66,-.88],7.4,8.8,"final"),
        task("F_30",trajectory(.03,.08,-.12,5.5),[-.8,.6],4.6,6.4,"final")]
    develop=[task("Dev_mix",trajectory(.05,.1,-.1),[.48,-.64],5.,7.,"development")]
    settings={
        "env":dict(dt_s=.001,decision_steps=20,action_scale_Nm=.05,observation_dim=58,
                   base="fourDlead plus frozen FF-VA",tau_s=.005,
                   observation_scales=dict(position_m=[.0002,.0002,.00005],slides_m=.06,slide_velocity_m_s=.03,
                                           reference_z_m=.05,reference_vz_m_s=.03,reference_az_m_s2=.03,
                                           reference_angular_acceleration_rad_s2=.02),
                   reward=dict(orientation_deg=.006,omega_rad_s=.003,xy_mm=.2,z_mm=.05,
                               weights=[1.,.2,.1,.001,.000025],action_Nm=.05,slew_Nm_s=2.5,cap=100.),
                   success=dict(final_position_mm=.75,final_orientation_deg=.1,dwell_s=.5),
                   warm_start="Fresh 58-input policy, zero actor output mean; old 52-input policies are separately evaluated with their original features."),
        "curriculum":dict(A=home,B=z,C=combined,stage_budgets=[4096,4096,24576],
                          pilot=8192,total=32768,admission="Exact task and geometry hashes plus baseline/random full episode validation",
                          gate="No protection or nonfinite transition; complete A and B episodes; both networks update; all joint tasks retained only when admitted"),
        "ppo":dict(n_steps=512,batch_size=64,n_epochs=10,learning_rate=.0003,gamma=.99**.4,
                   gae_lambda=.95**.4,clip_range=.2,ent_coef=0.,max_grad_norm=.5,target_kl=.03,
                   net=[64,64],log_std_init=-1.,device="cpu",seeds=list(SEEDS),
                   pilot=8192,total=32768,return_normalization=False),
        "eval":dict(final=final,development=develop,gains=dict(kp=25.,kd=1.),
                    recovery_bands_deg=[.1,.01],recovery_dwell_s=.5,
                    criterion="At least two new seeds improve both 1s load/unload IAE in at least two independent tasks, all complete, with Z RMSE, peak XY, and actual-force slew <=110% of classical on every task; report all absolute effects.",
                    selection="Final bounded checkpoints for all seeds, frozen before final policy evaluation")}
    for name,value in settings.items():
        path=ROOT/f"configs/phase2c7_{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        write_json(path,value)


if __name__=="__main__":
    prepare()
