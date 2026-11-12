from pathlib import Path
import hashlib,json
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'third_party/stewart_platform_mujoco'
MODEL=UPSTREAM/'scene.xml'
def load_model():return mujoco.MjModel.from_xml_path(str(MODEL))
def ids(model):
    return np.array([model.joint(f'joint_lu_{i}').qposadr[0] for i in range(1,7)]),np.array([model.joint(f'joint_lu_{i}').dofadr[0] for i in range(1,7)])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def equality_residual(data):
    mask=np.asarray(data.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    return np.asarray(data.efc_pos)[mask].copy()
def site_gap(model,data):
    return np.array([np.linalg.norm(data.site(f'site_c1_{i+12}').xpos-data.site(f'site_upper_{i}').xpos) for i in range(1,7)])
