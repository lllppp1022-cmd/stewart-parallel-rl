"""Resolve the platform-only distmax truncation; preserve original capped scan fields."""
import _bootstrap
import numpy as np,pandas as pd,mujoco
from src.model import ROOT
from src.phase1.plant import create_plant
from src.kinematics.stewart_geometry import StewartGeometry
from src.kinematics.inverse_kinematics import inverse_kinematics
from src.audit_control import scratch_reference_data

def main():
    path=ROOT/'results/phase2c3a_workspace_scan.csv';df=pd.read_csv(path);m=create_plant();g=StewartGeometry(m);ga=int(m.body('base').geomadr[0]);gb=int(m.body('upper').geomadr[0]);actual=[];seed=None
    for z in df.delta_z_m:
        p=g.home_position+[0,0,z];s=inverse_kinematics(g,p,g.home_rotation,seed);seed=s.chain_q;d=scratch_reference_data(m,g,s,p,g.home_rotation);actual.append(float(mujoco.mj_geomDistance(m,d,ga,gb,2.,None)))
    df['platform_gap_is_lower_bound']=df.platform_gap_m>=.15;df['cross_branch_gap_is_lower_bound']=df.cross_branch_gap_m>=.15;df['platform_gap_query_2m_m']=actual;df['platform_gap_query_2m_lower_bound']=np.array(actual)>=2;df.to_csv(path,index=False)
if __name__=='__main__':main()
