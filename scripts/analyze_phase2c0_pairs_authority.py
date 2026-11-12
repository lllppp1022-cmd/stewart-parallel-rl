"""Strict new paired experiments plus bounded local direction diagnostics."""
import _bootstrap,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json
from src.disturbance.metrics import vec

def main(root):
    root=Path(root);historical=ROOT/'results/runs/phase2b3_20261029_continuation_v2/frozen_evaluation/uneven_dynamic/C4-Observer';sources=dict(nominal_no_load=root/'paired_decomposition/paired_dynamic/nominal_no_load',nominal_with_load=root/'paired_decomposition/paired_dynamic/nominal_with_load',uneven_no_load=root/'paired_decomposition/paired_dynamic/uneven_no_load',uneven_with_load=historical);data={k:pd.read_csv(v/'physics.csv.gz',float_precision='round_trip') for k,v in sources.items()};baseline=data['nominal_no_load']
    for k,d in data.items():
        for key,n in [('q_ref',6),('p_ref',3),('rpy_ref',3)]:np.testing.assert_array_equal(vec(d,key,n),vec(baseline,key,n))
    for yes,no in [('nominal_with_load','nominal_no_load'),('uneven_with_load','uneven_no_load')]:
        for key,n in [('qpos',43),('qvel',42),('force',6)]:np.testing.assert_array_equal(vec(data[yes],key,n)[:1600],vec(data[no],key,n)[:1600])
    rows=[];e={k:(vec(d,'p',3)-vec(d,'p_ref',3))*1000 for k,d in data.items()};mask=(baseline.input_step>=1600)&(baseline.input_step<3650);peak=np.flatnonzero(mask)[np.linalg.norm(e['uneven_with_load'][mask],axis=1).argmax()]
    for key,d in data.items():
        i=np.flatnonzero(mask)[np.linalg.norm(e[key][mask],axis=1).argmax()];r=dict(case=key,source=str(sources[key]),new_experiment=key!='uneven_with_load',matched_reference=True,load_peak_time_s=float(d.time_s.iloc[i]),load_peak_norm_mm=float(np.linalg.norm(e[key][i])),same_historical_peak_time_s=float(d.time_s.iloc[peak]))
        for j,axis in enumerate('xyz'):r[f'{axis}_at_same_peak_mm']=float(e[key][peak,j]);r[f'{axis}_at_own_peak_mm']=float(e[key][i,j])
        rows.append(r)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase2c0_paired_decomposition.csv',index=False)
    components=dict(nominal_no_load=e['nominal_no_load'][peak],gain_increment_no_load=e['uneven_no_load'][peak]-e['nominal_no_load'][peak],load_increment_at_uneven=e['uneven_with_load'][peak]-e['uneven_no_load'][peak],load_increment_at_nominal=e['nominal_with_load'][peak]-e['nominal_no_load'][peak]);np.testing.assert_allclose(components['nominal_no_load']+components['gain_increment_no_load']+components['load_increment_at_uneven'],e['uneven_with_load'][peak],atol=1e-12)
    write_json(ROOT/'results/phase2c0_paired_vector_attribution.json',dict(time_s=float(baseline.time_s.iloc[peak]),xyz_mm={k:v.tolist() for k,v in components.items()},warning='Exact matched vector differences at one time, not additive norm percentages/global linear causality',preload_physical_pairs_bitwise_equal=True))
    # Actual measured finite response, same reference and before-probe state.
    sensitivity=[];columns=[];target_index=2999;norms=[.02,.02,.2,.01,.01,.005]
    for axis in range(6):
        traces=[]
        for sign in [-1,1]:
            folder=root/f'authority_preflight/authority_dynamic/axis{axis}_sign{sign}';d=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');before=2200
            np.testing.assert_array_equal(vec(d,'qpos',43)[:before],vec(baseline,'qpos',43)[:before]);dp=(vec(d,'p',3)-vec(baseline,'p',3))*1000;rr=Rotation.from_euler('xyz',vec(d,'rpy',3),degrees=True);rb=Rotation.from_euler('xyz',vec(baseline,'rpy',3),degrees=True);da=np.rad2deg((rr*rb.inv()).as_rotvec());s=(d.input_step>=2200)&(d.input_step<3200);delta=np.r_[dp[target_index],da[target_index]];traces.append(delta)
            row=dict(axis=axis,sign=sign,request_magnitude=norms[axis],sample_time_s=float(d.time_s.iloc[target_index]),local_xy_peak_increment_mm=float(np.linalg.norm(dp[s,:2],axis=1).max()),local_so3_peak_increment_deg=float(np.linalg.norm(da[s],axis=1).max()))
            for j,name in enumerate(['x_mm','y_mm','z_mm','rot_x_deg','rot_y_deg','rot_z_deg']):row['same_time_'+name]=float(delta[j])
            sensitivity.append(row)
        columns.append((traces[1]-traces[0])/2) # per signed tested amplitude, no mixed-unit SVD.
    S=np.column_stack(columns);scale=np.array([.25,.25,.25,.02,.02,.02]);B=S/scale[:,None];full_error=np.r_[e['uneven_with_load'][target_index],np.rad2deg((Rotation.from_euler('xyz',vec(data['uneven_with_load'],'rpy',3)[target_index],degrees=True)*Rotation.from_euler('xyz',vec(data['uneven_with_load'],'rpy_ref',3)[target_index],degrees=True).inv()).as_rotvec())];pred=[]
    for axes in [[2,3,4,5],[0,1,2,3,4,5]]:
        a=np.linalg.lstsq(B[:,axes],-full_error/scale,rcond=1e-8)[0];bounded=np.clip(a,-1,1);remaining=full_error+S[:,axes]@bounded
        pred.append(dict(axes=axes,normalized_tested_amplitude_request=bounded.tolist(),predicted_remaining_xyz_mm=remaining[:3].tolist(),predicted_remaining_so3_deg=remaining[3:].tolist(),effective_rank=int(np.linalg.matrix_rank(B[:,axes])),singular_values=np.linalg.svd(B[:,axes],compute_uv=False).tolist(),caveat='Offline local finite-response approximation from nominal no-load probes, transferred to slight-gain loaded state; not an executed policy or safety guarantee'))
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/phase2c0_authority_sensitivity.csv',index=False);write_json(ROOT/'results/phase2c0_authority_linear_diagnostic.json',dict(time_s=float(baseline.time_s.iloc[target_index]),before_probe_states_equal=True,predictions=pred,not_random_6d_exploration=True))
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for key,d in data.items():
        for j in range(3):axes[j].plot(d.time_s,e[key][:,j],label=key)
    for ax,title in zip(axes,['X error (mm): matched model/reference, explicit new pairs','Y error (mm)','Z error (mm)']):ax.set_title(title);ax.axvline(1.6,color='k',ls=':');ax.axvline(3.65,color='k',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase2c0_strict_pairs.png',dpi=150);plt.close(fig);print('Matched pairs and local authority validated',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);main(p.parse_args().run_root)
