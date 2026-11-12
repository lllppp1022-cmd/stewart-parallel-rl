"""Final paired effects, actual authority and independent metric recomputation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,read,write_json


def main():
    data=pd.read_csv(ROOT/'results/phase2c9b_generalization.csv');assert len(data)==72
    checks=[];actions=[];events=[];contacts=[];references={}
    for r in data.to_dict('records'):
        folder=Path(r['folder']);m=read(folder.parent/'metrics.json');task=m['task']
        d=pd.read_csv(folder/'physics.csv.gz');step=d.input_step.to_numpy();t=step*.001
        contacts.append(dict(task_id=r['task_id'],method=r['method'],seed=r['seed'],
            native_contact_peak=int(d.native_contacts.max()),warning_peak=int(d.warning_count.max()),
            native_force_saturation_steps=int(d.native_force_saturated.sum())))
        actual=Rotation.from_euler('xyz',d[[f'rpy_{j}' for j in range(3)]],degrees=True)
        ref=Rotation.from_euler('xyz',d[[f'rpy_ref_{j}' for j in range(3)]],degrees=True)
        angles=(actual*ref.inv()).as_rotvec()*180/np.pi;error=np.linalg.norm(angles,axis=1)
        xyz=d[[f'p_{j}' for j in range(3)]].to_numpy()-d[[f'p_ref_{j}' for j in range(3)]].to_numpy()
        force=d[[f'force_{j}' for j in range(6)]].to_numpy()
        add=d[[f'F_command_{j}' for j in range(6)]].to_numpy()-d[[f'F_C0_{j}' for j in range(6)]].to_numpy()
        values=dict(so3_peak_deg=float(error.max()),omega_peak_rad_s=float(np.linalg.norm(d[[f'twist_actual_{j}' for j in (3,4,5)]],axis=1).max()),
            xyz_rmse_mm=float(np.sqrt(np.mean(np.sum(xyz**2,axis=1)))*1000),z_rmse_mm=float(np.sqrt(np.mean(xyz[:,2]**2))*1000),
            xy_peak_mm=float(abs(xyz[:,:2]).max()*1000),force_peak_N=float(abs(force).max()),force_rate_N_s=float(abs(np.diff(force,axis=0)).max()*1000),
            force_squared_integral_N2_s=float(np.sum(force**2)*.001),add_peak_N=float(abs(add).max()),
            add_rate_N_s=float(abs(np.diff(np.vstack([np.zeros(6),add]),axis=0)).max()*1000))
        for name,event in [('load','on'),('unload','off')]:
            n=round(task[event]*1000);values[name+'_iae_deg_s']=float(error[(step>=n)&(step<n+1000)].sum()*.001)
        for key,value in values.items():
            np.testing.assert_allclose(value,m[key],rtol=1e-9,atol=1e-10)
            checks.append(dict(task_id=r['task_id'],method=r['method'],seed=r['seed'],metric=key,absolute_error=abs(value-m[key])))
        dref=d[[f'p_ref_{j}' for j in range(3)]+[f'rpy_ref_{j}' for j in range(3)]].to_numpy()
        if task['id'] in references:
            count=min(len(references[task['id']]),len(dref))
            np.testing.assert_array_equal(references[task['id']][:count],dref[:count])
        else:references[task['id']]=dref
        expected=((step>=round(task['on']*1000))&(step<round(task['off']*1000)))[:,None]*np.array([0,0,0,*task['torque'],0])
        np.testing.assert_allclose(d[[f'external_world_{j}' for j in range(6)]],expected,atol=1e-12,rtol=0)
        if r['method'] in METHODS:
            request=d[['requested_mx_Nm','requested_my_Nm']].to_numpy();accepted=d[['accepted_mx_Nm','accepted_my_Nm']].to_numpy()
            actions.append(dict(method=r['method'],seed=r['seed'],task_id=r['task_id'],mass_kg=r['mass_kg'],request_rms_Nm=float(np.sqrt((request**2).mean())),
                accepted_rms_Nm=float(np.sqrt((accepted**2).mean())),projection_fraction=float(d.projected.mean()),budget_peak_N=float(abs(add).max())))
            for name,event in [('load','on'),('unload','off')]:
                for end in (.02,.1,1.):
                    first=round(task[event]*1000)
                    mask=(step>=first)&(step<first+round(end*1000))
                    if not mask.any():continue
                    events.append(dict(method=r['method'],seed=r['seed'],task_id=r['task_id'],event=name,window_s=end,
                        mean_request_mx=float(request[mask,0].mean()),mean_request_my=float(request[mask,1].mean()),
                        so3_iae_deg_s=float(error[mask].sum()*.001)))
    effects=[];guards=['so3_peak_deg','omega_peak_rad_s','xyz_rmse_mm','z_rmse_mm','force_rate_N_s','force_squared_integral_N2_s']
    for r in data[data.method=='history_5frame'].to_dict('records'):
        for comparator in ('single_frame','adaptive','classical50','baseline'):
            b=data[(data.task_id==r['task_id'])&(data.method==comparator)]
            if comparator=='single_frame':b=b[b.seed==r['seed']]
            b=b.iloc[0];load=100*(1-r['load_iae_deg_s']/b.load_iae_deg_s);unload=100*(1-r['unload_iae_deg_s']/b.unload_iae_deg_s)
            recovery_comparable=r['unload_recovery_0.01_status']=='recovered' and b['unload_recovery_0.01_status']=='recovered'
            recovery_delta=(r['unload_recovery_0.01_s']-b['unload_recovery_0.01_s'])*1000 if recovery_comparable else np.nan
            effects.append(dict(seed=r['seed'],task_id=r['task_id'],mass_kg=r['mass_kg'],comparator=comparator,load_improvement_pct=load,
                history_load_iae_deg_s=r['load_iae_deg_s'],comparator_load_iae_deg_s=float(b.load_iae_deg_s),
                history_unload_iae_deg_s=r['unload_iae_deg_s'],comparator_unload_iae_deg_s=float(b.unload_iae_deg_s),
                load_absolute_reduction_deg_s=float(b.load_iae_deg_s-r['load_iae_deg_s']),
                unload_absolute_reduction_deg_s=float(b.unload_iae_deg_s-r['unload_iae_deg_s']),
                unload_improvement_pct=unload,all_guards=all(r[k]<=b[k]*1.1+1e-12 for k in guards),safe_complete=bool(r['success']),
                both_iae_improve=bool(r['success'] and load>=1 and unload>=1),
                unload_recovery_comparable=bool(recovery_comparable),unload_recovery_delta_ms=recovery_delta))
    effects=pd.DataFrame(effects);evidence={};qualified_seeds={}
    for comparator in ('single_frame','adaptive'):
        good=[]
        for seed in SEEDS:
            d=effects[(effects.seed==seed)&(effects.comparator==comparator)]
            good.append(bool((d[d.mass_kg!=20].both_iae_improve.sum()>=2) and d.all_guards.all() and d.safe_complete.all()))
        evidence[comparator]=sum(good)>=2
        qualified_seeds[comparator]=[seed for seed,passed in zip(SEEDS,good) if passed]
    local=effects[(effects.comparator=='single_frame')&effects.both_iae_improve&effects.all_guards&effects.safe_complete]
    status='STRONG ADAPTIVE RL SUCCESS' if all(evidence.values()) else 'HISTORY BENEFIT SUPPORTED' if evidence['single_frame'] else 'PARTIAL SUCCESS' if len(local) else 'NEGATIVE'
    for name,rows in [('physical_metric_checks',checks),('action_audit',actions),('action_events',events),('native_contact_audit',contacts)]:
        pd.DataFrame(rows).to_csv(ROOT/f'results/phase2c9b_{name}.csv',index=False)
    effects.to_csv(ROOT/'results/phase2c9b_paired_effects.csv',index=False)
    data.groupby(['method','unload_recovery_0.01_status']).size().reset_index(name='trials').to_csv(ROOT/'results/phase2c9b_recovery_status.csv',index=False)
    data.to_csv(ROOT/'results/phase2c9b_controller_comparison.csv',index=False)
    write_json(RUN/'physical_validation.json',dict(passed=True,trials=len(data),metrics=len(checks),reference_exact=True,external_atol=1e-12,
        max_absolute_error=max(c['absolute_error'] for c in checks)))
    write_json(ROOT/'results/phase2c9b_final_decision.json',dict(status=status,engineering_training_completed=True,
        evidence=evidence,qualified_seeds=qualified_seeds,all_seed_results_retained=True,final_trials=len(data),final_safe_complete=int(data.success.sum()),
        budget=read(RUN/'final_selection.json')['common_budget'],causal_ablation_required=bool(evidence['single_frame']),
        matched_capacity_control_required=bool(evidence['single_frame']),no_statistical_significance_claim=True))
    print(status,evidence,flush=True)


if __name__=='__main__':main()
