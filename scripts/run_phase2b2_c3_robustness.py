"""Single-factor small C3 uncertainty pilot. Nominal controller never sees actual parameters."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from run_phase2b2_action_sensitivity import run_trial,task,protocol
from src.model import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root;assert json.loads((root/'zero_gate.json').read_text(encoding='utf-8'))['passed'];plan=protocol();t=task('robust_force10');records=[];skipped=[]
    groups=[('mass',[(str(x),dict(actual_perturbation=dict(mass_scale=x))) for x in plan['mass_scales']]),('com',[(str(x),dict(actual_perturbation=dict(com_x_m=x))) for x in plan['com_x_m']]),('gain',[(str(x),dict(actual_perturbation=dict(gain_scales=[x]*6))) for x in plan['uniform_gain']]),('asymmetric',[(str(x),dict(actual_perturbation=dict(gain_scales=(1+x*np.array([-1,1,-1,1,-1,1])).tolist()))) for x in plan['asymmetric_gain_amplitude']])]
    for name,key in [('position','position_std_m'),('rotation','rotation_std_rad'),('velocity','velocity_std')]:groups.append((name,[(str(x),dict(measurement={key:x,'seed':42})) for x in plan[name+'_noise_std_'+('m' if name=='position' else 'rad' if name=='rotation' else 'SI')]]))
    groups.append(('delay',[(str(x),dict(measurement=dict(delay_steps=x,seed=42))) for x in plan['delay_steps']]))
    write_json(root/'robustness_plan_frozen.json',dict(groups=groups,noise_source='synthetic candidate, no hardware specification',boundary='p/R/v only; q/dq remain ideal',pair_methods=['C3','C0 for measurement cases'],single_factor=True))
    for name,levels in groups:
        failed=False
        for label,spec in levels:
            if failed:skipped.append(dict(factor=name,level=label,reason='prior smaller level failed'));continue
            methods=('C3','C0') if 'measurement' in spec else ('C3',)
            for base in methods:
                r=run_trial(root,t,base,stage='robustness',variant=f'{name}_{label}_{base}',use_residual=False,split=name,**spec);records.append(r)
                if not r['success']:failed=True
    write_json(root/'robustness_finished.json',dict(records=records,unexecuted=skipped,thresholds_unchanged=True,only_single_factors=True));print('ROBUSTNESS FINISHED',flush=True)
if __name__=='__main__':main()
