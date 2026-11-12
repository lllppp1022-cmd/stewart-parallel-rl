"""Full integration-prefix and platform-angular-response audit of saved probes."""
import numpy as np
import pandas as pd
from phase2c9b_protocol import ROOT,RUN,write_json


def main():
    rows=[]
    for mass in (20.,21.2,22.):
        root=RUN/'authority'/str(mass)
        source=root/'0/episode_0000'
        zero=pd.read_csv(source/'physics.csv.gz')
        with np.load(source/'integration_states.npz') as z:reference=z['states'][:4400].copy()
        for action in range(6):
            folder=root/str(action)/'episode_0000'
            with np.load(folder/'integration_states.npz') as z:np.testing.assert_array_equal(reference,z['states'][:4400])
            d=pd.read_csv(folder/'physics.csv.gz')
            angular=[f'twist_actual_{j}' for j in (3,4,5)]
            force=[f'force_{j}' for j in range(6)]
            rows.append(dict(mass_kg=mass,probe=action,full_native_prefix_bitwise_equal=True,
                platform_angular_velocity_difference_rad_s=float(abs(d[angular].to_numpy()-zero[angular].to_numpy()).max()),
                actual_force_difference_N=float(abs(d[force].to_numpy()-zero[force].to_numpy()).max()),
                projection_fraction=float(d.projected.mean()),shared_add_peak_N=float(abs(d[[f'F_command_{j}' for j in range(6)]].to_numpy()-d[[f'F_C0_{j}' for j in range(6)]].to_numpy()).max()),
                source=str(folder)))
    table=pd.DataFrame(rows)
    assert table[table.probe!=0].platform_angular_velocity_difference_rad_s.gt(1e-10).all()
    table.to_csv(ROOT/'results/phase2c9b_authority_audit.csv',index=False)
    write_json(RUN/'authority_validation.json',dict(passed=True,all_native_prefixes_equal=True,all_nonzero_probes_change_platform_angular_velocity=True,comparisons=len(rows)))


if __name__=='__main__':main()
