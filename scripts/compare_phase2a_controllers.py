"""Read-only comparison / independent metric recomputation; never starts a trial."""
import _bootstrap
import argparse
from analyze_phase2a_recovery import analyze
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);a=p.parse_args();df=analyze(a.run_root);f=df[df.protocol_role=='formal'];print(f[['task','controller','success','peak_position_mm','peak_orientation_deg','load_recovery_s','load_recovery_status','unload_recovery_s']].to_string(index=False))
