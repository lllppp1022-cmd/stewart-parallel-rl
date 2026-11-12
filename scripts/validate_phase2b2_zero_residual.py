"""Standalone strict zero-action gate; no gain/threshold changes."""
import _bootstrap
import argparse
from run_phase2b2_action_sensitivity import zero_checks
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();zero_checks(a.run_root)
