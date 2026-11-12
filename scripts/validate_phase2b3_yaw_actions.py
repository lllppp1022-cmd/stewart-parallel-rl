"""Standalone zero4 gate; separate new output directory required."""
import _bootstrap,argparse
from pathlib import Path
from run_phase2b3_yaw_compensation import zero
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);r=p.parse_args().run_root;r.mkdir(parents=True,exist_ok=False);zero(r)
