"""Supplement explicit pure 50 mm Z checks; retain the earlier 30 mm layer."""
import argparse
from phase2c9b_protocol import ROOT,RUN,cfg,read,write_json,sha,task
from evaluate_phase2c9b import trial


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);args=p.parse_args()
    reference=read(ROOT/'configs/phase2c8_training.yaml')['curriculum']['B'][1]['reference']
    masses=[20.,20.3,20.6,21.2,21.5,21.6,21.8,22.]
    for i,mass in enumerate(masses):
        if i%4==args.worker:
            trial(task(f'PureZ50_m{mass}',mass,reference,(0.,0.)),'baseline','preflight_z50')


if __name__=='__main__':main()
