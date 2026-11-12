"""Derived events from immutable recorded data; not a training writer."""
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,NAMES,write_json,sha

def main():
    target=ROOT/'results/phase2c7v_tensorboard'
    if target.exists():raise FileExistsError('Refuse duplicate events: '+str(target))
    sources=[ROOT/f'results/phase2c7v_{name}.csv' for name in ('reward_components','episode_returns','checkpoint_evaluation')]+[RUN/'updates.csv']
    d,e,v,u=[pd.read_csv(p) for p in sources]
    for seed in SEEDS:
        with SummaryWriter(str(target/f'seed{seed}')) as w:
            for _,r in d[d.seed==seed].iterrows():
                n=int(r.global_decision);w.add_scalar('training/reward_per_decision',r.reward,n)
                for key,label in zip(NAMES,('orientation','angular_velocity','xyz','action','slew')):
                    w.add_scalar('training/reward_'+label,r['reward_'+key],n)
                w.add_scalar('training/shared_projection_fraction',r.projection_fraction,n)
            for r in e[e.seed==seed].itertuples():
                w.add_scalar('training/complete_episode_return' if r.complete else 'training/incomplete_fragment_return',r.episode_return,int(r.end_decision))
            for _,r in u[u.seed==seed].iterrows():
                for key in ('actor_delta','critic_delta','train/value_loss','train/policy_gradient_loss','train/explained_variance','train/approx_kl','train/clip_fraction','train/std','value_target_mean','value_target_std','advantage_mean','advantage_std'):
                    w.add_scalar('optimizer/'+key.replace('train/',''),r[key],int(r.decisions))
            for _,r in v[v.seed==seed].iterrows():
                for key in ('evaluation_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','z_rmse_mm','xy_peak_mm','force_rate_N_s','success'):
                    w.add_scalar('evaluation/'+key,float(r[key]),int(r.decisions))
            w.add_text('provenance','DERIVED HISTORICAL DATA. Environment rewards exclude TimeLimit bootstrap. Complete returns exclude budget fragments. Dev_mix frozen deterministic evaluation; no new training.')
    write_json(target/'provenance.json',dict(sources={str(p):sha(p) for p in sources},original=str(OLD/'training'),seed_field='seed',step_field='global_decision / end_decision / decisions',no_training=True))
if __name__=='__main__':main()
