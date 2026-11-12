"""Actual PPO on/off RGB identity; 128 decisions, two64-step updates per branch."""
import _bootstrap
import numpy as np,torch,json,argparse
from research_phase2c2 import RUN,PROTOCOL,cfg_for
from src.model import write_json
from src.rl.phase2c2_env import MovingResidualEnv
from src.rl.ppo_trace import TracedPPO,SamplingCallback,parameter_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--updated-camera',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);out=[]
    if args.updated_camera:out=[json.loads((RUN/'observer_equivalence.json').read_text(encoding="utf-8"))['branches'][0]]
    on_name='on_v2' if args.updated_camera else 'on'
    for enabled in ([True] if args.updated_camera else [False,True]):
        f=RUN/'observer_test'/(on_name if enabled else 'off');f.mkdir(parents=True,exist_ok=False)
        e=MovingResidualEnv(f/'training',config=cfg_for(PROTOCOL['train']),record_episodes=[0,1] if enabled else [],mode='Actual Training')
        m=TracedPPO('MlpPolicy',e,seed=9201,device='cpu',n_steps=64,batch_size=32,n_epochs=2,policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),log_std_init=-1.5))
        with torch.no_grad():m.policy.action_net.weight.zero_();m.policy.action_net.bias.zero_()
        m.attach(e,f,f/'models');m.learn(128,callback=SamplingCallback(e,dict(protected_failures_in_last3_at_least=2)));h=parameter_hash(m.policy);m.save(f/'final.zip');m.finish_trace();e.close();out.append(dict(enabled=enabled,parameter_hash=h,physical_steps=e.total_physical_steps,decisions=e.global_decisions))
    comparisons={}
    for index in [1,2]:
        a=np.load(RUN/f'observer_test/off/rollout_{index:03d}.npz');b=np.load(RUN/f'observer_test/{on_name}/rollout_{index:03d}.npz')
        for key in a.files:comparisons[f'{index}_{key}']=bool(np.array_equal(a[key],b[key]))
    summaries=[json.loads((RUN/f'observer_test/{x}/training/episode_summary.json').read_text(encoding="utf-8")) for x in ['off',on_name]]
    same_end=[(s['reason'],s['physical_steps'],s['decisions'],s['return_']) for s in summaries[0]]==[(s['reason'],s['physical_steps'],s['decisions'],s['return_']) for s in summaries[1]]
    passed=all(comparisons.values()) and same_end and out[0]['parameter_hash']==out[1]['parameter_hash']
    write_json(RUN/('observer_equivalence_v2.json' if args.updated_camera else 'observer_equivalence.json'),dict(passed=passed,branches=out,rollout_arrays_exact=comparisons,episode_endings_exact=same_end,scope='RGB recorder on/off ACTUAL PPO; not a viewer thread test',budget='off128 + on128 + optional new-camera128 decisions; diagnostic only'))
    assert passed
if __name__=='__main__':main()
