"""Validate optional wrapper with real physics, without training a policy."""
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from phase2c8_protocol import ROOT,RUN,cfg,read,write_json
from src.rl.phase2c8_env import Phase2C8Env

def main():
    task=cfg('evaluation')['development'][0];root=RUN/'validation/normalization';c=cfg('return_norm')
    plain=Phase2C8Env(root/'plain',[task],seed=42,version='v3');plain.reset(seed=11);rewards=[]
    for _ in range(25):rewards.append(plain.step(np.array([.1,-.1]))[1])
    plain.close()
    raw=Phase2C8Env(root/'normalized',[task],seed=42,version='v3');vec=VecNormalize(DummyVecEnv([lambda:raw]),norm_obs=False,norm_reward=True,gamma=c['gamma'],clip_reward=c['clip_reward'],epsilon=c['epsilon']);obs=vec.reset()
    assert np.array_equal(obs[0],raw.obs)
    normalized=[]
    for _ in range(25):
        obs,r,_,_=vec.step(np.array([[.1,-.1]]));normalized.append(float(r[0]));assert np.array_equal(obs[0],raw.obs)
    np.testing.assert_allclose([r['reward'] for r in raw.records],rewards,rtol=0,atol=0)
    saved=root/'statistics.pkl';vec.save(saved);mean,var,count=vec.ret_rms.mean.copy(),vec.ret_rms.var.copy(),vec.ret_rms.count;vec.close()
    fresh=Phase2C8Env(root/'evaluation',[task],seed=42,version='v3');loaded=VecNormalize.load(saved,DummyVecEnv([lambda:fresh]));loaded.training=False;loaded.norm_reward=False;loaded.reset()
    assert not loaded.norm_obs and loaded.ret_rms.count==count
    for i in range(25):
        _,r,_,_=loaded.step(np.array([[.1,-.1]]));np.testing.assert_allclose(r[0],rewards[i],rtol=1e-6,atol=1e-8)
    assert loaded.ret_rms.count==count and loaded.ret_rms.mean==mean and loaded.ret_rms.var==var;loaded.close()
    with np.load(root/'plain/episode_0000/integration_states.npz') as x:
        for name in ('normalized','evaluation'):
            with np.load(root/f'{name}/episode_0000/integration_states.npz') as y:np.testing.assert_array_equal(x['states'],y['states'])
    write_json(RUN/'normalization_validation.json',dict(passed=True,physical_steps=1500,obs_normalization=False,raw_rewards_unchanged=True,physics_bitwise_equal=True,statistics_reload_equal=True,evaluation_statistics_frozen=True,actual_ppo_norm_training_executed=False))
if __name__=='__main__':main()
