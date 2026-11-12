"""Reconstruct all rollout GAE entries, including the recorded terminal bootstrap."""
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT,RUN,cfg,write_json
def main():
    c=cfg('training')['ppo'];rows=[]
    for path in (RUN/'training').glob('*/*/*/rollout_*.npz'):
        sample=pd.read_csv(path.parent/'training_samples.csv');update=int(path.stem.split('_')[-1]);part=sample[sample.global_decision.between((update-1)*512+1,update*512)]
        with np.load(path) as b:
            r,v,start,adv,returns=[b[k].reshape(-1).astype(float) for k in ('rewards','values','episode_starts','advantages','returns')];last=float(b['last_values'].reshape(-1)[0])
        reconstructed=np.empty(len(r));gae=0.
        for i in reversed(range(len(r))):
            nonterminal=1-float(part.iloc[-1].terminated or part.iloc[-1].truncated) if i==len(r)-1 else 1-start[i+1]
            next_value=last if i==len(r)-1 else v[i+1];delta=r[i]+c['gamma']*next_value*nonterminal-v[i];gae=delta+c['gamma']*c['gae_lambda']*nonterminal*gae;reconstructed[i]=gae
        np.testing.assert_allclose(reconstructed,adv,rtol=3e-6,atol=5e-5);np.testing.assert_allclose(reconstructed+v,returns,rtol=3e-6,atol=5e-5)
        rows.append(dict(method=path.parts[-4],seed=path.parts[-3],stage=path.parts[-2],update=update,max_advantage_error=float(abs(reconstructed-adv).max()),max_return_error=float(abs(reconstructed+v-returns).max()),last_bootstrap_value=last,all_512_entries_recomputed=True))
    pd.DataFrame(rows).to_csv(RUN/'gae_recomputation.csv',index=False);write_json(RUN/'gae_validation.json',dict(passed=True,rollouts=len(rows),max_advantage_error=max(r['max_advantage_error'] for r in rows),max_return_error=max(r['max_return_error'] for r in rows)))
if __name__=='__main__':main()
