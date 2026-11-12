"""Repeat the frozen paired Pilot in a NEW directory; never overwrites the original."""
import _bootstrap,argparse,copy,json,re,shutil,sys
from pathlib import Path
import research_phase2c2 as research
import train_phase2c2_paired as paired
from src.model import ROOT,sha,write_json

SOURCE=ROOT/'results/runs/phase2c2_20261030_v1'

def validate(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):raise ValueError('Use a simple new run name: letters, numbers, underscores or hyphens')
    target=ROOT/'results/runs'/name;models=ROOT/'models/phase2c2'/name
    if target.exists() or models.exists():raise FileExistsError('Run/model directory already exists; choose a NEW run name')
    frozen=json.loads((SOURCE/'protocol_frozen.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'results/phase2c2_artifact_manifest.json').read_text(encoding='utf-8'))['files']
    historical=json.loads((SOURCE/'historical_file_hashes.json').read_text(encoding='utf-8'))
    for relative in ['scripts/train_phase2c2_paired.py','scripts/research_phase2c2.py','src/rl/phase2c2_env.py','src/rl/moving_reference.py','src/rl/phase2c2_recording.py','src/rl/ppo_trace.py']:
        item=manifest.get(relative) or manifest.get(relative.replace('/','\\'))
        expected=item['sha256'] if item else historical.get(relative) or historical.get(relative.replace('/','\\'))
        if expected is None or sha(ROOT/relative)!=expected:raise RuntimeError('Frozen source changed: '+relative)
    hyper=json.loads((ROOT/'configs/phase2c1_ppo.yaml').read_text(encoding='utf-8'));hyper.update(log_std_init=-1.5,training_seeds=[42,123],cumulative_stage_decisions=[256,2048,4096],record_training_episodes=[2,8,16,32])
    configs=[]
    for seed in [42,123]:
        c=json.loads((SOURCE/f'seed{seed}/config_frozen.json').read_text(encoding='utf-8'));actual=research.cfg_for(frozen['train']);expected=copy.deepcopy(c['env']);actual.pop('reference_cache');expected.pop('reference_cache')
        if actual!=expected or hyper!=c['ppo']:raise RuntimeError(f'Current settings differ from frozen seed{seed}; refusing a misleading reproduction')
        configs.append(c)
    if not json.loads((SOURCE/'training_gate.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Source training gate failed')
    if not json.loads((SOURCE/'observer_equivalence_v2.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Source recorder gate failed')
    return target,models,frozen,configs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-name',required=True);p.add_argument('--check-only',action='store_true',help='Validate configuration and paths without creating directories or training');a=p.parse_args()
    target,models,protocol,configs=validate(a.run_name)
    print('Source:',SOURCE);print('New results:',target);print('New checkpoints:',models);print('Seeds42/123; stages256 ->2048 ->4096, joint gate retained.')
    if a.check_only:print('CHECK PASSED. No new training or output directory created.');return
    target.mkdir(parents=True,exist_ok=False)
    for relative in ['protocol_frozen.json','training_gate.json','observer_equivalence_v2.json','baselines/V_move/four_d_lead/metrics.json']:
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/relative,destination)
    write_json(target/'reproduction_provenance.json',dict(source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol_frozen.json'),reused_checks=['training_gate','recorder_equivalence','validation_baseline'],checks_are_prior_evidence_not_new_experiments=True,requested_new_budget_per_seed=4096,metadata_note='Current source fixes local episode journal identity; physical algorithm/configuration unchanged; compare parameter hashes, not timestamped ZIP hashes'))
    # The functions retain their original control/sampling implementation.
    # Only output roots and the already-frozen protocol are rebound.
    research.RUN=target;research.PROTOCOL=copy.deepcopy(protocol);paired.RUN=target;paired.PROTOCOL=copy.deepcopy(protocol)
    paired.main()
    rows=[]
    for seed in [42,123]:
        old=json.loads((SOURCE/f'seed{seed}/training_result.json').read_text(encoding='utf-8'));new=json.loads((target/f'seed{seed}/training_result.json').read_text(encoding='utf-8'))
        rows.append(dict(seed=seed,original_decisions=old['decisions'],new_decisions=new['decisions'],parameter_hash_equal=old['final_parameter_hash']==new['final_parameter_hash']))
    write_json(target/'reproduction_parameter_comparison.json',rows)
    print('Training complete. Parameter comparisons:',rows)
if __name__=='__main__':main()
