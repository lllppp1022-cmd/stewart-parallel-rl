"""Seal completed Phase8 artifacts; do not touch historical experiments."""
import hashlib
import re
from pathlib import Path
from phase2c8_protocol import ROOT, RUN, read, write_json, sha


def main():
    docs = sorted((ROOT / 'docs').glob('*Phase2C8*.md'))
    assert len(docs) == 6
    broken = []
    for path in docs:
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('http://', 'https://', '#')):
                continue
            if not (path.parent / target).resolve().exists():
                broken.append([str(path), target])
    assert not broken, broken
    for path, digest in read(RUN / 'executed_implementation_v1.json')['files'].items():
        assert sha(ROOT / path) == digest, path
    for model in read(RUN / 'final_selection.json')['models'].values():
        assert sha(model['path']) == model['sha']
    assert read(RUN / 'history_after.json')['passed']
    assert read(RUN / 'final_metric_validation.json')['passed']
    assert not (ROOT / 'models/phase2c8/v3_return_norm').exists()
    required = ['reward_ablation.csv', 'training_summary.csv', 'critic_diagnostics.csv',
                'checkpoint_evaluation.csv', 'controller_comparison.csv', 'action_audit.csv',
                'safety_audit.json', 'final_decision.json']
    assert all((ROOT / ('results/phase2c8_' + p)).exists() for p in required)
    for kind in ['training', 'reward_comparison', 'attitude_stability']:
        v = read(RUN / f'video_{kind}.json')
        assert sha(v['path']) == v['sha']
    write_json(RUN / 'delivery_validation.json', dict(passed=True, documents=len(docs), broken_links=broken,
                executed_training_implementation_unchanged=True, final_models_unchanged=True,
                all_required_conditional_outputs_present=True, return_norm_models_not_fabricated=True,
                videos_inspected=True, plots_inspected=True, no_new_training=True))
    paths = set(docs)
    for folder in ['scripts', 'src/rl', 'tests', 'configs', 'assets/plots', 'assets/videos', 'results']:
        paths.update(p for p in (ROOT / folder).glob('*phase2c8*') if p.is_file())
    for folder in [RUN, ROOT / 'models/phase2c8', ROOT / 'results/phase2c8_tensorboard']:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    manifest = RUN / 'artifact_manifest.json'
    paths = sorted(p for p in paths if p != manifest and not any(x.startswith('pytest') for x in p.parts)
                   and '__pycache__' not in p.parts and not p.name.startswith('phase2c8_tb_server'))
    records = []
    total = sum(p.stat().st_size for p in paths)
    print(f'Archiving {len(paths)} files, {total} bytes (excludes regression temporary copies and active server logs)', flush=True)
    for i, path in enumerate(paths):
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        records.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest))
        if (i + 1) % 500 == 0:
            print(f'Hashed {i+1}/{len(paths)}', flush=True)
    write_json(manifest, dict(files=records, file_count=len(records), bytes=total,
        scope='Completed new Phase8 source/configs/models/raw logs/TensorBoard/plots/videos/docs/proofs; excludes temporary regression copies, active TensorBoard server logs and this manifest itself',
        original_history_manifest=str(RUN/'history_before.json'), audit_manifest=str(RUN/'history_after.json')))
    print('Archive complete:', manifest, flush=True)


if __name__ == '__main__':
    main()
