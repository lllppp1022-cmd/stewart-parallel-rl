# GitHub release report

Release date: 2026-11-12, Asia/Shanghai.

Repository: https://github.com/lllppp1022-cmd/stewart-parallel-rl

Owner: lllppp1022-cmd. Visibility: Private, confirmed by GitHub CLI. Default branch will be main after the first push. Upload status is pending remote verification in this initial preparation commit.

## Published content

English and Chinese READMEs, original src/scripts/configs/tests directory layout, 94-stage research documents, selected numerical summaries, upstream Stewart MJCF/assets with MIT notice, generated proxy geometry, geometric reference caches, six 32768-decision terminal policy checkpoints, three original demonstration videos, GIF excerpts and cover images. One 1000-step historical state prefix supports the default GUI acceptance check.

The upstream model commit is b0195ba9f40f6e0876220be1712eb47f5fc24025. THIRD_PARTY_NOTICES.md preserves attribution. No blanket MIT license is assigned to original project code.

## Excluded content

Bulk 1 ms physical journals, full integration-state archives, rollout arrays, intermediate/duplicate models, TensorBoard events, local temporary runs, unrelated robot projects and credentials remain local. Excluded raw data are needed for complete independent historical recomputation and the full research test suite.

## Verification performed

- Recomputed published Phase 2C-9B method means from all 72 per-trial evaluation rows.
- Reloaded all six terminal models and verified their SHA256 and observation/action dimensions.
- Executed 1000 real MuJoCo steps with a 21.5 kg Plant and a nominal 20 kg controller.
- Executed the headless home controller for one simulated second.
- Ran four model/transmission/Jacobian tests, all passed.
- Executed the Qt viewer for 1000 steps, checked normal close and bitwise equality to the archived prefix.
- Executed training and evaluation CLI --help entries without starting training.
- Decoded all three videos and verified their bytes against the original files. Reviewed generated cover imagery. GIFs are excerpts of these videos.
- Scanned source/text/checkpoint metadata for private keys, GitHub tokens, AWS access keys and personal home paths. Details are in release/security_audit.json.

The initial smoke test failed because of sandbox temporary-directory permissions. It now writes isolated runtime outputs under this checkout. Subsequent smoke attempts identified two omitted runtime assets, nominal_task_model.json and phase1_proxy_query.xml, which were added before the passing test. The proxy asset mesh directory was made relative for portability. GUI validation initially required omitted full historical data; a small archived prefix was included for this explicit check. Original control algorithms and original research artifacts remain in the source project.

## Reproducibility scope

Documentation and selected summaries are available. The model, core kinematics, home control, final policy loading and short dynamic execution passed local checks. The Windows GUI passed a bounded local run. Dependencies are pinned from the actual Python 3.11.17 environment, including PySide6_Essentials. A fresh network installation was not performed. Complete long training and complete historical tests were not repeated for this release. Historical acceptance reports retain the old Phase 0 learning-package assertion failure and initial sandbox test errors.

Run scripts/setup_portfolio.py after cloning. It resolves __PROJECT_ROOT__ tokens in metadata. Tests requiring omitted archives should be run only after restoring those archives. Full training entry points retain archived fixed namespaces and require a separate run/model namespace for a new experiment.

## Scientific result

Phase 2C-9B completed 196608 formal PPO decisions and 3932160 physical integration steps. All 72 final trials completed. History PPO improved mean unload IAE by about 10.75% versus single-frame PPO, while load IAE increased about 5.18%. Cross-seed/task benefit and superiority over classical adaptive control were not established. Stage decision: NEGATIVE with engineering training completed.
