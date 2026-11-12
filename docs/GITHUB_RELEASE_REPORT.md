# GitHub release report

## Final publication status — SOURCE UPLOAD VERIFIED; REPORT PUSH BLOCKED

The 947-file project content is remotely verified at 18bc7967a8e1a9aed22bfbcf000a42745cf73b3c. The subsequent report commit could not be pushed: two automatic approval review attempts disconnected before starting Git. Final local and remote HEAD equality is therefore pending. This is an approval-service execution failure, not a GitHub authentication or Git transfer rejection. Do not interpret the content verification below as final-HEAD equality.

Repository: https://github.com/lllppp1022-cmd/stewart-parallel-rl

Visibility: Private. Default branch: main.

The resumed command git -c http.version=HTTP/1.1 push -u origin main executed successfully, reporting Everything up-to-date and setting main to track origin/main. Git ls-remote and the GitHub tree API independently confirmed commit 18bc7967a8e1a9aed22bfbcf000a42745cf73b3c before this report-only update. API verification found the English and Chinese READMEs, src, scripts, configs, tests, docs, upstream model/license, all six terminal PPO checkpoints and demonstration videos/GIFs. The complete tree was not truncated.

Authentication used the existing Git Credential Manager credential and the portable official GitHub CLI at stewart_portfolio_build/tools/gh/bin/gh.exe. Its absence from the user's PATH does not affect Git push. SSL verification remained enabled; the push selected HTTP/1.1. Earlier approval-service read attempts intermittently failed before execution, but subsequent read and push commands completed normally. No force push, history replacement or visibility change occurred.

This report update is committed and pushed separately, followed by a final Git/API SHA equality check. The exact final local and remote HEAD values are stored in the external publication receipt to avoid a self-referential commit hash inside this file. Earlier BLOCKED/empty-repository entries below are retained as the publication history and are superseded by this final status.

Outstanding reproduction limitations remain the excluded bulk raw logs, historical fixtures and intermediate checkpoints. They do not prevent access to the published source, six final models and selected demonstrations. No new PPO training or experiment-result modification occurred during publication.

## Resumed publication check

The existing four local commits were preserved, with clean main at f87cfa1a70503a060f07b7bbd4159139b78967bb before this report update. Origin is the requested HTTPS repository. GitHub CLI authenticated as lllppp1022-cmd with repo scope. The read:org scope warning does not prevent access to this personal repository. CLI repository inspection confirmed Private visibility and an empty default branch. git ls-remote origin succeeded with no refs.

No environment proxy or Git HTTP proxy was present. Git uses Windows schannel. SSL verification was not disabled. The official PyTorch CPU index lists 2.15.0+cpu, matching the installed version and requirements.txt. The README command requesting torch==2.15.0 against that CPU index is compatible with the CPU local version, so no dependency change was necessary. A clean reinstall was not performed.

Two attempts to execute git -c http.version=HTTP/1.1 push -u origin main were rejected before execution because the automatic approval review stream disconnected. This is an approval service transport failure, not a finding that the push is unsafe. No push process ran during this resumed attempt. REMOTE UPLOAD VERIFIED cannot be asserted. Remote source, media and commit equality remain pending. Publication needs the approval service to recover or the user to execute the prepared normal push in their own PowerShell.

Release date: 2026-11-12, Asia/Shanghai.

Repository: https://github.com/lllppp1022-cmd/stewart-parallel-rl

Owner: lllppp1022-cmd. Visibility: Private, confirmed by GitHub CLI. Local branch: main. Remote default branch: not set because the repository is empty.

**Verification result: LOCAL RELEASE READY; REMOTE UPLOAD BLOCKED BY CONNECTION FAILURE.**

Validated complete content commit: 85d7dc6ae831002096bb48e22bf6f20a56168622. This report is updated in a subsequent local documentation commit. The final local HEAD is recorded in the external publication receipt.

The first push failed with: `fatal: unable to access 'https://github.com/lllppp1022-cmd/stewart-parallel-rl.git/': Send failure: Connection was aborted`.

A read-only follow-up confirmed the correct owner and Private visibility. The commits/main API returned HTTP 409, `Git Repository is empty.` The repository has no uploaded README or source yet. Network writes stopped under the user's instruction to stop publication on connection errors. No force push was attempted.

## Published content

Prepared locally: English and Chinese READMEs, original src/scripts/configs/tests directory layout, Phase 0 through Phase 2C-9B research documents, selected numerical summaries, upstream Stewart MJCF/assets with MIT notice, generated proxy geometry, geometric reference caches, six 32768-decision terminal policy checkpoints, three original demonstration videos, GIF excerpts and cover images. One 1000-step historical state prefix supports the default GUI acceptance check. These files have not been uploaded.

The upstream model commit is b0195ba9f40f6e0876220be1712eb47f5fc24025. THIRD_PARTY_NOTICES.md preserves attribution. No blanket MIT license is assigned to original project code.

## Excluded content

Bulk 1 ms physical journals, full integration-state archives, rollout arrays, intermediate/duplicate models, TensorBoard events, local temporary runs, unrelated robot projects and credentials remain local. Excluded raw data are needed for complete independent historical recomputation and the full research test suite.

No GitHub token, password, personal private key or local credential store is included. The official GitHub CLI used the existing Windows Git credential in memory for account verification and private repository creation. GitHub CLI was downloaded to the separate build directory and is not part of the published tree.

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
- Extracted a clean checkout from the complete Git archive into an independent directory, ran setup_portfolio.py and repeated the six-model/1000-physical-step/72-row metric smoke test successfully.

The initial smoke test failed because of sandbox temporary-directory permissions. It now writes isolated runtime outputs under this checkout. Subsequent smoke attempts identified two omitted runtime assets, nominal_task_model.json and phase1_proxy_query.xml, which were added before the passing test. The proxy asset mesh directory was made relative for portability. GUI validation initially required omitted full historical data; a small archived prefix was included for this explicit check. Original control algorithms and original research artifacts remain in the source project.

The initial staged-name parser encountered Git's quoted Unicode paths. The follow-up uses NUL-delimited names. A subsequent tracked-file check found that the upstream directory was not yet in the first content commit; its eleven licensed files were explicitly committed before the clean-archive verification. Local clone through MSYS was blocked by sandbox named-object permissions, so the independent checkout test used git archive. These local preparation failures are recorded separately from scientific experiment results.

## Reproducibility scope

Documentation and selected summaries are available. The model, core kinematics, home control, final policy loading and short dynamic execution passed local checks. The Windows GUI passed a bounded local run. Dependencies are pinned from the actual Python 3.11.17 environment, including PySide6_Essentials. A fresh network installation was not performed. Complete long training and complete historical tests were not repeated for this release. Historical acceptance reports retain the old Phase 0 learning-package assertion failure and initial sandbox test errors.

Run scripts/setup_portfolio.py after cloning. It resolves __PROJECT_ROOT__ tokens in metadata. Tests requiring omitted archives should be run only after restoring those archives. Full training entry points retain archived fixed namespaces and require a separate run/model namespace for a new experiment.

## Scientific result

Phase 2C-9B completed 196608 formal PPO decisions and 3932160 physical integration steps. All 72 final trials completed. History PPO improved mean unload IAE by about 10.75% versus single-frame PPO, while load IAE increased about 5.18%. Cross-seed/task benefit and superiority over classical adaptive control were not established. Stage decision: NEGATIVE with engineering training completed.

## Outstanding remote acceptance

Remote README rendering, GIF links, source/config/test presence and local-to-remote commit equality remain unverified because the private repository is empty. All three videos and all six terminal models are prepared locally but have not been uploaded. No Git LFS or GitHub Release write was needed for these individual file sizes.

After GitHub HTTPS connectivity is restored and publication resumes, the prepared command is git push -u origin main from the independent stewart-parallel-rl directory. The remote should then be reread using the official CLI, comparing main HEAD and inspecting README, configs, src, tests, media and checkpoint paths. Repository visibility must remain Private.
