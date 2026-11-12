# Project structure and execution dependencies

- src/ contains full-chain kinematics, traditional control, disturbance estimation, dynamics, Gymnasium environments and the shared viewer/recording implementation.
- scripts/ preserves original imports and includes training, evaluation, analysis and visualization entry points.
- configs/ contains frozen stage/task/PPO settings.
- third_party/stewart_platform_mujoco/ contains the MIT-licensed upstream model and assets.
- models/phase2c9b/ contains the six final 32768-decision policies.
- results/ contains selected summaries, necessary geometric caches and final metadata. Raw experiment trajectories remain outside the release.
- tests/ contains original research tests, many of which require omitted historical archives. scripts/portfolio_smoke.py validates the distributed subset.
- docs/ preserves the research timeline and its negative results. All published local paths use __PROJECT_ROOT__.
- release/source_manifest.json maps copied files to original and sanitized SHA256 hashes.

Run scripts/setup_portfolio.py once after cloning. This converts portable tokens in JSON/CSV/Markdown metadata into paths under this checkout. The source snapshot files stay available in git. No model weights or control laws are transformed.
