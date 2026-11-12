# Phase 2C-6 验收与下一阶段决策

**PHASE 2C-6 PARTIAL SUCCESS — NEED TARGETED ABLATION**

独立策略对照的双 IAE 胜场以同权限经典反馈为参照。达到两个独立种子、每种子至少两个任务双 IAE 改善，同时满足安全/控制代价条件才可进入下一阶段。
V2 可用种子 [42, 123, 2026]，课程完整种子 [42, 123, 2026]，课程执行过的种子 [123, 2026, 42]。
有限的种子和时序不支持统计显著性判断。现有动作在轻中扰动具有可测的控制权，强扰动补偿余量较小。只有闭环独立收益支撑时，才考虑历史状态或更大运动任务。

| 新方法 | 双 IAE 胜场/3 | 平均加载 IAE/经典 | 平均卸载 IAE/经典 | 无保护且完成 | XYZ/变化率门槛 |
| --- | --- | --- | --- | --- | --- |
| V2_42 | 0/3 | 1.0763 | 0.9950 | True | True |
| V2_123 | 1/3 | 0.9926 | 1.0153 | True | True |
| V2_2026 | 1/3 | 1.0246 | 1.0360 | True | True |
| C2_42 | 0/3 | 1.0657 | 1.0011 | True | True |
| C2_123 | 1/3 | 0.9332 | 1.0180 | True | True |
| C2_2026 | 0/3 | 1.0971 | 1.0162 | True | True |

独立和强回归共 44 次真实闭环，44 次达到原协议。硬保护依据原限位、共享预算、weld 与部分几何覆盖。Qt 与无 GUI 的匹配确定性评价逐位核对 6000 个完整积分状态。

历史审计核对 11901 项持久文件，另外验证 3 处授权的测试检查器修复。唯一缺失例外为历史清单中的临时 Word 锁。修复 diff (omitted historical artifact: `../results/runs/phase2c6_20261107_v1/lock_fix.diff`) · 修复哈希 (omitted historical artifact: `../results/runs/phase2c6_20261107_v1/lock_fix_receipt.json`) · 历史核对 (omitted historical artifact: `../results/runs/phase2c6_20261107_v1/history_after.json`)

回归和阶段验收分三批执行，共 364 项通过、0 项失败、1 项历史预期失败。[测试汇总](../results/phase2c6_test_summary.json)

## 实际视频

- actual_v2_training (omitted historical artifact: `../assets/videos/stewart_phase2c6_actual_v2_training.mp4`)，450 帧、18.0 s，逐帧对齐 1 ms 归档状态哈希与时间。
- attitude_evaluation (omitted historical artifact: `../assets/videos/stewart_phase2c6_attitude_evaluation.mp4`)，1200 帧、48.0 s，逐帧对齐 1 ms 归档状态哈希与时间。
- curriculum_training (omitted historical artifact: `../assets/videos/stewart_phase2c6_curriculum_training.mp4`)，600 帧、24.0 s，逐帧对齐 1 ms 归档状态哈希与时间。
- reward_ablation (omitted historical artifact: `../assets/videos/stewart_phase2c6_reward_ablation.mp4`)，600 帧、24.0 s，逐帧对齐 1 ms 归档状态哈希与时间。

图上角度尺度单独标注。实际机器人画面来自训练或评价现场 RGB，几何保持真实比例。

若继续研究，优先做等交互预算的固定分布续训对照，分离课程分布与额外更新的影响。再单独审计不同扰动强度的价值目标尺度及卸载后的补偿方向，继续沿用当前动作权限和安全门槛。

## Windows 查看命令

```powershell
Set-Location __PROJECT_ROOT__
$py = 'D:\miniconda\envs\stewart_rl\python.exe'
& $py scripts/view_phase2c6_policy.py --task E_mix
& $py -m tensorboard.main --logdir results/runs/phase2c6_20261107_v1/training --host 127.0.0.1 --port 6036
```

本轮在 Phase 2C-6 停止。
