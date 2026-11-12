# Phase 2C-8 奖励消融真实PPO训练

日期：2026-11-09（Asia/Shanghai）；独立实验目录：`__PROJECT_ROOT__\results\runs\phase2c8_20261109_v1`。所有数值由原始CSV/NPZ/JSON生成，不修改历史结果。

六组真实训练已经完成，不是只创建代码。两方法相同初始化、任务RNG、场景、控制器、动作/观测、PPO配置、预算与安全约束，只改变声明的Reward公式。三种子为42、123、2026。

| method | seed | new_decisions | physical_steps | complete_episodes | fragments | updates | adam_steps | failures | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_control | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.071634 |
| v2_control | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.080746 |
| v2_control | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 818.692782 |
| v3 | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.266636 |
| v3 | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.062844 |
| v3 | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 819.255763 |

每组16次PPO更新（512决策一次），每次80次Adam，共7680次Adam；所有Actor/Critic变化范数>0，保存模型重载参数哈希一致。实采原始高斯动作、裁剪动作、接受动作、58维观测、奖励、价值目标均保存。随机训练与确定性评价分开。

## Pilot决策

| seed | all_task_cost_guard | safe_complete | promising | no_large_iae_regression | load_improvement_pct | unload_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | True | True | False | True | 2.710042 | -1.263222 |
| 123 | True | True | False | True | 0.690514 | 3.517143 |
| 2026 | True | True | False | True | -3.721075 | -3.435162 |

没有任何一个V3种子同时达到开发加载和卸载各≥1%的改善，要求至少两个种子。虽然均安全且各代价护栏通过，仍然不晋级。V2和V3公平停在8192；继续32768、16384/24576/32768 checkpoint及ReturnNorm训练模型均**未产生、未执行**。

## 完整回合与课程片段

60个完整回合，12个课程切换/最终预算片段。片段仍保留原始物理日志与部分return，不加入完整episode return均值。一次rollout可跨episode，一个episode可跨512决策更新，`policy_version`逐物理步记录，不假设整个episode权重固定。

开发在0、4096、8192三个checkpoint对三个固定任务分别评价，另有传统对照，共60次真实闭环、900000物理步。所有评价0次梯度更新。统一V2只读计分用于跨奖励对比；这不改变训练reward。

相对各自初始策略的固定开发任务变化（各任务等权，正百分比为误差改善）：

| method | seed | common_v2_return_initial | common_v2_return_final | load_iae_deg_s_improvement_pct | unload_iae_deg_s_improvement_pct | xyz_rmse_mm_improvement_pct | return_up_but_any_iae_worse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_control | 42 | -80.323748 | -71.029570 | 4.265562 | 3.062125 | 3.108070 | False |
| v2_control | 123 | -80.323748 | -74.828106 | 0.896227 | -2.573409 | 2.875654 | True |
| v2_control | 2026 | -80.323748 | -79.848024 | -1.301616 | -2.853563 | 0.260900 | True |
| v3 | 42 | -80.323748 | -72.579520 | 6.860005 | 1.837584 | 2.594752 | False |
| v3 | 123 | -80.323748 | -72.288301 | 1.580553 | 1.034245 | 3.262346 | False |
| v3 | 2026 | -80.323748 | -87.842415 | -5.071125 | -6.386749 | -2.086710 | False |

其中 2 组出现统一V2回报提高、但至少一个IAE恶化。这是实测的目标取舍，不只是理论可能性；V3 seed2026的统一回报和两个IAE也未同步改善。该表与“V3相对匹配V2”的晋级门禁是两个不同对照，不能混用。

## 学习曲线与TensorBoard

训练奖励与分量 (omitted historical artifact: `../assets/plots/phase2c8_training_reward.png`)

Critic与优势 (omitted historical artifact: `../assets/plots/phase2c8_critic_learning.png`)

固定开发任务 (omitted historical artifact: `../assets/plots/phase2c8_fixed_task_learning.png`)

已启动并实际HTTP200验收：[本轮TensorBoard](http://127.0.0.1:6034/)。18个run：12个原始A/B事件流、6个从原始CSV复算的全程derived事件流。derived不是新训练；`actual/`是SB3实际事件。旧6031/6032/6033面板未改。

查看derived的 `development/*` 固定任务曲线，并同时看加载/卸载IAE、SO3/XYZ，不能只看训练return。累计episode reward每决策会累积，但完整episode return只在结束时更新；更新指标每512决策出现。训练已停止，所以刷新页面不会继续产生训练点。

## 已实际执行的命令

下面是六组已运行命令的形式，用于记录来源，不要在已有输出目录上重复运行：

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/train_phase2c8_reward_ablation.py --method v2_control --seed 42 --phase pilot
& D:\miniconda\envs\stewart_rl\python.exe scripts/train_phase2c8_reward_ablation.py --method v3 --seed 42 --phase pilot
```

同样命令分别使用seed123、2026，共六组。独立复现实验需要先配置新的RUN和模型输出命名空间并重新冻结配置，当前脚本固定本轮归档路径，不提供会覆盖本轮结果的“一键重复训练”。不运行continue：它的冻结门禁为false。

只读重汇总/指标复算可执行：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts/evaluate_phase2c8.py collect
& D:\miniconda\envs\stewart_rl\python.exe scripts/audit_phase2c8_final.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/audit_phase2c8_gae.py
```

这些命令实际运行通过，只写本轮派生汇总/审计文件，不训练模型。每个种子的原始physics、integration_states、RGB帧、rollout NPZ、task_sequence和TensorBoard位于独立 `training/<method>/seed<seed>/<A或B>`。
