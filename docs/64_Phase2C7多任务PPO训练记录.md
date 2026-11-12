# Phase 2C-7 多任务PPO训练记录

三种子真实完成新增98304次PPO决策、1966080次MuJoCo积分。这里不计预检、测试和独立评价的额外物理成本。每次正常动作20个1 ms积分步。Pilot共同门禁通过后，三个种子均获得相同的累计32768决策预算。

| seed | decisions | physical_steps | updates | adam_steps | complete_episodes | failures | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 32768 | 655360 | 64 | 5120 | 32 | 0 | 2129.86 |
| 123 | 32768 | 655360 | 64 | 5120 | 32 | 0 | 2104.44 |
| 2026 | 32768 | 655360 | 64 | 5120 | 32 | 0 | 2107.8 |

上表continue行的决策/物理步/更新是累计数；完整回合和墙钟为该次Stage C进程自身值。全部阶段的真实组成如下，预算截断的不完整回合保留，不能当成完整成功回合。

| seed | phase | stage | decisions | physical_steps | reward_mean | reward_min | failures | task_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | pilot | A | 4096 | 81920 | -0.480355 | -11.5504 | 0 | A0,A2,A3,A4 |
| 42 | pilot | B | 4096 | 81920 | -0.256627 | -11.6069 | 0 | B0,B1,B3,B4,B5 |
| 42 | continue | C | 24576 | 491520 | -0.209817 | -6.17203 | 0 | C0,C1,C2,C3,C4,C5,C6 |
| 123 | pilot | A | 4096 | 81920 | -0.267432 | -11.1515 | 0 | A0,A1,A2,A3,A4 |
| 123 | pilot | B | 4096 | 81920 | -0.272238 | -5.15247 | 0 | B1,B2,B4 |
| 123 | continue | C | 24576 | 491520 | -0.194584 | -6.37787 | 0 | C0,C1,C2,C3,C4,C5,C6 |
| 2026 | pilot | A | 4096 | 81920 | -0.218606 | -10.946 | 0 | A0,A1,A2,A3,A4 |
| 2026 | pilot | B | 4096 | 81920 | -0.4585 | -11.5218 | 0 | B0,B2,B3,B4 |
| 2026 | continue | C | 24576 | 491520 | -0.207172 | -5.93435 | 0 | C0,C1,C2,C3,C4,C5,C6 |

## 价值估计诊断

| seed | updates | adam_steps | max_value_loss | minimum_explained_variance | maximum_target_std | minimum_target | maximum_KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 64 | 5120 | 11009.2 | -14.2381 | 91.864 | -302.736 | 0.0108505 |
| 123 | 64 | 5120 | 10636.2 | -3.01814 | 90.4832 | -290.659 | 0.0150931 |
| 2026 | 64 | 5120 | 11291.9 | -4.80452 | 92.5201 | -308.467 | 0.0118195 |

早期1.2 N·m Home任务出现约−300量级的价值目标和约1万量级Value Loss。原始物理记录显示该任务单决策cost约最高11，持续外载下误差累积，使GAE/价值目标比轻载任务大。大Loss本身不等于数值发散；同时检查有限性、目标分布、解释方差、完整回合和安全记录。本轮没有因最终测试结果修改奖励、网络、归一化或追加预算。

共记录135个训练回合片段，其中126个完整结束、9个被课程边界或预算截断；没有把后者计为完整成功或失败。所有训练回合片段的奖励cap率及观测裁剪率均为0，详见`phase2c7_reward_diagnostics.csv`。实际运行版本见`runtime_versions.json`，本次使用MuJoCo 3.17.0、SB3 2.9.2、CPU版PyTorch，不能套用早期阶段的版本声明。

每512决策保存一次完整rollout、Actor/Critic变化范数、实际Adam步数、模型SHA、Value Loss/KL/EV/优势与价值目标分布。`training_samples.csv`保存原始高斯动作、裁剪动作、实际接受残差、Actor均值/标准差及奖励五项。每个回合保留1 ms物理CSV与完整积分状态；选定seed42回合现场录制RGB。

原始高斯动作是float32，CSV使用NumPy最短可往返字符串。核验时恢复float32后与全部rollout逐值完全相等；不能把十进制CSV默认解析为float64后的约1e−8表示差异当成动作不一致。初次采用1e−15 float64容差的失败测试日志保留，修正核验数据类型后通过，未改写任何原始动作数据。

模型目录为`models/phase2c7/seed42`、`seed123`、`seed2026`。包含initial、pilot、逐次update及final。Pilot与Stage C之间明确结束预算截断回合并开始新完整回合。网络和优化器继续，PyTorch/NumPy/Python全局随机状态恢复；SB3加载后首次VecEnv reset会按各自训练种子重新播种环境任务采样。实际Stage C任务序列已与该种子的PCG64序列核对（`stage_c_rng_audit.json`）。虽然代码赋回了保存的环境RNG状态，该首次reset会覆盖它，因此不能宣称环境随机流无缝续接，更不能宣称恢复了被截断回合的连续物理现场。三个种子均使用相同规则，没有挑选采样顺序。

## 查看与训练命令

以下训练命令已在本次实验实际运行。输出目录有存在检查，重复运行不会覆盖旧模型。再次训练应先创建新的独立实验配置和输出目录，不能删除本轮目录后冒充复现。

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/train_phase2c7_ppo.py --seed 42 --phase pilot
# 123、2026也执行相同Pilot；完整开发评价和共同门禁通过后：
& D:\miniconda\envs\stewart_rl\python.exe scripts/train_phase2c7_ppo.py --seed 42 --phase continue
```

TensorBoard日志已实际通过`--inspect`验证。下面是启动网页的命令；网页服务需在终端保持运行，Ctrl+C关闭。

```powershell
& D:\miniconda\envs\stewart_rl\python.exe -m tensorboard.main --logdir results/runs/phase2c7_20261107_v1/training --port 6032
```

网页地址为http://127.0.0.1:6032。旧6031页面若仍指向历史目录，不会自动显示新阶段曲线。

以下复算/视频命令也已经实际执行。它们读取本次原始数据并更新本阶段派生产物，不重新训练，也不改写旧阶段实验。

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts/collect_phase2c7.py collect
& D:\miniconda\envs\stewart_rl\python.exe scripts/render_phase2c7_videos.py dynamic_ppo_training
& D:\miniconda\envs\stewart_rl\python.exe scripts/render_phase2c7_videos.py classical_vs_ppo
& D:\miniconda\envs\stewart_rl\python.exe scripts/render_phase2c7_videos.py dynamic_disturbance_demo
```
