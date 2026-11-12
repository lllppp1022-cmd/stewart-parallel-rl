# Phase 2C-8 训练协议与随机数审计

日期：2026-11-09（Asia/Shanghai）；独立实验目录：`__PROJECT_ROOT__\results\runs\phase2c8_20261109_v1`。所有数值由原始CSV/NPZ/JSON生成，不修改历史结果。

## 历史冻结与运行环境

历史冻结清单 `history_before.json` 共 41396 个文件，事后逐文件SHA核验 41396 个、变化 0 个。旧模型、控制器、场景、Reward-V2、7/7V视频均保留。新四份配置的训练前SHA位于 `config_freeze.json`。实际执行实现SHA另存 `executed_implementation_v1.json`，此源码快照在Pilot结束后登记，不伪称为训练前时间戳。

Python：`D:\miniconda\envs\stewart_rl\python.exe`；版本 `3.11.17 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:16:54) [MSC v.1942 64 bit (AMD64)]`；系统 `Windows-10-10.0.26200-SP0`。

| package | actual_version |
| --- | --- |
| mujoco | 3.17.0 |
| stable-baselines3 | 2.9.2 |
| torch | 2.15.0+cpu |
| numpy | 2.4.6 |
| gymnasium | 1.4.0 |
| pandas | 3.1.0 |

训练使用CPU、每进程1线程；当前Torch为CPU构建，CUDA available=False。不把用户历史GPU型号当作本次GPU训练证据。六个独立训练进程没有共享随机数或物理对象。

## 冻结的控制与PPO

物理1ms、传统控制1000Hz、PPO50Hz，每决策20个真实 `mj_step`；58维原因果观测；2维归一化高斯动作经裁剪后映射每轴±0.05N·m。C4、fourDlead和FF-VA未改；共享补偿相对C0每支链±5N、20N/s，原生±400N；原工作区、姿态、Jacobian和weld检查保留。实际驱动力变化率不等于“相对C0附加补偿”的20N/s限速，二者分别记录。

| PPO设置 | 实际冻结值 |
| --- | --- |
| n_steps | 512 |
| batch_size | 64 |
| n_epochs | 10 |
| learning_rate | 0.0003 |
| gamma | 0.9959879355809816 |
| gae_lambda | 0.9796917302662298 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| max_grad_norm | 0.5 |
| target_kl | 0.03 |
| net | [64, 64] |
| log_std_init | -1.0 |
| device | cpu |
| seeds | [42, 123, 2026] |
| pilot | 8192 |
| total | 32768 |
| return_normalization | False |

Actor/Critic分别64×64，初始Actor均值为零、log_std=-1；这是零均值随机探索，不是零动作训练。V2/V3同种子初始参数哈希一致，六组均从新初始模型出发。旧7期模型只在离线奖励审计中作为historical42，不能代替新V2匹配对照。

## RNG修复与恢复边界

任务抽样改为独立 `PCG64(SeedSequence([training_seed,8008]))`。Python/NumPy全局/Torch动作RNG与任务RNG分别保存。每episode记录task、外载、参考、阶段、抽样前后状态及Gym收到的seed。`gym.Env.reset(seed=...)`只处理自身RNG，不重置任务采样器。

SB3 load/set_env后清空VecEnv待应用seed，再恢复全局RNG；独立任务采样器也防止未来reset seed传播再次覆盖任务序列。Stage A/B分别4096步，在课程切换处显式结束预算片段并reset。A最后一次抽样后状态等于B首次抽样前状态，V2/V3每个种子的实际任务序列完全相同。

实际连续/恢复试验：A前缀512决策，连续B512与保存重载B512两条分支；总计1536测试决策、30720真实物理步。两分支动作、观测、奖励、rollout目标、物理状态、最终参数逐位相等。详见 `resume_validation.json`。此证明只覆盖声明的课程边界，**不支持任意回合中间崩溃后无缝恢复**；这种失败必须归档重做，不能静默reset冒充连续。

## 数据划分与实际预算

训练复用18个既有精确联合预检任务配置，A/B/C各阶段候选保留；新Dev为D8_roll、D8_pitch及D8_mix，其中D8_mix与旧Dev_mix物理任务一致，明确不是全新开发任务。Final四个任务的完整参考/外载组合与旧任务不同，配置训练前冻结。参考轨迹几何缓存复用旧审核资产，并非所有轨迹几何均首次出现。新Dev和Final先做14次零动作/有界随机联合安全预检；这属于准入检查，不是策略选择。所有最终PPO策略评价在 `final_selection.json` 冻结后才开始。

| task | z_mm | roll_deg | pitch_deg | rise_s | Mx_Nm | My_Nm | on_s | off_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T8_roll | 50.000000 | 0.100000 | 0.000000 | 5.500000 | -0.900000 | 0.000000 | 3.700000 | 5.600000 |
| T8_pitch | 50.000000 | 0.000000 | -0.100000 | 6.500000 | 0.000000 | 1.050000 | 9.700000 | 11.400000 |
| T8_mix | 50.000000 | -0.100000 | 0.100000 | 6.500000 | -0.660000 | -0.880000 | 7.600000 | 9.300000 |
| T8_combined50 | 50.000000 | 0.250000 | -0.250000 | 6.000000 | 0.960000 | -0.720000 | 4.900000 | 6.700000 |

| method | seed | new_decisions | physical_steps | complete_episodes | fragments | updates | adam_steps | failures | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_control | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.071634 |
| v2_control | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.080746 |
| v2_control | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 818.692782 |
| v3 | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.266636 |
| v3 | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.062844 |
| v3 | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 819.255763 |

正式共 49152 决策、983040 物理步，96 次PPO更新、7680 次Adam。没有将恢复验证1536决策算进正式预算。每组10个完整回合、2个课程/预算片段；片段未被标成失败或完整回合。未发生训练保护失败。

共同开发门禁：至少2/3种子同时改善加载和卸载均值≥1%；每个种子IAE退化不得>10%；各任务SO3峰、角速度、Z/XYZ/XY、实际力变化率及力平方积分不得超过匹配V2的110%；全部安全完成。此门槛训练前冻结。

| seed | all_task_cost_guard | safe_complete | promising | no_large_iae_regression | load_improvement_pct | unload_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | True | True | False | True | 2.710042 | -1.263222 |
| 123 | True | True | False | True | 0.690514 | 3.517143 |
| 2026 | True | True | False | True | -3.721075 | -3.435162 |

结果门禁失败，**六组均在8192停止，Stage C、32768扩展和ReturnNorm正式训练未执行**。没有只追加表现好的种子。保存所有终端checkpoint并进入最终评价，没有从最终测试选择best seed。

## 真实积分账本

| 运行类别 | 已归档物理步 |
| --- | --- |
| development | 900000 |
| final | 488000 |
| preflight | 212000 |
| reward_audit | 75000 |
| training | 983040 |
| validation | 62260 |
| viewer | 2000 |

共 2722300 步（含准入、奖励审计、开发、最终评价、恢复/零动作/归一化验证及两个GUI1000步测试）；历史pytest包含多种日志格式，未将其混入这个统一账本。关闭/渲染不增加隐藏物理步，重置初始化没有计作动态运动。

## 模型证据

| model | checkpoint | SHA256 |
| --- | --- | --- |
| v2_control/seed42 | __PROJECT_ROOT__\models\phase2c8\v2_control\seed42\checkpoint_008192.zip | 07051a28b2569660a1aacaabce8df8273bbd17754580addd633049ab13f041be |
| v2_control/seed123 | __PROJECT_ROOT__\models\phase2c8\v2_control\seed123\checkpoint_008192.zip | 8ea1658d54f04acb64def5962e791e166c6e2fff9c69b5383eb7a7ad43251542 |
| v2_control/seed2026 | __PROJECT_ROOT__\models\phase2c8\v2_control\seed2026\checkpoint_008192.zip | b9fb9593725adbe5a41789138cbbf113979b99d1047aa50407e75d9491ddba0c |
| v3/seed42 | __PROJECT_ROOT__\models\phase2c8\v3\seed42\checkpoint_008192.zip | 674eccc030b4618ad24c1b3438aaae6794b3a0b69cb3e9778831b8189023580b |
| v3/seed123 | __PROJECT_ROOT__\models\phase2c8\v3\seed123\checkpoint_008192.zip | ef19350cb4187af6a3072f8cf4f3c7b9ae2ef56366cb50e837572a6cfcbd8021 |
| v3/seed2026 | __PROJECT_ROOT__\models\phase2c8\v3\seed2026\checkpoint_008192.zip | ad13d2e1a0025eb1caa32ab272cebd249853b5d2d5a1118d594a38ef127d18d0 |

每512决策的update模型、0/4096/8192 checkpoint、边界RNG/优化器均保留。96个rollout的GAE及bootstrap独立复算通过；未覆盖任意中途恢复这一限制仍保留。
