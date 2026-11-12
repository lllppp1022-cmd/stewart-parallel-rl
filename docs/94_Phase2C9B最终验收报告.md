# Phase 2C-9B 最终验收报告

日期2026-11-12，Asia/Shanghai。独立归档 `__PROJECT_ROOT__\results\runs\phase2c9b_20261112_v1`。数值从实际CSV/JSON/NPZ重算。

**NEGATIVE**。工程训练完成与历史控制收益分别判定。正式完成196608决策、3932160真实物理步，最终评价72次，其中72次完整。历史是否比单帧与经典适应更好的结论，见报告92/93逐任务及逐种子结果。

全任务等权均值中，历史相对单帧的加载IAE变化为+5.181%，卸载IAE变化为-10.752%，XYZ-RMSE变化为+1.161%（正号代表误差增加）。历史相对经典适应的加载IAE变化为-0.071%，卸载IAE变化为+0.190%。均值存在加载/卸载取舍，不能用一个较低的均值证明逐任务一致收益。

| method | load_iae_deg_s | unload_iae_deg_s | so3_peak_deg | omega_peak_rad_s | z_rmse_mm | xyz_rmse_mm | xy_peak_mm | force_rate_N_s | force_squared_integral_N2_s | trials | complete_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | 0.008142953 | 0.001711776 | 0.01550442 | 0.004548312 | 0.02064156 | 0.106775 | 0.2310752 | 55.77894 | 654277.2 | 8 | 8 |
| baseline | 0.00826866 | 0.00173431 | 0.01566663 | 0.004549443 | 0.02064354 | 0.1078529 | 0.2333898 | 55.77888 | 654277.2 | 8 | 8 |
| classical50 | 0.008198825 | 0.001707345 | 0.01555567 | 0.004549105 | 0.02064306 | 0.1071818 | 0.2318699 | 55.77899 | 654277.2 | 8 | 8 |
| history_5frame | 0.008137195 | 0.001715035 | 0.01542578 | 0.004546252 | 0.02064523 | 0.1018865 | 0.2267767 | 55.81502 | 654276.5 | 24 | 24 |
| single_frame | 0.00773639 | 0.001921659 | 0.01477843 | 0.004550909 | 0.020625 | 0.1007173 | 0.2239247 | 55.77954 | 654277.8 | 24 | 24 |

| method | seed | decisions | physics | complete | fragments | failures | wall_s | ppo_updates | adam_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| history_5frame | 42 | 32768 | 655360 | 42 | 2 | 0 | 3565.515 | 64 | 5013 |
| history_5frame | 123 | 32768 | 655360 | 42 | 2 | 0 | 3535.053 | 64 | 5082 |
| history_5frame | 2026 | 32768 | 655360 | 42 | 2 | 0 | 3532.015 | 64 | 5120 |
| single_frame | 42 | 32768 | 655360 | 42 | 2 | 0 | 3560.418 | 64 | 5120 |
| single_frame | 123 | 32768 | 655360 | 42 | 2 | 0 | 3538.838 | 64 | 5120 |
| single_frame | 2026 | 32768 | 655360 | 42 | 2 | 0 | 3533.351 | 64 | 5120 |

历史冻结核验45735文件、变化0个。测试执行统计如下，失败详情和首次沙箱错误均保留。

| run | total | passed | failed |
| --- | --- | --- | --- |
| historical_tests | 455 | 385 | 70 |
| historical_tests_unrestricted | 455 | 454 | 1 |
| new_tests | 27 | 27 | 0 |

原Phase 0“环境不得安装Torch/SB3”断言与后续训练环境冲突，保留失败。首次沙箱运行pytest临时目录产生访问错误，使用独立目录重跑，并明确区分执行环境错误与代码失败。新增测试对真实模型、物理日志、GAE和全种子预算作核验，没有删除历史失败。

全量审计首次因float64标量GAE递推与SB3 float32向量递推的舍入差异失败（最大超差约8.6×10⁻⁵）；失败日志full_audit.log保留。随后按训练库实际float32逐步运算重算，未修改任何训练权重、奖励或已发生的样本。

提前检查点评价工具最初启动了三条同进程线程试验，随后终止并改为独立进程执行。三条未完成启动记录保存在 abandoned_thread_evaluation，未纳入开发指标或策略选择。该工程尝试的全部实际步数不能从终止记录精确恢复，因此不声称包含这部分的全工程积分总数精确已知。正式PPO、完成的预检和评价均有独立完整账本。另一次ProcessPool尝试受Windows沙箱命名管道访问限制而未开始物理积分，日志保留；最终使用独立Python子进程。

| file | frames | seconds | sha |
| --- | --- | --- | --- |
| stewart_phase2c9b_unknown_mass.mp4 | 800 | 32 | da10d3470208e58faca7ab2949da7802bac73f5d22bdd1fda23e4f30abfceb16 |
| stewart_phase2c9b_history_training.mp4 | 1500 | 60 | 1dd2faffa60e709e3450fdb27a19e24a87fb1efc021bef4a8ae3ca85c13f6707 |
| stewart_phase2c9b_single_vs_history.mp4 | 3150 | 126 | fd401b4e831d5f37457321e7906e69fec9622dcb251e7a37a6de4d7996fba43a |

视频来自训练现场或冻结策略真实积分，逐帧时间和qpos SHA与原日志对应。训练视频标记ACTUAL STOCHASTIC TRAINING，评价标记FROZEN DETERMINISTIC EVALUATION。机器人保持真实物理角度，曲线纵轴辅助显示微小误差。全部种子在最终比较表及比较视频中保留。

PowerShell打开冻结策略GUI，两个方法均在1000步自动验收模式检查与headless前缀一致。任务选择绑定实际质量、方向及加载卸载时序，只允许已通过评价的组合。

GUI顶部先选任务、控制器和训练种子，然后点击Reset new episode，选择才会作用于新的物理回合；状态栏ACTIVE始终显示正在执行的方法、种子和真实质量。Pause/resume暂停继续，拖动机器人画面调整相机，Close and archive或关闭窗口正常归档。这里是冻结策略实时动力学评价，不是新的训练；训练随机探索请观看现场训练视频或TensorBoard。

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/view_phase2c9b_policy.py --method history_5frame --seed 42 --task Final_m21.5_6
& D:\miniconda\envs\stewart_rl\python.exe scripts/view_phase2c9b_policy.py --method single_frame --seed 42 --task Final_m21.5_6
```

TensorBoard实际日志为实验目录training，各方法/种子/预算阶段独立。完整开发曲线和奖励分量还可以从CSV重算。当前服务器地址及HTTP验收见 tensorboard_validation.json，固定端口6037。

访问 http://127.0.0.1:6037/#timeseries 。本轮服务实际读取18个run：两阶段训练日志12个及完整课程derived日志6个。推荐先选derived/history_5frame_seed42或对应单帧/其他种子，查看episodes/complete_return、fixed_development/load_iae_deg_s、fixed_development/unload_iae_deg_s及optimization/train/value_loss。五个固定开发点0、8192、16384、24576、32768已通过HTTP读取核验。不要把旧6031端口的其他阶段曲线当成本轮结果。训练已完成停止，图表保持不再增长属于正常现象。

```powershell
& D:\miniconda\envs\stewart_rl\python.exe -m tensorboard.main --logdir results/runs/phase2c9b_20261112_v1/training --host 127.0.0.1 --port 6037
```

实际复算入口：analyze_phase2c9b.py training、analyze_phase2c9b_history.py、plot_phase2c9b.py。训练原始目录已存在，重复训练须使用新的独立命名空间；既有正式目录不会覆盖。

本轮仍只有有限轨迹、质量和随机种子。评估为固定策略利用历史反馈，评价阶段不更新网络。未执行实机实验，几何代理不完整，离散预检不提供连续安全保证。对任意回合中途崩溃的无缝恢复未作保证，显式预算边界恢复通过。

测量沿用理想位姿与支链状态反馈，没有在本轮增加传感器噪声。驱动滞后固定为5 ms合成一阶模型，质量采用均匀密度等效变化；对偏心附加载荷、未知液压动态或实机噪声的结果尚未验证。

明确未执行/未完成：闭环历史信息替换（本轮完成的是只读输入敏感性）；匹配容量对照未触发、未训练；任意回合中途的无缝恢复；实机与噪声泛化。历史环境隔离测试仍有1项失败，没有宣称全部历史测试通过。以上均不作为已完成实验。

本阶段结束后暂停，等待审核。
