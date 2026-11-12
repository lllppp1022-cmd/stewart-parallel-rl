# Stewart Phase 2B-3：执行器增益失配、Yaw补偿与任务筛选

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 正式结论

**PHASE 2C BLOCKED**。四维Yaw通道有效，C4-Observer已用允许的因果信息解决主要增益Yaw偏置。额外因果残差仍有小数值改善，但没有证明值得在最强传统基线上增加PPO训练。完成本轮后暂停，不创建Gymnasium或训练任何网络。

## 实际运行清单与成本

| stage | trials | successes | physics_steps |
| --- | --- | --- | --- |
| action_preflight | 8 | 8 | 48000 |
| development | 12 | 12 | 72000 |
| frozen_evaluation | 30 | 30 | 180000 |
| oracle_diagnostic | 2 | 2 | 12000 |
| safety_preflight | 4 | 4 | 24000 |
| zero_equivalence | 2 | 2 | 12000 |

有效动力学共58回合，348000个1ms积分步（348仿真秒），全部按原任务完成。另1次工程尝试在0步失败，共59次尝试，未超过预先66回合最大研究预算。0步异常不混入动力学成功率分母，也没有删除。原±1%安全失败只做离线分析，无重新执行。

主目录：`results/runs/phase2b3_20261029_v1`；缺失工况续跑：`results/runs/phase2b3_20261029_continuation_v2`；索引`phase2b3_run_index.json`。两者是同一冻结矩阵，不是两轮调参或择优重跑。每个有效回合6000步，保存完整CSV.gz、NPZ、配置、plant、action_profile、原metrics及独立metrics_recomputed、事件、summary。没有神经网络checkpoint，因为没有训练。

额外观察器验收：独立off/on各2048步（共4096），原生MuJoCo显示镜像+Qt+现场RGB开启/关闭的完整458维integration逐位一致，最大差0、无中断、正常退出。自动测试里的短动力学归测试成本，不混作科学实验。

## 全量测试与独立复核

**212项测试全部通过**，包含原189项，新增17项单元与6项实际证据测试。原代码/模型/全部4814历史文件SHA256不变。保留原pandas弃用警告2条。第一轮新增单测15过2失败（错误假设世界Z外载的局部Fx/Fy必须零、错误字段dt），修正新测试后17通过，未改控制实现或旧测试来通过；首轮失败日志保留。后发现参考调度接口duration缺失，修复并回归验证，0步异常完整保留。

`phase2b3_validation.json`复核58回合：原始CSV与NPZ完整qpos/qvel/time逐位一致，状态有限；actual force=gain×cmd（按原native clip）；共享±5N/20N/s和原±400N通过；后六ctrl只来自外载，未混入残差；无额外噪声、延迟、质量、COM；独立scratch重算预积分映射最大差0；全部物理回合无MuJoCo警告。安全指标按回合详见完整CSV，不以自动测试替代物理实验。

## 关键结果

| variant | yaw_peak_deg | yaw_IAE_deg_s | position_rmse_mm | max_force_rate_N_s | shared_add_peak_N | shared_add_rate_N_s |
| --- | --- | --- | --- | --- | --- | --- |
| C3 | 0.072464 | 0.304304 | 0.099751 | 34.276039 | 0.462415 | 16.139786 |
| C3_4d20 | 0.017787 | 0.103389 | 0.099053 | 34.259655 | 0.695178 | 20.000000 |
| C4-Observer | 0.006965 | 0.004136 | 0.098903 | 37.568773 | 0.723465 | 19.232843 |
| C4-PD | 0.017787 | 0.102979 | 0.099116 | 36.887400 | 0.651467 | 18.562608 |
| C4_4d20 | 0.005268 | 0.003084 | 0.098845 | 37.565361 | 0.760772 | 20.000000 |

上表是6个确定性冻结工况的描述平均，非统计显著结论。全部正式方法各6/6完成且无保护；这不等于全程都在严格.25mm/.02deg精度带，也不等于全机构碰撞或实机认证。

## 20个研究问题

1. **增益为何Yaw偏移？** 正支撑力乘交替gain误差，乘交替Yaw力臂后同号叠加；±.5%约-.234Nm，角速度/姿态方向与记录一致。
2. **原C3估计准确吗？** 局部小幅范围明显响应，±.5%均值约-.234345Nm，与离线真值近似；非精确传感器，仍有建模残差。
3. **四维能控制Yaw？** 八个正负探针真实积分方向正确，完整零动作等价，非可视化假动作。
4. **非目标耦合多少？** 逐回合CSV有XY/Z/R/P峰值；C4平均XY约.150mm，单个动态工况约.338mm，不能声称无耦合。
5. **PD还是Observer？** 本模型理想观测下Observer更好；开发与最终一致。未做新噪声/延迟下C4鲁棒性，不能外推硬件。
6. **共享预算保持？** 是，C3+Yaw+额外lead合计每腿±5N、20N/s，未加第二预算。
7. **±.5%能改善？** 开发无外载Home/periodic中C4明显降低Yaw；最终不同组合亦有效。
8. **±1%能解释？** 离线真实力矩约加倍，负Yaw越界，观测器有响应但未被C3主动补偿；局部方程留有残差，不做全局证明。
9. **需立即重跑±1%？** 不需要、未重跑；当前安全权限和任务选择不要求把已失败案例变成探索环境。
10. **加载前已出现？** 是，历史失败.384s早于2s外载；新无载保持和前载指标单列。
11. **加外载仍有效？** 独立Z/Roll/Pitch/动态组合均完成，Yaw压低；各轴代价详表，没有筛掉退化项。
12. **其他轴是否更差？** 部分工况有微小位置退化，例如skew_pitch，已报告。C4未承诺所有轴最优。
13. **控制代价取舍？** Observer平均实际总力峰值变化率比C3约增9.6%；共享限速仍合法；力平方积分近似，不能包装节能。
14. **哪些特权？** Oracle读真实gain/当前外载/实际映射，仅诊断；C3、PD、Observer、两个20Hz补偿均无真值/未来输入。
15. **经典Yaw能解决主要问题？** 是，当前理想反馈及小幅未知gain范围，Observer六个最终工况Yaw峰值都在.02°带内。
16. **残差超过C4的价值？** 有小数值改善，但未形成足够额外收益；平均位置RMSE改善约.000058mm，不能直接据此启动PPO。
17. **四维优于六维？** 对任务结构更直接、避免任意Fx/Fy请求；未训练、未做等权限广泛扫描，不宣称普遍优越。
18. **哪些历史有用？** 过去已下发命令、速度增量、SO3误差及因果动量估计可判断聚合Mz偏置；Yaw激励不足以稳定辨识所有6个gain。
19. **未来哪些工况适合？** 若重新立项，已安全的±.25/±.5%开发、最终≤.4%组合可作候选；但需先定义相对强C4确有价值的新目标，不能直接当获准训练集。
20. **可进入Phase2C吗？** 当前BLOCKED，主要因学习价值证据不足；不自动进入Gym/PPO。

## 实际视频与窗口验收

`assets/videos/stewart_phase2b3_yaw_compensation.mp4`：18秒1920×1080/25fps，450帧全部解码、450不同帧；3段C3/C4-PD/C4-Observer均来自reverse_z**现场live RGB**。每帧qpos哈希、时间、步数与CSV匹配，无未来曲线、无重新积分。另验证实际记录帧的支链位移非零。标题ACTUATOR MISMATCH AND YAW COMPENSATION STUDY，显示gain类型（仅标签）、Yaw/目标、Z/R/P、六腿行程/实际力、载荷、共享预算、保护/运行状态。没有PPO标签，也没有把毫米级动作夸大。

观看器是历史归档回放，明确ARCHIVE REPLAY；不把回放当新动力学。真实原生显示使用独立状态镜像，由实际owner发布，无GUI线程写live状态。GUI采集开/关完整积分相同，已真实Windows运行。

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/watch_phase2b3.py --trial results/runs/phase2b3_20261029_v1/frozen_evaluation/reverse_z/C4-Observer --validate-seconds 2
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2b3_observer.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/analyze_phase2b3_yaw_failure.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/compare_phase2b3_controllers.py --run-root results/runs/phase2b3_20261029_v1 --continuation-root results/runs/phase2b3_20261029_continuation_v2
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests -p phase2a_pytest_temp_plugin
```

上述入口及参数均已实际运行（测试运行带XML输出）。持续查看回放可去掉--validate-seconds 2，关闭窗口结束；本轮只对2秒自动关闭做了有界GUI验收，不虚称进行了无界时长验收。无需为看视频重跑实验。

重现实验的主入口是run_phase2b3_yaw_compensation.py（已运行），必须指定全新run-root，已有目录拒绝覆盖。首次原运行、修复后的续跑入口及全部日志保留。代码已修复参考适配，未来全新运行可直接完成冻结矩阵；本轮未为了证明命令再重复58回合。单独validate_phase2b3_yaw_actions.py的零动作函数已通过主流程执行，独立CLI未另跑，不列作额外实际运行成本。

## 未执行、限制与暂停

未执行：±1%失败重跑；任意四维/六维动作组合安全穷举；C4新噪声/延迟或更大失配/组合质量测试；实机、硬实时调度/传感器标定；全机构碰撞/形式化安全认证；Gym/PPO/SAC/BC。没有这些结果，更无训练checkpoint。原观察器是Home局部模型，不能外推大姿态。

本阶段交付源码、原始物理数据、图、18秒视频、全部失败/退化和报告。按用户要求暂停，等待审核。
