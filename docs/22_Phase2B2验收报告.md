# Phase 2B-2：受限残差动作可控性与鲁棒性

日期：2026-10-29。项目：`__PROJECT_ROOT__`。主实验：`results/runs/phase2b2_20261029_v1`。本阶段没有训练网络，没有封装Gymnasium。

## 验收结论

**PHASE 2C CONDITIONALLY BLOCKED**。动作接口有效、零残差逐位等价、预算/原生力及观察器闭环通过。有限因果残差仅有微小且不跨场景一致的改善；尚无模型失配下超过强C3的可靠额外收益，也未形成可安全随机探索且收益明确的任务。本结论是研究门禁，不是把工程可运行等同PPO有效。本轮完成后暂停。

## 实际执行清单

| stage | trials | completed_success |
| --- | --- | --- |
| development | 12 | 12 |
| frozen_evaluation | 18 | 18 |
| oracle_diagnostic | 4 | 4 |
| preflight | 26 | 26 |
| robustness | 26 | 25 |
| test_safety_precheck | 2 | 2 |
| zero_equivalence | 2 | 2 |

90回合中89完整完成、1真实保护失败，共534584次1ms物理积分（534.584仿真秒）。这个汇总含探针、预检、开发、正式、oracle、鲁棒性，不作为统一任务成功率。64动作研究+26鲁棒性，恰好达到预先冻结90回合上限；没有无限搜索或额外更大失配实验。

另有观察器非侵入性2048步off/on各一次；首次受沙箱WinError5管道限制，已完成2048步off后退出，目录和错误记录保留；有界授权重试真实Qt+原生MuJoCo及RGB通过。因此观察器验证实际额外6144积分，不把它混作新科学回合或正式成功样本。自动测试内短动力学单独归测试成本。

## 独立复算与证据

`phase2b2_validation.json`逐一检查90回合：完整458维状态有限、CSV qpos/qvel/time与NPZ逐位一致、全频实际共享附加幅值/变化率、±400N原生命令与实际力、参数组隔离、延迟因果性、窗口误差；独立scratch前状态重算切空间广义力映射最大误差5.55e-17。完整6s零动作和保护事件与原C0/C3一致。历史2902个文件SHA256未变。

全部原159测试和新增测试真实运行：**189/189通过**。首轮183通过，最终新增6条验收证据测试后再跑。XML与日志在results，原两条pandas float_precision弃用警告保留；未修改旧测试来通过。独立复核脚本首次因读取actions_frozen中selected层级错误KeyError退出（40条后），修复验证脚本后v2全部90通过；不是动力学失败，旧日志保留。

录像18秒、1920×1080、25fps、450帧全部解码且450帧不同；来源是3条正式回合现场actual_rgb，而不是重积分/运动学播放。每帧qpos哈希、物理步/时间/载荷与CSV相符，曲线只显示过去。另检查记录帧支链实际位移非零。真实运动很小，保持物理尺寸，没有视觉夸大。视频比较C3/C3_task/C3_leg，无PPO标签或虚构RL训练。

## 二十项问题逐项回答

1. **零残差复现？** 是，C0/C3各6000步完整integration、12ctrl、力、位姿、执行器状态逐位一致，保护事件和终止原因相同。
2. **三维映射？** 世界upper_center处J_l.T解算，通过单位/方向单测、实际映射及真实力复核；受预算影响不保证W_requested=W_realized。
3. **六维真实响应？** 六腿各±.1N逐通道测试有不同有符号位移，真实mj_step积分，非qpos动画。
4. **共享预算？** 是，C3原补偿和新增请求合计每腿±5N、20N/s，不叠加成±10N。
5. **有益方向？** 因果lead的目标Z修正使8N加载/卸载峰值微降；倾转开发也有改善，不能据此断言全局方向都好。
6. **恶化耦合？** 单腿力明显引入XY/Yaw；Leg1+.5N约.482mm额外Y偏移，任务力矩也有横向耦合。
7. **20/50Hz？** 不普遍单调；开发lead20略优，C0 damping50可能增加总力变化率，1kHz基础控制不变。
8. **卸载超调？** 8N Z约降1.30%；周期Roll仅微降且整体IAE恶化，不宣称全面解决。
9. **不增加力变化率改善？** 8N总力变化率基本持平有微小峰值改善，但平方积分略增；周期共享补偿变化率增加，不能宣称无代价Pareto优势。
10. **特权空间？** oracle在受限预算下有更小部分误差，有限诊断有物理空间，但非理论最优证明。
11. **是否需要真值？** 正式lead/damping仅用当前/历史观测，无载荷时刻；oracle使用真值单列。尚未证明可因果学到oracle收益。
12. **质量/质心敏感？** 小幅1/2%与.1/.25mm均完成，质量产生重力估计偏置；不能外推大失配。
13. **gain敏感？** 是，统一.98轻退化，交替±1%在外载前Yaw保护失败。
14. **噪声/延迟敏感？** 部分平台小噪声影响很小，5ms延迟峰值升约8.57%；全链仍理想，非硬件鲁棒性证明。
15. **未来任务？** 优先安全小幅统一增益/质量失配与有限延迟，需先确认残余收益；不直接对Yaw失败任务随机探索。
16. **三/六维？** 三维更贴当前任务且少耦合。匹配反馈六维几乎相同，不是任意六维策略强弱结论。
17. **动作范围依据？** 已实测任务±.5N/±.025Nm、各腿±.1N及Leg1正.25/.5N；不是任意同时组合或±5N随机安全证明。
18. **新违规？** 1例交替gain±1%姿态保护，完整保存；其余89回合完成。本轮没有降低阈值。
19. **值得学习的问题？** 有失配和延迟敏感性，但未证明新增残差在这些条件下的可靠优势；强C3使名义可改进空间很小。
20. **进入Gym/PPO？** 有条件阻塞，须先确认可控、可观测、收益足够且安全的失配目标及代价门禁。本轮不创建Gym或训练。

## 实际测试过的PowerShell入口

下面在项目根运行。回放是历史真实动力学记录，窗口会标识ARCHIVE REPLAY，不运行新实验。播放按钮控制归档时间，原控制面板交互不能改变历史记录。原生动态窗口实际验收由observer脚本打开，固定2048步并自动正常关闭；这是有界验证，不替代原无限交互窗口。

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/watch_phase2b2.py --trial results/runs/phase2b2_20261029_v1/frozen_evaluation/test_force8_time/C3_task --validate-seconds 2
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2b2_observer.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/compare_phase2b2_residuals.py --run-root results/runs/phase2b2_20261029_v1
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2b2.py --run-root results/runs/phase2b2_20261029_v1
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests -p phase2a_pytest_temp_plugin
```

持续观看时去掉`--validate-seconds 2`，保留原回放窗口直到用户关闭；去掉参数的完整命令本轮未另做无限时长运行。主实验两脚本已经真实执行，不建议为看视频重跑90回合。若复现实验必须用全新run-root，原目录存在会拒绝覆盖。MP4直接在播放器打开，无需安装RL。

## 交付与未执行事项

设计/动作/鲁棒性详见docs19～21；核心表为phase2b2_c3_comparison.csv、action_sensitivity.csv、probe_response.csv、robustness.csv、oracle_diagnostics.csv，完整90行all_trials.csv。raw回合均含config、plant、action_profile、summary、原始状态和独立复算，另有decision_effectiveness逐决策投影/无效/施加延迟统计。

未执行：Gymnasium、PPO/SAC/BC；失配下残差正式干预；任意同时六腿动作箱安全验证；更大或组合不确定性；真实硬件与完整传感器实验；形式化/完整碰撞安全保证。本轮没有把这些列为已完成。无新的训练模型/checkpoint可交付，因为未训练网络。观察器首次权限错误和validator首轮错误均保留；最终有效验收已通过。

不自动开始Phase2C。建议用户先审阅周期场景退化、Yaw失配失败和控制代价，决定是否值得再定义更有解释力的小型学习任务。
