# Stewart Phase 2C-0：未知执行器动态、突变负载与任务可行性

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 可关闭的合成一阶模型

模型仅表示SYNTHETIC FIRST-ORDER FORCE RESPONSE：τ_i dF_drive_i/dt+F_drive_i=F_command_i。τ为研究候选，不是液压阀、电液伺服、摩擦或实机辨识模型。原MJCF、质量/惯量、原生执行器及C3/C4均未改。

dt=.001s，α_i=−expm1(−dt/τ_i)，F_drive_next=clip(F_drive_previous+α(F_command−F_drive_previous),±400N)。τ=0和disabled原样直通。正负阶跃、保持、卸载共4000个数组采样与解析解核对；这4000不是MuJoCo物理步。

## 采样顺序和三个信号

1. 在物理步开始，原控制器读取允许测量与原当前参考，C0/C3/C4生成经共享预算的F_command。
2. 原观测器记忆的是这个已下发F_command，而不是隐藏F_drive/F_actual。
3. plant模块用当前已知命令和内部旧状态更新一次F_drive_next；无未来物理状态。
4. 前六native ctrl写F_drive_next，后六ctrl仍只由原外扰profile写入；actual force=native clip(actual gain×drive)。
5. 原mj_step恰好一次、原mj_forward、安全检查、原.2s受限保持及记录；CSV状态时间为积分后t(k+1)，输入/控制步为k。

ZOH对执行器离散状态在固定命令下解析精确；将更新端点驱动力在一个1ms数值积分步内保持，是明确的离散plant接口约定，不宣称连续力波形和整个机器人积分也被解析精确求解。没有额外物理步，也未把一阶响应重复应用两次。1kHz指仿真离散控制周期，不是Windows硬实时保证；视频按物理时间25fps播放，不冒充原墙钟采集速度。

## 如何复用原执行循环

execution_hook.py从**只读冻结**DisturbanceOwner.step构造同一执行函数，只在原self.last_force赋值之前插入force=self.apply_actuator_command(force)。不全局替换mujoco函数，不用mjcb_control（避免mj_forward重复触发状态更新），不改历史源文件。AST删除这唯一插入后与原函数完全相同，单测验证且保留原唯一mj_step。这样保护保持的原力命令也经过同一动态模块；Hold时不能重置隐藏驱动状态或跳过滞后掩盖惯性。

## 初态与严格关闭等价

隐藏驱动力初值使用冻结名义Home参考平衡支链力ff，六缸约80～90N，保存actuator_dynamics.json初始化依据。所有方法初始qpos/qvel完全相同，没有从不合理的零驱动力起步，也未写live qpos制造稳定。disabled与enabled+τ0均不增加隐含一步延迟。

| method | enabled | tau_s | bitwise_equal | maximum_difference | signals_and_observer_equal | protection_termination_equal |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | False | 0 | True | 0.000000 | True | True |
| C3 | False | 0 | True | 0.000000 | True | True |
| C4 | False | 0 | True | 0.000000 | True | True |
| C4 | True | 0 | True | 0.000000 | True | True |

原C0/C3完整6000步对比Phase2B2 force10；C4完整6000步对比Phase2B3 reverse_z（含相同旧gain、外载和参考）。全部458维integration、物理信号、原观测器diag及保护/终止逐位一致。仅等价对照保留旧静态gain；新滞后研究全部nominal gain，不混入gain/质量/噪声/延迟。

## 信息边界

| 信号 | C4可读 | 独立安全/离线可读 | 说明 |
| --- | --- | --- | --- |
| 平台p/R/v测量 | 是 | 是 | nominal为真值，噪声/延迟工况只扰此通道 |
| 全链q/dq | 是 | 是 | 保持原理想反馈，非完整硬件传感器模型 |
| 原参考、名义模型、当前/过去下发请求 | 是 | 是 | 原C3/C4不改 |
| C3等效扰动与历史 | 是 | 是 | 不是实际外力传感器 |
| actual τ、hidden drive、actual force | 否 | 是 | 未定义实际力传感器，不得输入Actor |
| 外载真值、实际隐藏plant参数 | 否 | 是 | 仅单独特权诊断可用 |
| 未来状态 | 否 | 否 | 没有未来状态提供给控制器 |

每步记录observer_memory_wrench和known_command_wrench，独立复算一致；非零τ下它们与nominal_drive_wrench确实不同，证明未把内部输出力偷换成已知命令。原C3仍面对未知执行器模型误差，dhat同时含外载、动态误差与局部近似误差。

## 日志与共享预算

F_command为最终有界请求；F_drive_previous/drive为内部驱动；force为MuJoCo真实力；f_cmd历史别名保持“命令”语义，ctrl前六在新变体中是drive。F_add_total=F_command−F_C0仍合计C3、C4及候选补偿，±5N、20N/s不变；实际drive/force各±400N。不能把原命令总变化率误认为附加限速。

458维NPZ是完整MuJoCo integration状态，不包含Python额外状态。隐藏执行器drive/prev逐步保存在CSV，τ/初值在JSON；若未来要中途恢复执行，还需恢复原控制器内部状态或从前缀重建，不能仅凭NPZ宣称完整Python系统断点续跑。本轮观看使用已记录RGB/状态，不重新积分。

lagged_residual_wrench/lagged_additional_wrench通过独立平行影子滤波（同初值/τ，输入为实际访问状态上的C3/C0命令历史）提供离线实际增量。它不是另一条完整闭环反事实轨迹，也不输入控制器。旧W_physical_*保留原“瞬时命令反事实”定义，有lag时不能当作纯新增补偿作用；新分析用明确的三信号和lagged_*字段。

所有原行程、weld1e-4m/rad、条件数50、Yaw±.2°、Roll/Pitch±.5°、相对Home XY±2mm/Z±5mm、部分代理5mm及.2s保持不变。代理仍PARTIAL PROXY QUERY，不是整机或连续安全认证。
