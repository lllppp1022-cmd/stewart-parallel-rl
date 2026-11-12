# Stewart Phase 2B-3：执行器增益失配、Yaw补偿与任务筛选

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 冻结与数据来源

历史Phase0～2B-2共4814文件已冻结，原控制器、模型、实验数据及旧失败记录逐项SHA256核对一致。`results/phase2b3_frozen_history.json`为清单。首先只读分析Phase2B2的名义C3、交替±.5%及已失败±1%日志，没有重跑±1%。隐藏actual gain只用于本文件离线归因，控制器不可访问。

## 六腿增益如何形成Yaw力矩

世界upper_center处采用原支链Jacobian J_l。每腿名义gain为1，实际力为native clip(g_i F_cmd_i)。力偏差ΔF=F_actual−F_cmd，瞬时六维广义力偏差ΔW=A(q).T ΔF，其中A是独立约束切空间的dl/d世界平台速度映射，与原J_l在同一物理定义下核对。这里保留真实状态下的映射，不把笛卡尔弦长当作实际移动副长度。

六腿的Yaw力臂符号交替。原正支撑力约80～90N，交替±.5%使实际力差约±.4N；gain符号与力臂符号相乘后，六腿Yaw贡献朝同一方向叠加，而不是互相抵消。原CSV新增逐腿yaw_lever_arm_j与yaw_moment_contribution_j，可逐项求和得到下面的ΔMz。对其他轴也完整保留六维ΔW，不把所有运动都归给Yaw。

| case | preload_gain_Mz_mean_Nm | preload_observer_Mz_mean_Nm | local_unexplained_Mz_RMS_Nm | preload_yaw_rate_peak_rad_s | failed | first_failure_s |
| --- | --- | --- | --- | --- | --- | --- |
| nominal | 0.000000 | -0.000000 | 0.000001 | 0.000001 | False | N/A |
| alternating_0.005 | -0.234104 | -0.234345 | 0.000056 | 0.006028 | False | N/A |
| historical_failed_0.01 | -0.468238 | -0.471408 | 0.000131 | 0.012055 | True | 0.384000 |

名义无失配ΔMz=0，观测偏置约-4.76×10⁻⁷Nm。±.5%时真实增益力矩约-.234104Nm、原C3内部估计约-.234345Nm；±1%保护前约-.468238/-.471408Nm。加倍失配近似加倍Yaw力矩与早期角速度，而不是随机颜色或可视化变化。角速度方向为负，Yaw随之偏负，支撑了“力偏差→Yaw力矩→角速度→姿态→保护”的因果链。

原C3确实有完整六维估计器！它主动补偿仅选择[2,3,4]三个分量，未把Mz用于控制；不能说Yaw未进入观测器。C3的等效估计还包含建模残差，并非精确外力传感器。局部方程M0 vdot=Wcmd−W0−K0x−D0v+d_eq，离线用速度增量和名义矩阵复算。±.5%剩余Mz局部平衡RMS约5.55×10⁻⁵Nm，±1%保护前约1.31×10⁻⁴Nm；约束柔顺、变量质量/几何与数值微分仍存在，未宣称全部精确解释。此处没有把constraint反力读入控制器。

## 历史失败时间线

±1%：.200s约Yaw=-.085917°、世界ωz=-.011872rad/s；.384s达到Yaw=-.200308935°，触发原±.2°保护。发生在2s外载前，因此是支撑状态下固定增益失配问题。原0.2s受限保持到.584s，Yaw继续到约-.395282473°：Hold是受限关节保持而非瞬间消除增益偏差/动量。不能称“已恢复”。

保护后原控制器reset，日志中的观测器零值不表示真实扰动消失。所有偏置统计只用保护前有效段。load/unload未发生，指标右删失/N/A，不填0、不用短RMSE与6秒完成轨迹排序。原±1%失败文件未删、未移、未改，也未重新执行。

图：`phase2b3_gain_force_wrench.png`、`phase2b3_yaw_failure_chain.png`；数据`phase2b3_gain_yaw_mapping.csv`，离线脚本`analyze_phase2b3_yaw_failure.py`不调用mj_step。
