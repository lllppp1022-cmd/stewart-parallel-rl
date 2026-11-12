# Stewart Phase 2B-3：执行器增益失配、Yaw补偿与任务筛选

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 四维接口及零动作

输入顺序[Fz(N),Mx(Nm),My(Nm),Mz(Nm)]，世界坐标、upper_center力矩参考点；W=[0,0,Fz,Mx,My,Mz]，解原J_l.T F=W。不新建另一套控制Jacobian。`task_residual_4d.py`把保持的任务请求在每1ms用原当前J映射，再交给**冻结的**ResidualController和SharedBudgetGovernor。内层PD/Observer为1kHz，因果残差20Hz；适配器只负责转换，不复制C0/C3。

四维全零分别在原force10任务完整6000步对比旧C0/C3：完整458维mjSTATE_INTEGRATION、qpos/qvel、actuator状态、12ctrl、实际力、原C3全部diag观测状态逐位一致。保护事件、终止原因相同，不通过放宽阈值获得等价。

| base | complete_state_bitwise_equal | maximum_difference | signals_and_observer_equal | protection_and_termination_equal |
| --- | --- | --- | --- | --- |
| C0 | True | 0.000000 | True | True |
| C3 | True | 0.000000 | True | True |

## 共享预算与已知控制命令

F_add_total=F_cmd−F_C0，包含原C3全部补偿和新增Yaw/四维请求。每腿总附加±5N、每1ms变化≤.02N即20N/s；还与原±400N命令余量取交集。不是给C4再加一套±5N预算。原生实际力限制±400N不变。预算日志只在budget_applicable有效段解释；保护保持有独立语义。

ResidualController在最终命令与原C3输出不同时，用原remember_applied_command更新下一步**已下发合成命令**记忆；不额外调用observer.update，不改原C3 previous_compensation更新律。它知道下发命令，不知道actual gain。误差仍通过反馈被因果估计。零动作不增加观测更新或随机数，不额外积分。

F_residual_requested、F_residual_applied、F_add_total分别为请求、相对原C3的施加量、相对C0的总附加量。W_requested为名义目标，W_realized为名义Jacobian下的投影结果；W_physical_residual是**仅供离线**的实际增益/真实力映射反事实增量，不冒充请求精确实现。Yaw预算占用会与C3的Z/R/P补偿竞争。

## 小幅Yaw预检

先验证历史成功的±.5%无外载Home；随后名义Home按±.005、±.025、±.1、±.25Nm逐级预检，各施加.8～1.8s，提前声明的时钟不读取真实外载。8个回合全完成，正负Yaw实际响应符号正确。每次仍6s完整运行，记录撤销后的恢复及非目标轴变化。

| variant | success | yaw_peak_deg | yaw_rate_peak_rad_s | max_xy_error_mm | roll_error_peak_deg | pitch_error_peak_deg | shared_add_peak_N | shared_add_rate_N_s | max_weld_translation_m | max_weld_angle_rad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mz_-0.005 | True | 0.002573 | 0.000129 | 0.000347 | 0.000009 | 0.000012 | 0.011371 | 11.371176 | 0.000000 | 0.000056 |
| Mz_-0.025 | True | 0.012818 | 0.000644 | 0.001360 | 0.000022 | 0.000057 | 0.046344 | 20.000000 | 0.000000 | 0.000056 |
| Mz_-0.1 | True | 0.051235 | 0.002574 | 0.005165 | 0.000072 | 0.000226 | 0.185419 | 20.000000 | 0.000000 | 0.000056 |
| Mz_-0.25 | True | 0.128048 | 0.006431 | 0.012772 | 0.000172 | 0.000564 | 0.463568 | 20.000000 | 0.000000 | 0.000056 |
| Mz_0.005 | True | 0.002550 | 0.000129 | 0.000249 | 0.000007 | 0.000010 | 0.011371 | 11.371176 | 0.000000 | 0.000056 |
| Mz_0.025 | True | 0.012795 | 0.000644 | 0.001177 | 0.000014 | 0.000055 | 0.046374 | 20.000000 | 0.000000 | 0.000056 |
| Mz_0.1 | True | 0.051213 | 0.002574 | 0.004984 | 0.000062 | 0.000224 | 0.185480 | 20.000000 | 0.000000 | 0.000056 |
| Mz_0.25 | True | 0.128026 | 0.006431 | 0.012609 | 0.000163 | 0.000563 | 0.463805 | 20.000000 | 0.000000 | 0.000056 |

这是单独Yaw方向测试，不是整个4维动作箱的安全证明。控制器Mz请求上限.30Nm，在有限闭环开发/评价中验证；不能把它推广到任意组合持续输入。无失配同时任意Fz/Mx/My/Mz的随机组合未扫描。

六维接口仅使用Phase2B2既有真实单腿±.1N与Leg1+.5N证据作为物理耦合参考，不重复自由6维大扫描。Leg1+.5N额外Y约.482mm，任务Yaw映射则把请求直接限定到Mz；两者量纲/权限不同，旧单腿底座为C0、本轮Yaw探针底座为C3，也不是同基线的受控性能对照，不能仅凭大小宣称四维PPO优于六维PPO。四维减少无意义瞬时Fx/Fy请求，但真实闭环仍有XY耦合，且缺少直接Fx/Fy权限。

## 冻结安全与初态

原1ms积分/1kHz传统控制/100Hz控制Jacobian不变；条件数≤50；weld平移/转动1e-4m/rad；Yaw±.2°、Roll/Pitch±.5°、相对Home XY±2mm/Z±5mm；代理间隙5mm、20ms查询；原行程与.2s保持逻辑保留。碰撞仍为PARTIAL PROXY QUERY。所有真实运动通过原data.ctrl和mj_step完成，没有动态写live qpos。

增益只在每次创建actual模型副本、首次积分前配置，nominal控制模型保持原样；无质量、COM、噪声或延迟混入。本轮固定未知gain，运行中只切换外载，不突然切换gain。原前三维/ZRP动作代码及MJCF均未修改。

## 参考适配器的真实问题

新参考为已允许periodic的0.85时间缩放，q/p/R取原参考.85t，dq与twist乘.85，姿态空间和行程未扩大。首个最终动态工况在首次积分前因适配器没有duration属性被调度器拒绝；0步异常和空日志保留。修复duration=原duration/.85、添加回归测试后，仅在独立continuation_v2目录执行缺失工况。最终6秒预算不变，未改控制参数或障碍。不能把首次0步异常报成动力学成功。
