# Phase 2B-2：受限残差动作可控性与鲁棒性

日期：2026-10-29。项目：`__PROJECT_ROOT__`。主实验：`results/runs/phase2b2_20261029_v1`。本阶段没有训练网络，没有封装Gymnasium。

## 控制路径与冻结边界

新增 `residual_action.py` / `residual_governor.py` / `phase2b2_experiment.py` 包装原C0/C3；未复制或改写其控制律。历史2902文件逐一SHA256一致，详见冻结清单及validation。1kHz物理与原控制周期不变，C3原Jacobian100Hz刷新。C3参数：观测带宽30rad/s、补偿增益1；C1-L泄放1/s。原weld、姿态、行程、条件数50、部分碰撞代理5mm保护不变；部分代理不能构成完整碰撞保证。

动作只写前六个力执行器；后六路外载由实验profile独立写入。输入非法、维度错误、非有限、Jacobian病态时保守拒绝/保护，不求伪逆制造结果。实际状态始终通过原mj_step积分；仅显示镜像和独立运动学检查允许设置scratch状态。

## 三维与六维动作

三维为世界坐标、upper_center参考点的 `[δFz(N),δMx(Nm),δMy(Nm)]`，嵌入 `[0,0,δFz,δMx,δMy,0]`，解 `J_l.T δF=W`。请求与实现分开记录；不以理论逆映射代替真实作用验证。六维直接按Leg1..6 MJCF原序输入支链力N。

20Hz每50个物理步决策；50Hz每20步决策。任务动作保持任务广义力请求，1kHz通过当前可用J映射；支链动作直接保持六力。两种保持语义有差别。正式六维反馈由同一个三维反馈在决策时映射并逐腿裁到±0.1N；这不是任意六维最优策略。自由六维物理耦合由12个单腿正负探针独立验证。

## 共享预算

原输出记F_base（C0或完整C3），原名义输出F_C0。只存在一个预算：`F_add_total=F_cmd-F_C0`，包含C3自身补偿及新增残差。

每个物理步、每支链将候选附加力投影到以下三个区间交集：

1. [-5,+5]N。
2. 上一步**实际合成附加力**±0.02N（dt=.001s，即20N/s）。
3. [-400-F_C0,+400-F_C0]N原生命令余量。

区间为空拒绝。实际原生力±400N不变。约束的是附加力，不是F_C0全部力变化率；总力变化率必须另看控制代价。零请求且原输出可行时原样返回base.copy，保留冻结控制器的浮点数值，不加多余积分。数值容差1e-10N与原C3检查量级一致，未放宽实验阈值。

全程零动作已严格复现旧轨迹。曾施加非零后再归零，仍可能存在共享限速尾迹，不能声称瞬时消除残差或重置观测器。F_residual_applied定义为最终cmd减冻结base，不等于总附加补偿；在投影时可以与请求不同。

## C3已知输入闭环

原C3一次command只执行一次观测器更新。若最终合成命令改变，包装器通过原observer.remember_applied_command记录当前测量坐标/速度及J.T@最终cmd，供下一步使用；不再次更新观测器，不改C3自身previous_compensation状态。全零保持原记忆写入路径，因此逐位等价。

此处已知的是下发命令，不是真实未知gain下的实际力。实际gain和完整真实状态仅用于安全和离线评价。不能以传入真实gain“修复”模型失配。实际相对作用额外记录为W_physical_*：由预积分状态的独立约束切空间映射与实际执行器力，减去同状态下基线命令经actual gain/native clip的反事实力。它只用于日志，不是向策略提供的观测。

## 候选动作与信息

非特权动作只接收当前/历史p、R、v、当前参考与C3可观测等效扰动。damping：Z误差250N/m、速度50Ns/m；倾转误差/角速度均2（相应Nm单位）。lead：对相邻决策的等效扰动差分乘-0.02s。任务反馈裁剪±[0.5N,0.025Nm,0.025Nm]。无扰动固定探针按预先冻结时钟.8～1.8s施加，与外载时间无关；ramp为预定义低频波形。lead/damping均是经典因果诊断，不是神经网络或新RL算法。

特权oracle读取当前实验载荷真值，1kHz给理想抵消请求，仍受共享预算。标记PRIVILEGED ORACLE DIAGNOSTIC，只有该stage允许创建，不能进入正式排序，且有信息/频率优势，不代表可学上界或严格理论最优。

## 日志语义

每回合保留config/plant/action_profile/initial_state/home_baseline、physics.csv.gz、458维完整integration_states.npz、events/safety_events、summary、原metrics及独立metrics_recomputed。记录12ctrl、实际force、qpos/qvel、时间、参考、p/R、传感器capture/read、请求/施加/共享补偿及广义力。CSV row为积分后状态，input_step/测量时刻对应此前积分输入；不混用前后时刻。预算字段只在budget_applicable=True解释，保护保持阶段不将占位值解释为合法残差。C1/C1-L/C2为原始控制器对照，不伪造其F_C0预算字段；其原积分限制不变。

恢复标准为位置≤.25mm且SO3≤.02deg连续.3s，需先真实越界；未越界=N/A，未恢复和提前截断分别标注。6s任务完成标准仍是原最后.5s位置<.75mm、姿态<.1deg且无保护失败。

## 零残差与已知输入实测

| base | full_integration_state_bitwise_equal | all_physical_signals_equal | maximum_state_difference | same_terminal_reason |
| --- | --- | --- | --- | --- |
| C0 | True | True | 0.000000 | True |
| C3 | True | True | 0.000000 | True |

额外核对零动作保护事件相同。无外载时已知+0.5N任务Z输入，1.0～1.7s观测器Fz范围[-0.0001068923080253569, 0.00036984976669502317]N，均值-0.000061236N，没有误估成-0.5N外扰；修正实际命令记忆1004次。
