# Phase1动力学与Jacobian力映射验证

## 虚功与实际广义力

J_l为6×6，平移列是dl/dp，转动列是dl/dtheta；输入世界角速度，不是欧拉角速度。由P=F^T qdot_slide=(J_l^T F)^T twist得到映射，F以N计，wrench前三维N、后三维N·m。

在4个真实闭合构型（Home、Z/Roll、Roll/Pitch、周期轨迹）各3组seed42有界力/twist上进行12次核对。用完整链IK有限差分，再用MuJoCo mj_differentiatePos构造42×6约束切空间T；独立核对T^T qfrc_actuator与J_l^T F。

- 虚功最大误差：7.10542736e-15W。
- MuJoCo广义功率最大误差：8.36872793e-10W。
- wrench最大分量差：2.00578292e-08（力/力矩各自单位）。
- 行程Jacobian最大差：1.91160754e-10。

映射是执行器贡献，不把重力、惯性、弹簧和约束反力都归给6个F。静态平衡通过完整名义模型求出；动态控制没有假装使用精确逆动力学加速度补偿。

## 轨迹与名义动力学

7种参考：Home、Z+4mm、Roll/Pitch各+0.4°、Z+2.5mm/Roll+0.25°、Roll/Pitch各+0.25°、带平滑包络的2秒周期Roll/Pitch（最大0.2°/0.15°）。位置初态与Phase0相同。各6秒，非周期动作1秒保持、2秒五次缩放、后续保持；周期动作在1～5秒，最后1秒归零保持。

这是六维状态/力控制，但当前主动激励只有Z/Roll/Pitch；X/Y/Yaw目标恒定，不声称已验证全六轴独立工作空间。全部组合均在网格节点与离节点做独立IK、行程和条件数检查，而非从单轴端点推断。

下表为7个名义任务、每任务完整物理步的pooled RMSE，未混入扰动或0.5ms重复复核；“平均每任务RMSE”是另一统计量，另存在CSV，不能混淆。

| controller | completed | trials | pooled_position_rmse_mm | pooled_orientation_rmse_deg |
|---|---|---|---|---|
| C0 | 7 | 7 | 0.070064967 | 0.0035956991 |
| C1 | 7 | 7 | 0.17667238 | 0.0043828386 |
| C2 | 7 | 7 | 0.070056885 | 0.0035953074 |

## 0.5ms复核（控制周期保持1ms）

| controller | trajectory | position_rmse_1ms_mm | position_rmse_halfms_mm | orientation_rmse_1ms_deg | orientation_rmse_halfms_deg | matched_time_max_position_difference_mm | matched_time_max_slide_difference_mm |
|---|---|---|---|---|---|---|---|
| C0 | z_roll | 0.034239289 | 0.034251177 | 0.0020219695 | 0.0020197434 | 0.000489718 | 0.00020364601 |
| C1 | z_roll | 0.082886889 | 0.083361976 | 0.0021038675 | 0.0021081054 | 0.0019485931 | 0.00049888931 |
| C2 | z_roll | 0.033937451 | 0.033975994 | 0.0020240448 | 0.0020232901 | 0.00029895711 | 0.00025059216 |
| C0 | periodic | 0.16026282 | 0.16069564 | 0.0075610638 | 0.0075653954 | 0.0010014581 | 0.00031184344 |
| C1 | periodic | 0.40133419 | 0.40317096 | 0.0099031543 | 0.0099217781 | 0.006022687 | 0.001550765 |
| C2 | periodic | 0.1603161 | 0.16070886 | 0.0075617871 | 0.0075655882 | 0.00086302592 | 0.00045529667 |

原配置solref=[0.0001,1]及solimp不变，refsafe开启。1ms物理步下有效时间常数至少2ms，0.5ms下至少1ms；约束有效刚度随此规则变化。因此不能把这个比较宣称为固定连续约束参数的纯积分收敛证明。plant.json逐次记录原始solref/solimp、refsafe与物理步长。

控制器始终1kHz；0.5ms时力保持两个物理步，PID积分仍按1ms更新，Jacobian刷新仍100Hz。误差以位置向量和SO(3)角度分别计算；平移/方向weld残差分别记录，mixed_equality_residual不统一解释为米。

所有指标从physics.csv.gz的XYZ和四元数独立复算，与运行器统计的差见phase1_metric_recomputation.csv。失败回合不会从表中删除；保护早停必须结合进度与终止原因解释RMSE。

## 外扰、计算耗时和复现限制

5种扰动：upper质量/惯量均×1.10（20→22kg）；upper质心局部X+3mm；前6力增益均×0.9；增益分别[0.9,1.05,0.95,1,0.92,1.02]；2～4秒施加upper_center局部[0,0,-10N,0.5N·m,0,0]。这不是执行器时间常数或迟滞模型，不声称研究了所有动态响应不一致。

外载site坐标系已用mj_jacSite和qfrc_actuator核对，局部→世界变换误差为0，见phase1_external_frame_validation.json。后6通道只在明确外载组启用。

controller_compute_s含参考求值与控制/条件监控，不含记录、渲染和独立碰撞查询；另存每步物理耗时和总墙钟。Python循环通常慢于物理时钟，尾部计算时间也会超过1ms，不宣称硬实时控制性能。本轮没有独立隔离CPU基准，耗时是实际执行现场的观测。
