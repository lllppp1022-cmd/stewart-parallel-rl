# Phase1控制器设计

## 模型、变量和冻结边界

物理模型来源与Phase0相同，原XML未改。1ms物理步，1kHz控制。原±400N控制/力限幅、slide[-0.1,0.35]m及weld solref/solimp均保留。平台p/R、世界线速度v/角速度omega来自upper_center（不是偏移后的质心），Jacobian与wrench也以该点为参考。

q_slide/dq_slide是6个移动副相对行程/速度；actuatorpos与相对行程一致，不能当作安装点弦长或约1m的支链总长度。u是名义力控制输入，F_actual为考虑plant增益及原生限幅后的实际执行器力。后6个外扰通道与这6个控制输入分开。

src/audit_control.py完整冻结，5000步原Z参考C0复现得到力指令、qpos差异均为0（逐位相同）。Phase1比较将三者置于同一新参考上：每20ms完整链式IK/静平衡节点，三次Hermite给出连续行程与速度，1ms逐点检查行程，离网格直接IK核对最大误差见phase1_preflight/*_interpolation.json（约1e-11m）。这不是声称每1ms都在线求解一次IK，也不是未经检查的过冲样条。

## C0：冻结平衡力+支链PD

F_raw = F_nominal_static + 5000(q_ref-q_slide) + 200(dq_ref-dq_slide)。P为N/m，D为N·s/m。保留Phase0增益、符号和接口；新轨迹的静力参考在名义模型预计算，没有读取扰动物理参数。没有加入新的加速度前馈。

## C1：支链PID

在相同P/D与F_nominal_static上，I_state += error×0.001s；Ki=3000N/(m·s)，积分状态上限±0.005m·s，对应最多±15N积分补偿。输出仍±400N。使用实测slide速度反馈，避免对位置差分直接微分。

条件抗饱和：如果候选输出已超力限且积分会继续推向饱和，则冻结该分量积分；反向消饱和可继续。保护保持时清零积分。无效数值/越行程目标被拒绝。抗饱和和原生限幅有独立测试，但正式试验没有实际触及积分或力上限，不能把“有测试”说成这些极端工况已广泛验证。

PID不是先验更优。其名义周期任务有积分相位/累积误差，Ki在正式矩阵前冻结，没有根据结果追加预算或调参来制造优势。

## C2：Jacobian任务空间控制

采用世界twist=[vx,vy,vz,wx,wy,wz]，世界wrench=[Fx,Fy,Fz,Tx,Ty,Tz]，参考点upper_center。

W_act = J_l^T F_actual；控制时W_cmd = J_l^T F_nominal_static + K_task e_pose + D_task(twist_ref-twist_actual)。e_pose旋转部分为Log(R_ref R_actual^T)，不是欧拉角相减。求F_raw = solve(J_l^T,W_cmd)，随后遵守原±400N限幅。超限即不能精确实现所需wrench，不能把裁剪说成精确广义力实现。

为匹配C0而非人为削弱基线，K_task=J_home^T(5000I)J_home、D_task=J_home^T(200I)J_home，保留完整交叉项。不同块单位分别对应N/m、N/rad、N·m/m、N·m/rad，不能把整个矩阵叫作统一N/m。

J_l从冻结名义几何与实际可观测关节构型计算，100Hz刷新；P/D力更新1kHz。三个方法都运行同一条件数监控。条件数按0.35m尺度归一化，阈值50；接近奇异或非法解则拒绝并保护。六输入六维局部非冗余映射，没有使用盾构零空间力分配公式。

## 共享安全与模型隔离

每物理步检查有限状态、原生警告、slide范围、weld平移/角残差（都维持1e-4m/1e-4rad）、受限位姿；50Hz独立proxy查询只覆盖已定义的配对，5mm为这些代理的保守触发门槛。实际XY偏移2mm、Z偏移5mm、Roll/Pitch0.5°、Yaw0.2°是受限试验保护界限，不是完整可认证工作空间。

失败后停止参考推进，转为以违规时实际slide构型为目标的受限PD保持0.2秒，积分清零，并保留继续运动。日志desired reference仍显示原任务参考，以计算未完成误差；实际保持目标另从事件qpos与control_time精确重建在phase1_applied_reference_audit.csv，不把两者混用。保持是响应式仿真措施，不保证能瞬时停住或回到安全集。

所有补偿读取冻结nominal模型；独立plant副本只接受质量/惯量、质心和增益扰动。Controller.command不接收plant模型。真正外载仅由运行器写入后6通道，不能由C0/C1/C2偷偷托举平台。
