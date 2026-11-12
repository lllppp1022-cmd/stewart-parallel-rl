# Stewart 项目全阶段研究总结

以原始结果目录为索引。表中规模是对应证据表的行数，不跨实验合并成功率。

| 阶段 | 问题 | 技术 | 证据 | 结果规模 | 边界或承接 |
| --- | --- | --- | --- | --- | --- |
| 0 | 闭链与驱动可用性 | 原MJCF、实际关节链IK、步长对照 | results/phase0_timestep_sensitivity.csv | 14 | 2ms严门槛失败，采用1ms基线 |
| 1 | 平台轨迹与力控制 | 传统反馈、Jacobian力映射、代理几何查询 | results/phase1_controller_comparison.csv | 42 | 工作区与碰撞查询仅覆盖验证样本 |
| 1.5 | 实时交互 | 动力学线程、Viewer及数据归档 | results/phase1_5_run_summary.csv | 13 | 运动和录制同步 |
| 1.5.1 | 交互可读性与持续运行 | 彩色支链、相机、状态镜像和事件日志 | results/phase1_5_1_visual_validation.json | JSON | 区分实际积分与显示 |
| 2A | 突加载荷 | 传统控制对照和恢复窗口 | results/phase2a_disturbance_summary.csv | 51 | 以物理指标筛选抗扰方法 |
| 2B-1 | 改进经典抗扰 | 反馈和因果扰动估计 | results/phase2b1_controller_comparison.csv | 45 | 观测器与控制联合评估 |
| 2B-2 | 残差权限与鲁棒性 | 共享限幅、动作探针和失配试验 | results/phase2b2_action_sensitivity.csv | 38 | 有效残差依赖状态和预算 |
| 2B-3 | 增益失配与Yaw | 四维任务空间经典补偿 | results/phase2b3_controller_comparison.csv | 30 | 筛选可控目标 |
| 2C-0 | 未知执行器动态及突变负载 | 滞后、噪声和观测延迟；C4基线 | results/phase2c0_actuator_lag_validation.csv | 4000 | 失配能否形成学习任务需分别验证 |
| 2C-1 | 首次真实残差PPO | Fz单维修正、Gymnasium与价值审计 | results/phase2c1_training_summary.csv | 3 | 训练成功不等于超过经典对照 |
| 2C-2 | 有限动态任务与真实录像 | 小幅Z参考与PPO对照 | results/phase2c2_evaluation_comparison.csv | 10 | 小位移需用支链与位置曲线辨认 |
| 2C-3A | 50mm几何与动力学可行性 | 工作区扫描、IK、低速动力学验证 | results/phase2c3a_workspace_scan.csv | 101 | 仅采用准入轨迹 |
| 2C-3B | 50mm速度和加速度跟踪 | FF-VA和多速度误差对照 | results/phase2c3b_controller_comparison.csv | 49 | 使用增强传统基线 |
| 2C-4 | 50mm残差PPO | Z跟踪与外载学习 | results/phase2c4_controller_comparison.csv | 24 | 按事件分段评价 |
| 2C-5 | Roll/Pitch动态与力矩抗扰 | 二维力矩策略、姿态保护、Home强扰动边界 | results/phase2c5_torque_boundary.csv | 12 | 4.8 N·m触发保护 |
| 2C-6 | 姿态优先奖励及课程 | V2与中强外力矩 | results/phase2c6_training_summary.csv | 9 | 区分历史模型复用与新增训练 |
| 2C-7 | 综合Z与姿态运动 | 三阶段PPO和冻结评价 | results/phase2c7_training_summary.csv | 6 | 强经典反馈仍是主要对照 |
| 2C-7V | 回报和实际动作可解释性 | 检查点闭环、1ms行为分析、真实训练画面 | results/phase2c7v_checkpoint_evaluation.csv | 28 | 发现恢复种子覆盖及回报与IAE分歧 |
| 2C-8 | 匹配奖励消融与Critic诊断 | 独立任务RNG、V2/V3三种子 | results/phase2c8_training_summary.csv | 6 | 未晋级，转向可测模型失配 |

## 模型与运动证据

动平台名义质量20 kg，主惯量为约 [1.578161, 0.790792, 0.790786] kg·m²，原始模型数值见 results/phase0_model_inspection.json。工作区几何扫描覆盖轴向 −100 至 +100 mm、间隔2 mm，只能解释为扫描边界。动态准入另见 results/phase2c3a_workspace_limits.json 的 L30/L50。2C-8实际联合轨迹包括50 mm升降及Roll +0.25°、Pitch −0.25°。

历史量化锚点如下。

| 阶段 | 原始证据与观测 |
| --- | --- |
| 0 | 1 ms Home 5000步通过，weld角误差峰约5.56e−5 rad，见 phase0_timestep_sensitivity.csv |
| 1 | 1 ms名义C0完成7/7，均值位置RMSE 0.05204 mm，见 phase1_group_summary.csv |
| 1.5 | phase1_5_run_summary.csv保留owner_exception及其0步记录，交互验收另见 phase1_5_interaction_validation.json |
| 1.5.1 | 32000步交互视觉验收，积分状态与12个控制输入对照逐位一致，见 phase1_5_1_visual_validation.json |
| 2A | Home突加载荷的C0 pilot位置峰0.37425 mm，加载窗口未恢复，见 phase2a_disturbance_summary.csv |
| 2B-1 | 历史benchmark的C0加载位置IAE 0.76196 mm·s，见 phase2b1_controller_comparison.csv |
| 2B-2 | 90次研究试验、534584步，RL准入有条件阻断，见 phase2b2_decision.json |
| 2B-3 | 58次物理试验、348000步，通过局部Yaw动作权限检查，未证明C4之外的学习收益，见 phase2b3_decision.json |
| 2C-0 | 564000步，Home Z准入，原XY任务受阻；只验证聚合因果反馈，见 phase2c0_decision.json |
| 2C-3A | 30 mm、10 s上升、无外载的C0试验实际峰29.9999 mm，Z RMSE 0.06514 mm，见 phase2c3a_controller_comparison.csv |
| 2C-3B | 30 mm、10 s上升、无外载的开发FF-V试验Z RMSE 0.004175 mm，见 phase2c3b_controller_comparison.csv |
| 2C-7 | 三种子共98304决策和1966080物理步，预算前缀计入累计但不重复相加，见 phase2c7_training_summary.csv |
| 2C-8 | 六组49152决策、983040物理步，32个最终评价全完成，见 phase2c8_safety_audit.json |

平台任务空间为三平移三转动。原模型 nq=43、nv=42、nu=12；六支链含移动副和两端转动机构，六个 weld 实现闭链。仅前六个执行器驱动支链，剩余为显式外扰通道。真实关节链考虑铰轴非共点偏置，IK求滑块坐标，Jacobian经差分和力映射验证。几何代理未覆盖所有原网格与万向节。

Phase 0 实测小幅升降与倾转。3A–3B扩展至50mm，5验证主动Roll/Pitch，7与8联合执行升降及姿态。可达性看IK和支链行程，动态可控性看真实力驱动，安全只引用实际检查过的任务与保护边界。

## 抗扰强度边界

2C-5的Home正Roll、C0、2.0–3.5 s突加力矩预检中，0.15、0.5和1.2 N·m的SO(3)峰值分别为0.007316°、0.024404°、0.058600°，三次均完整。源表为 results/phase2c5_torque_preflight.csv。

早期0.15和0.5 N·m要随各阶段的姿态及外载波形阅读，不与后期强扰动直接汇总。0.5–1.2 N·m动态工况查阅 phase2c7_joint_preflight.csv；Phase 8的最终联合任务最大模长1.2 N·m。以下为 phase2c5_torque_boundary.csv 的定点原始结果。2.4 N·m只支持对应Home任务，4.8 N·m保护失败保留。

| 任务 | 轴 | 幅值 N·m | 步数 | SO3峰 度 | 保护原因 | 可恢复 |
| --- | --- | --- | --- | --- | --- | --- |
| Home | 0 | 1.2 | 6000 | 0.01954333814029536 |  | True |
| Home | 0 | 2.4 | 6000 | 0.0781629524310331 |  | True |
| Home | 0 | 4.8 | 2720 | 0.1965303981926596 | restricted_position_envelope | False |
| Home | 1 | 1.2 | 6000 | 0.019546761012768157 |  | True |
| Home | 1 | 2.4 | 6000 | 0.07804064590900615 |  | True |
| Home | 1 | 4.8 | 2721 | 0.19655781867350844 | restricted_position_envelope | False |
| Home | 2 | 1.2 | 6000 | 0.019278615821274898 |  | True |
| Home | 2 | 2.4 | 6000 | 0.07703596077855843 |  | True |
| Home | 2 | 4.8 | 2727 | 0.19521838528146696 | restricted_position_envelope | False |
| Home | 3 | 1.2 | 6000 | 0.01928154367920776 |  | True |
| Home | 3 | 2.4 | 6000 | 0.07703758645049226 |  | True |
| Home | 3 | 4.8 | 2727 | 0.1952158189987323 | restricted_position_envelope | False |
