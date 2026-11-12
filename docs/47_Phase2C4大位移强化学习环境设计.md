# Phase2C-4：50mm残差PPO环境设计

日期2026-11-05。工程`__PROJECT_ROOT__`，独立实验`__PROJECT_ROOT__\results\runs\phase2c4_20261105_v1`。用户本轮明确授权带着已知XY耦合代价研究有限Z残差，未削弱FF-VA、未扩成六维、未修改旧权重。

## 真实动力学与动作

复用StewartResidualEnv的Gym/CSV/关闭契约、LargeOwner、完整链式参考、原FF-VA包装、SharedBudgetGovernor、ActuatorLag、外载与安全流程。物理1ms、原C4/FF控制1kHz；动作20Hz，每决策50个真实mj_step。L50-T6完整Home→上升6秒→顶部保持→下降6秒→Home，总15秒，完整300决策。不是6秒episode。

动作1维Box[-1,1]，delta_Fz=0.1*a N，原完整J解`J.T*Fleg=[0,0,deltaFz,0,0,0]`。利用已冻结FF包装的probe端口在唯一共享governor前合成，不额外增加预算；C4最终命令记忆包含FF和PPO。`ff_probe_effect_N`仅是同一状态下无RL反事实命令差，不等于独立闭环或真实力传感。Fcommand、Fdrive及原生force分别保存，Actor只看已知命令，不看后两者。

固定工况：tau5ms、世界Z−5.5N、加载5.2s、卸载11s，原Home及姿态。原六执行器与后六外扰通道严格隔离，所有运动通过ctrl和mj_step。

## 零动作与Reset

完整15000步逐位比较qpos/qvel/act/mocap/ctrl/其他必要积分状态、实际力、drive、C4 observer、共享governor、参考、载荷及失败状态；完整积分档案逐位相同。第二episode前50步也逐位复现第一episode，Reset重新构建物理和控制状态、局部随机数、观测历史、上一动作与外载时钟。

第一轮实现存在一次回合末观测指针错误：原runner在完成后清空active，新观测错误访问它。保留失败CSV/控制台，改为独立已知参考指针后完整重验；没有修改物理控制器，也没有在错误环境上开展PPO。见`validation/failed_terminal_observation_attempt1`。

## 53维因果字段

| 字段 | 维度 |
| --- | --- |
| position_error | 3 |
| rotation_error | 3 |
| actual_twist | 6 |
| reference_twist | 6 |
| reference_z_acceleration | 1 |
| relative_slides | 6 |
| slide_velocity | 6 |
| known_command | 6 |
| causal_observer | 6 |
| reference_ff_z | 1 |
| accepted_total_add | 6 |
| last_accepted_residual | 1 |
| actual_z | 1 |
| target_z | 1 |


单帧53维；3帧159维只保留接口、未混入首轮训练。实际平台/支链反馈按理想可测状态处理，与实机传感器不等价。没有tau、外载真值/时序开关、隐藏drive、实际执行器力或未来物理状态。已知当前参考解析导数合法；observer是因果历史摘要，因此“单帧Actor”不代表系统没有任何历史信息。

尺度：位移Z/目标0.05m，slide0.06m，线速度0.03m/s，角速度0.05rad/s，参考加速度0.03m/s²，Z误差1e−5m、XY误差5e−5m，角误差0.000349rad，已知命令减Home后/5N，observer力/20N、力矩/0.8Nm，参考FFz/20N，接受总附加/5N，上次接受Z残差/0.1N。固定缩放、±10裁剪，无VecNormalize，评价不更新统计。随机预检观测裁剪率为0，未沿用旧1mm行程尺度。

## 奖励

冻结候选配置：`{"z_scale_m": 1e-05, "xy_scale_m": 5e-05, "angle_scale_rad": 0.00034906585039886593, "z_weight": 1.0, "xy_weight": 0.1, "rotation_weight": 0.05, "action_weight": 0.01, "slew_weight": 0.005, "cost_cap": 100.0, "failure_extra": 100.0, "gamma": 0.99}`。每1ms计算Z、XY、SO(3)、请求动作和总附加命令变化率代价，物理积分乘0.001/0.05。cap100限制单决策最大物理代价；各分量按相同比例裁剪，原始实际力变化另外评价。不得把Box动作裁剪率与PPO比率clip_fraction混为一谈。

对固定正/负/随机/交替动作真实完整轨迹比较奖励，并用相同其他状态的误差干预验证Z改善增益、横向恶化惩罚。失败惩罚超过剩余最大折扣代价加原保护200ms上界，避免提前失败获利。独立向量方程复算物理分量与每决策积分，42条训练episode档案通过；物理reward与SB3因TimeLimit加入bootstrap后的effective_rewards分别存在rollout NPZ，不能混为一谈。

正常完整15秒truncated=True并独立complete/success；安全失败terminated，执行原真实保护Hold并计入物理步；训练预算结束中途episode标incomplete，不能算失败或完成。正常尾部TimeLimit的价值bootstrap沿用原SB3语义。

保护仍为原大工作区、每链相对C0总附加±5N/20N/s、原生±400N、weld及Jacobian和部分几何代理检查。887个网格凸包配对、103个连接组件排除和30个无原生硬止挡被动铰链的局限保留；无接触不等于全几何安全或实机认证。
