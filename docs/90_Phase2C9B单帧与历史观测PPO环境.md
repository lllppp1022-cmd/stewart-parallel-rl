# Phase 2C-9B 单帧与历史观测PPO环境

日期2026-11-12，Asia/Shanghai。独立归档 `__PROJECT_ROOT__\results\runs\phase2c9b_20261112_v1`。数值从实际CSV/JSON/NPZ重算。

复用StewartAdaptiveEnv和HistoryObservation，58维当前帧逐字段沿用原归一化，历史为[t−4,t−3,t−2,t−1,t]、290维，20 ms间隔、首尾80 ms。初始重复当前合法帧，reset清空缓冲。当前观测包含因果观测器内部反馈和前一动作，因此“单帧”仍具有传统控制器的内部记忆。历史策略增加的是Actor可见的4个过去帧。

零动作、非零动作、reset历史顺序及22 kg Plant和20 kg控制模型隔离均完成真实测试。恢复验证在显式预算边界保存优化器、任务RNG和全局随机状态，重复重载分支的动作、物理轨迹与最终参数逐位相等。测试用1536决策单列，不计正式训练。任意回合中途崩溃恢复尚不支持。

两种Actor/Critic隐藏层64×64，原始V2奖励，原512 rollout、64 batch、10 epoch、lr0.0003及历史gamma/GAE。实际参数量如下。输入维度不同造成参数量差异，不称完全同容量。

由初始checkpoint反读的有效设置还包括vf_coef=0.5、normalize_advantage=True、clip_range_vf=None；完整实际设置见 effective_ppo_settings.json。单帧16069个可训练参数，五帧45765个。

| method | seed | input_dim | parameters | initial_hash |
| --- | --- | --- | --- | --- |
| single_frame | 42 | 58 | 16069 | 1211db177cf88a4b32ff2ccb91559dc087de83e0ee4b1059b6d924cbfc3d09e9 |
| single_frame | 123 | 58 | 16069 | da4da2c398c40c2cfe57e32fd5acefd347e6205bc1302d17ee6c72d40d3b39b7 |
| single_frame | 2026 | 58 | 16069 | 33a8213d16e02a3426cc535d563d9d2b939f70377aae17420c69e513a472315c |
| history_5frame | 42 | 290 | 45765 | a4ef728656d3f2eed1637272e44f50120a01872a946f7226e473b8ab9527e1e4 |
| history_5frame | 123 | 290 | 45765 | c7e161b3c90a3ba099299f611b5fe97b7daad1f8e35e0a2fd5697468f79bbe6b |
| history_5frame | 2026 | 290 | 45765 | d42b1cfafafdb87d9dc8f47b160148cdd39dc6fedc4834628e1f8ca2a82033cc |

训练质量与任务来自独立PCG64采样器。所有种子两方法任务序列逐条核对，Box裁剪和真实随机动作由原回调验证。质量标签与实际驱动力隐状态只用于日志，不进入58维帧。随机策略训练和确定性评价分开。
