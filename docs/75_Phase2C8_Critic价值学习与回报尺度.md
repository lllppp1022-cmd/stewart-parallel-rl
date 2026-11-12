# Phase 2C-8 Critic价值学习与回报尺度

日期：2026-11-09（Asia/Shanghai）；独立实验目录：`__PROJECT_ROOT__\results\runs\phase2c8_20261109_v1`。所有数值由原始CSV/NPZ/JSON生成，不修改历史结果。

## 实际价值拟合

| method | seed | value_loss_max | EV_median | negative_EV_updates | mean_prediction_minus_target | last_prediction_minus_target |
| --- | --- | --- | --- | --- | --- | --- |
| v2_control | 42 | 10445.045670 | 0.109566 | 2 | 8.885998 | 12.961007 |
| v2_control | 123 | 10947.042456 | 0.007629 | 7 | 12.609104 | 52.901291 |
| v2_control | 2026 | 10231.904218 | 0.152944 | 4 | 8.419509 | 0.683506 |
| v3 | 42 | 4852.961433 | 0.120464 | 3 | 8.065690 | 12.018328 |
| v3 | 123 | 5115.571838 | 0.060373 | 6 | 10.618934 | 37.684784 |
| v3 | 2026 | 4762.890170 | 0.162134 | 4 | 7.715415 | 2.457669 |

`prediction_minus_target>0`表示对负回报预测得过于乐观，即低估未来代价。六组的平均偏差均为正，部分更新Explained Variance为负。Value Loss下降或绝对值较小不等于控制改善，也不是发散的充分条件。没有NaN/Inf、没有模型更新中断。

最大loss所在更新及实际任务：

| method | seed | update | stage | value_loss | target_mean | target_std | value_mean | task_composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_control | 42 | 14 | B | 10445.045670 | -62.164558 | 93.690300 | -14.706143 | {'B5': 72, 'B3': 440} |
| v2_control | 123 | 1 | A | 10947.042456 | -54.451710 | 89.898300 | 0.022737 | {'A3': 512} |
| v2_control | 2026 | 11 | B | 10231.904218 | -63.511726 | 89.253914 | -10.907181 | {'B1': 36, 'B3': 476} |
| v3 | 42 | 14 | B | 4852.961433 | -53.544365 | 63.180439 | -19.491627 | {'B3': 440, 'B5': 72} |
| v3 | 123 | 1 | A | 5115.571838 | -38.391788 | 60.911808 | 0.022737 | {'A3': 512} |
| v3 | 2026 | 11 | B | 4762.890170 | -53.601765 | 60.088497 | -15.415108 | {'B1': 36, 'B3': 476} |

上述峰值需要结合初始Critic预测接近零、实际目标负值及其方差理解。不同任务强度/课程确实产生不同目标分布，但本轮不是仅在后期大位移困难任务出现loss；不能把全部误差归因于扰动强度，更不能仅凭loss宣布Critic发散。完整96行包含target/value/GAE均值、标准差、范围、5/95分位，及该rollout实际task_composition。

## 独立GAE审计

96个512决策rollout逐个使用原reward、value、episode_starts、last_values和最后done重算GAE、return；最大优势误差 0.000249240098526、最大return误差 0.000249150042691，符合float32累计复算 `rtol=3e-6, atol=5e-5`。这是数值一致性容差，不是安全门槛。TimeLimit bootstrap保留，真正失败终止不bootstrap；预算尾端是训练片段，不伪造time_limit。

`pre_value_mse`是优化前rollout预测对目标MSE；SB3 `value_loss`是训练中各minibatch/epoch loss平均，两者不必一致。EV基于该rollout优化前预测与目标，不是优化后独立验证EV。三者在CSV分开记录。

## H2与ReturnNorm状态

V3最大Value Loss约为V2的一半，但原始奖励定义和目标尺度也改变；不能将该数字解释成纯Critic稳定性提升，更不能当作姿态抗扰改进。

ReturnNorm准入要求V3先通过共同物理收益门禁；本轮未通过，所以 **V3-ReturnNorm PPO Pilot和长训练均未执行**。没有归一化模型、没有归一化物理优越性结论；H2因果实验未完成。

仅完成工程验证：VecNormalize `norm_obs=False`、`norm_reward=True`、gamma沿用原值、clip_reward=1e6、epsilon=1e-8；1500真实物理步验证原始reward/物理状态不变、统计保存重载一致、评价 `training=False/norm_reward=False` 冻结统计。58维物理观测不做第二次归一化。

VecNormalize改变训练reward、GAE和Critic目标，也影响Actor优化，**不是Critic-only**。未实施PopArt或单独Critic目标缩放。不用封装测试冒充归一化PPO训练。

## 后续合理性

本轮支持“仍存在价值拟合问题”这一观察，不支持“已证明价值尺度是控制瓶颈”。若后续经审核继续，先采用独立、对等预算的训练诊断，并预先规定物理非劣标准；不要只追求Value Loss更小或无限扩大预算。本阶段停止于现有门禁。
