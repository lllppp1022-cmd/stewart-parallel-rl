# Phase 2C-1：Gymnasium环境、动作与采样边界

本阶段首次实际训练Stewart残差PPO，不修改Phase0～2C-0的模型、控制器和结果。正式运行目录为`results/runs/phase2c1_20261029_v1`；具体计数、评价及结论见《32_Phase2C1_PPO训练与评价报告》。本文件解释接口，不构成实机安全保证。

## 1. 真实动力学和时间尺度

`src/rl/stewart_residual_env.py`中的`StewartResidualEnv`包裹冻结的`PhysicalOwner`，继续使用原MuJoCo完整闭链模型、原控制执行路径和`mj_step`。模型为nq=43、nv=42、nu=12；前六通道驱动支链，后六通道施加独立外扰，Actor不能写后六通道。

物理积分及C4控制均为1kHz。Actor每50个积分步决策一次，即20Hz；动作保持50ms，C4、共享预算及保护仍逐1ms运行。正常回合6s、120次决策、6000次积分。保护在首次触发的物理步即停止当前动作采样，然后执行原0.2s保护保持；保持积分另行计数，没有新的Actor调用。Reset建立新物理owner并明确初始化状态；执行期间不通过写live qpos移动平台。

正常6s截止：`terminated=False, truncated=True`。保护失败：`terminated=True, truncated=False`。稳定到达由历史完整回合判据单独计算，时间截止本身不等于成功。训练预算结束时的16决策部分回合标记`budget_cutoff`，既不伪称完整成功，也不计为失败。

## 2. 一维Actor与原四维映射

动作空间为`Box(-1,1,(1,),float32)`。Actor原始输出来自SB3对角高斯；它可能超过[-1,1]，由SB3和环境裁剪。网络中的Tanh是隐藏层激活，不是输出的squashed Gaussian。

```
raw Gaussian action → Box clip → δFz = 0.5 * clipped_action [N]
                       → [δFz,0,0,0] → 原task4_to_leg
                       → 与C4共享原预算 → 命令力 → 隐藏lag → MuJoCo实际力
```

`action_adapter.py`复用原`YawObserverCompensation`和`task4_to_leg`。新增Fz与原C4的Yaw补偿合成，六支链补偿和C3共同经过原`SharedBudgetGovernor`：相对C0每支链±5N、20N/s，原生总力±400N。没有给PPO单独增加力预算。0.5N尺度通过9种任务组合×3类动作共27个完整回合预检才冻结使用；这只证明有限预检未触发保护，不是任意策略安全证明。

日志中的`rl_applied_command`是实际最终命令与“同一测量状态、同一历史预算下仅保留C4的单步反事实命令”的差。它用于分析限幅后的即时动作有效性，不是另一条完整C4闭环轨迹，也不是隐藏执行器的实际物理力。`F_command`、`F_drive`、`force`分别保存命令、隐藏一阶驱动力、MuJoCo实际力。投影原因、预算占用、native饱和等由原物理日志及控制trace保留。

## 3. 观测和可见信息

`MeasuredState`是白名单数据结构。当前位置/姿态/速度来自C4最新测量，一般比新积分状态滞后1ms；六支链q/dq采用当前理想反馈。其他输入为当前参考、已知C4命令、C4等效扰动估计及上一实际学习补偿。没有读取实际tau、隐藏drive、真实载荷、未来加载时刻、任务编号、回合剩余时间或真实模型参数。

当前帧55维，加50ms因果响应摘要30维，共85维；最近三帧堆叠为255维。摘要包含区间位置误差均值/绝对峰值、速度均值、估计变化、命令变化及实际残差均值。Reset以初始帧填充历史，没有未来帧。当前单帧85维版本已通过接口测试，本轮没有额外训练该消融。

归一化是`configs/phase2c1_env.yaml`中的固定物理尺度，没有VecNormalize或评价更新统计量。主要尺度：XYZ误差[0.25,0.25,0.05]mm、姿态0.02deg、线/角速度0.01m/s和0.05rad/s、行程1mm、支链速度0.01m/s、已知命令5N、估计20N/0.8Nm、学习残差0.1N。最终float32观测裁剪至[-10,10]并检查有限值。

这是理想全链q/dq反馈的仿真，不是具备全部传感器和噪声处理的硬件系统。

## 4. 奖励和终止计分

每个真实物理步计算Z/XY/SO(3)误差、实际命令学习残差幅值、变化率和请求动作代价；单位和权重全部冻结在env YAML。Z尺度0.05mm，因此小误差仍产生可测量代价。所有时段都计分，不通过加载标志选择奖励窗口。

单物理步综合代价上限20，通过同一缩放因子保留分量比例。决策奖励为负代价积分除以50ms。保护失败再扣`20+20*ceil(剩余时间/0.05)`，避免从同一前缀提前失败省掉剩余最大代价。保护保持的真实代价和物理步同样记录。这不是“保证所有不同任务的回报排序”；它约束同一前缀之后的故意失败收益。

`training_samples.csv`保存物理奖励；`rollout_*.npz`另存SB3有效奖励。SB3遇TimeLimit截断会加入`gamma*V(terminal_observation)`，所以有效奖励与物理奖励可能不同。此项来自当时Critic，不是实际未来回报。GAE使用有效奖励、价值预测及episode_starts计算，分别独立复算。不能把正常bootstrap差值误报为奖励实现错误。

## 5. 任务、随机数和预算

Reset只从六种预检过的训练任务T0～T5采样；tau为2/5/10ms，载荷约−5～−6N，加载时刻和持续时间不同。V0专用于2048和4096决策检查。最终E0采用未训练的7.5ms tau，E1采用新组合与加载时刻；只在安全预检中检查可运行性，模型选择不读取其评价表现。正式最终评价清单在三种子训练结束后、评价前另行冻结。

本轮没有混入质量、增益、噪声或观测延迟。5ms观测延迟、四维联合动作和更长历史均未训练。

三个训练种子42、123、2026，每个累计阶段256→2048→4096；前缀计入总预算。rollout长度256，增量均为256整数倍，不发生SB3向上凑整超预算。每256次决策更新Actor/Critic；优化时物理步计数保持不变。每次10 epochs×4 minibatches，正常为40 Adam steps；不能把epoch计数当成rollout更新次数。

停止条件事先冻结：非有限数值；最近三回合至少两次保护失败；V0任务不成功，或加载IAE超过C4两倍，或实际力峰值变化率超过C4的1.5倍。这是有限Pilot运行门禁，不是宣称性能优秀的标准。

## 6. 记录和观察器

每个物理步保存`physics.csv.gz`和458维`mjSTATE_INTEGRATION`，另存Python一阶lag配置、初始驱动和逐步drive日志。MuJoCo积分状态本身不含全部Python控制器记忆，不能将单个state向量称为可独立恢复完整Python训练的checkpoint。策略checkpoint是SB3模型/优化器，不是物理进程暂停快照。

`ppo_trace.py`只截获原始Actor已经生成的一次随机动作，不额外采样，不执行第二条predict轨迹。每次更新记录参数变化范数、GAE、价值目标、loss/KL/entropy/clip_fraction/EV、优化耗时和模型SHA256。保存/重载与验证用`preserve_rng`隔离Torch、NumPy和Python随机状态，防止重载模型影响后续训练采样。

只录seed42预先选定的episode2、16、32，25fps真实现场RGB。视频脚本只读这些RGB和状态日志，不运行策略，不积分，不重建假现场。曲线只显示当前帧及过去。episode2跨越首次256步更新，版本按每个决策准确标记。实际训练视频与重新运行确定性策略评价视频分开。

原Qt仪表盘/Windows原生镜像Viewer已做固定动作off/on对照：50次决策、2500积分步，两侧完整状态、观测、奖励逐位相同。该实验是环境观察器验证，不是两次独立PPO训练参数哈希对照。正式训练采用无GUI+选定回合现场录制；本轮未把Qt操控窗口直接接入PPO主循环。避免让窗口按钮改动训练目标。

## 7. PowerShell接口

从项目根目录使用独立stewart_rl环境；旧项目不需重新安装。

```powershell
Set-Location __PROJECT_ROOT__
$py = 'D:\miniconda\envs\stewart_rl\python.exe'
& $py -m pip check
& $py -m pytest -q -p phase2a_pytest_temp_plugin -p phase2c1_pytest_policy
Invoke-Item .\assets\videos\stewart_phase2c1_ppo_training.mp4
Invoke-Item .\assets\videos\stewart_phase2c1_ppo_evaluation.mp4
Import-Csv .\results\phase2c1_update_events.csv | Select-Object seed,update,global_decisions,actor_change_norm,critic_change_norm,value_loss
```

以下是本轮实际使用过的入口形式，**不要直接复用已有run目录重跑**，脚本会拒绝覆盖。当前阶段结束后不会自动再训练。完整新增运行需新目录、复制预检、重新执行门禁，并遵守同样预算。

```powershell
& $py scripts\train_phase2c1_ppo.py --run-root results/runs/phase2c1_20261029_v1 --seed 42
& $py scripts\evaluate_phase2c1_ppo.py --run-root results/runs/phase2c1_20261029_v1 --part freeze
& $py scripts\evaluate_phase2c1_ppo.py --run-root results/runs/phase2c1_20261029_v1 --part classical
& $py scripts\evaluate_phase2c1_ppo.py --run-root results/runs/phase2c1_20261029_v1 --part model --seed 42
& $py scripts\analyze_phase2c1_training.py --run-root results/runs/phase2c1_20261029_v1
```

重现入口不是本轮已完成预算外的继续训练指令。视频、CSV、TensorBoard均可直接查看，无需重跑网络。
