# Phase 2C-8 最终验收报告

日期：2026-11-09（Asia/Shanghai）；独立实验目录：`__PROJECT_ROOT__\results\runs\phase2c8_20261109_v1`。所有数值由原始CSV/NPZ/JSON生成，不修改历史结果。

**阶段判定：PHASE 2C-8 PARTIAL SUCCESS。实际完成六组8192决策匹配PPO Pilot、60次开发评价和32次最终评价；V3未通过训练扩展门禁。局部姿态改善存在，但尚无稳定超过强传统控制的证据。**

这不是32768完整训练成功，也不是归一化实验完成。H1奖励更换的稳定收益假设本次未获支持；H2归一化因果试验未启动。

## 交付与实际结果

| method | seed | new_decisions | physical_steps | complete_episodes | fragments | updates | adam_steps | failures | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_control | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.071634 |
| v2_control | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 803.080746 |
| v2_control | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 818.692782 |
| v3 | 123 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.266636 |
| v3 | 2026 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 802.062844 |
| v3 | 42 | 8192 | 163840 | 10 | 2 | 16 | 1280 | 0 | 819.255763 |

正式PPO：49152决策、983040物理步、96次更新、7680次Adam，60个完整回合加12个预算片段。全部Actor/Critic真实变化；原始随机动作与每1ms实际状态完整保存。训练保护失败0。

| method | load_iae_deg_s | unload_iae_deg_s | so3_peak_deg | omega_peak_rad_s | xyz_rmse_mm | z_rmse_mm | xy_peak_mm | force_rate_N_s | force_squared_integral_N2_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.006745369 | 0.001882850 | 0.012019611 | 0.005513322 | 0.097514186 | 0.009229255 | 0.208619390 | 39.415338504 | 688112.452741372 |
| classical50 | 0.006637909 | 0.001849752 | 0.011945082 | 0.005511254 | 0.096606129 | 0.009226960 | 0.206269534 | 39.414804058 | 688112.431412519 |
| v2_control | 0.006696714 | 0.001778534 | 0.012009532 | 0.005512858 | 0.095902074 | 0.009221658 | 0.204358815 | 39.412371007 | 688111.566510175 |
| v3 | 0.006677556 | 0.001817157 | 0.011999991 | 0.005512834 | 0.096364649 | 0.009224670 | 0.205274790 | 39.413328398 | 688111.901657252 |

32次最终评价全部安全完整结束。微小数值改善不宣称统计显著；仅三个训练种子、四个有限任务。每个种子保留，不筛掉退化种子。详见报告76逐任务表。

| seed | all_task_cost_guard | safe_complete | promising | no_large_iae_regression | load_improvement_pct | unload_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | True | True | False | True | 2.710042 | -1.263222 |
| 123 | True | True | False | True | 0.690514 | 3.517143 |
| 2026 | True | True | False | True | -3.721075 | -3.435162 |

扩展门禁0/3种子满足同时改善要求，因此没有进入Stage C或32768，也没有ReturnNorm PPO。不对V3长期训练或未运行的归一化作预测性优越结论。

## 验收及保留问题

- 历史SHA：41396文件，0变化；旧模型、报告、视频未覆盖。
- 原测试：416通过、1失败。失败为 `tests/test_phase0.py::test_isolated_environment_no_learning_packages`，旧Phase0断言环境没有Torch/SB3；当前按后续研究安装了这两个包，条件冲突。未删除、skip或篡改此失败，原日志及JUnit保留。
- 新验收：36通过、0失败；合并本次执行总计452通过、1失败，不能报告“所有测试通过”。
- 新测试覆盖RNG、阶段恢复、匹配初始化/预算、58维/2维接口、控制限幅、零动作、reward重算、真实参数变化、模型重载、测试隔离、归一化封装、GUI、视频和负门禁。
- 96个GAE rollout独立复算通过；正式983040物理行reward全部复算通过；最终32条轨迹×24指标复算通过，另核对恢复状态。
- 最终外载逐位比较首次失败由约1e-14量级坐标变换浮点舍入引起，日志保留；改为1e-12数值核验，不改变安全门槛或原始结果。
- 两次Windows Qt真实GUI（V2/V3）分别1000物理步：暂停零积分、单决策20步、前缀与无GUI保存状态逐位一致、关闭正常归档。物理均通过实际ctrl与mj_step执行。
- 归一化1500物理步仅封装测试，不计为归一化PPO训练。任意回合中途崩溃续训尚不支持；声明的课程边界恢复已通过。

原测试命令（本次已执行，输出保留）：

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests --ignore=tests/test_phase2c8.py -q --basetemp=results/runs/phase2c8_20261109_v1/pytest_history --junitxml=results/runs/phase2c8_20261109_v1/historical_tests.xml
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests/test_phase2c8.py -q --junitxml=results/runs/phase2c8_20261109_v1/new_tests.xml
```

再次运行测试请换独立basetemp/日志名，避免pytest清理既有测试临时归档。模型原始运行目录不作为临时目录。

## 观看真实机器人与训练视频

GUI复用既有MuJoCo+Qt架构，显示冻结策略重新执行的真实动力学；不是重新训练，也不是训练现场。启动命令的两个方法均已用 `--validate-steps 1000` 实际测试。正常观看去掉该验收退出参数：

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/view_phase2c8_policy.py --method v3 --seed 42 --task T8_mix
& D:\miniconda\envs\stewart_rl\python.exe scripts/view_phase2c8_policy.py --method v2_control --seed 42 --task T8_mix
```

控制面板可选任务、baseline/classical50/PPO及六个已验收模型，选择后Reset生效。Pause暂停积分，Single decision只推进20个物理步；Reset新建独立运行日志。相机拖拽/滚轮；任务完成后停在终态，Reset可重新执行；关闭窗口归档退出。不能把本GUI标签“LIVE DETERMINISTIC EVALUATION”称为在线训练。

训练过程请看实际采样RGB视频和日志：

| file | seconds | frames | fps | size_MB | sha256 |
| --- | --- | --- | --- | --- | --- |
| stewart_phase2c8_training.mp4 | 60.000000 | 1500 | 25 | 7.911659 | c3e48274f65841bf38eb5d14605ff4ea41e87cd1348d12d2568cc5894164477e |
| stewart_phase2c8_reward_comparison.mp4 | 128.000000 | 3200 | 25 | 10.230187 | e1b9fd0025d78708d6070ec9574667720ecbcf227bb3bbee063c03165739061d |
| stewart_phase2c8_attitude_stability.mp4 | 75.000000 | 1875 | 25 | 6.532095 | 80155f4a249d616344e911529c7a42336399812ac1c14af82ed1900de892a15b |

三个视频1920×1080、25fps，均经过正常解码、不同帧变化、关节位移及时间/状态SHA对应检查。另已查看三张代表帧，文本、机器人和曲线清晰。训练视频为seed42的V2/V3各A/B真实episode拼接，B含50mm真实升降；来自训练现场RGB，图表后处理，不重新predict生成轨迹。画面显示policy version，回合跨更新时会变化。未录制不存在的Stage C训练。另两视频分别T8_mix八控制器、T8_combined50传统/经典/V3三种子的**独立确定性评价**，各自有明确标签。

训练扰动引起的姿态偏差本来很小；没有放大机器人真实运动。右侧角度/误差曲线使用自适应纵轴帮助看清，标题保留单位和真实范围。50mm升降是真实位移。单条演示不是整体成功率证据。

```powershell
Start-Process .\assets\videos\stewart_phase2c8_training.mp4
Start-Process .\assets\videos\stewart_phase2c8_reward_comparison.mp4
Start-Process .\assets\videos\stewart_phase2c8_attitude_stability.mp4
```

视频已实际解码并检查；上述系统关联播放器命令只是打开同一文件，不声称额外进行过播放器GUI自动测试。

TensorBoard已在 [http://127.0.0.1:6034/](http://127.0.0.1:6034/) 运行，HTTP和18run标量接口验证通过。若以后进程已结束，可以用本次实际启动参数重开：

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe -m tensorboard.main --logdir_spec "actual:results/runs/phase2c8_20261109_v1/training,derived:results/phase2c8_tensorboard" --host 127.0.0.1 --port 6034
```

## 25项研究问题

### 1. V3是否更符合姿态抗扰目标？

公式增加平滑绝对误差，结构上更接近IAE；实际并未证明全时域折扣目标等价于两个事件窗口的抗扰目标。

### 2. 相对V2是否改善加载IAE？

最终跨种子/任务均值改善0.286%，幅度小且不一致；开发三种子分别一升一降等取舍，没有通过共同门禁。

### 3. 是否改善卸载IAE？

最终均值改善-2.172%（负值即恶化），不能说更好。

### 4. 是否降低姿态峰值？

最终均值改善0.079%，属于微小差异，未宣称统计显著。

### 5. 是否降低最大角速度？

最终均值改善0.00042%，近乎不变。

### 6. 是否增加XYZ误差？

相对V2最终XYZ_RMSE改善-0.482%（负值为增加），无超过预设10%护栏的全面恶化。

### 7. 奖励与物理评价是否冲突？

固定开发集有2组出现共同V2计分比初始提高、但至少一个IAE恶化。总回报结合全程姿态、位置与动作代价，不能替代事件窗口IAE；逐组数据见报告74。

### 8. Critic是否存在持续偏差？

六组平均prediction-minus-target均为正，部分更新EV为负，表明对负回报存在过于乐观的拟合偏差；没有NaN/Inf，不称发散。

### 9. 回报尺度处理是否改善EV？

未执行ReturnNorm PPO，无法判断；只验证封装、统计保存/冻结及物理等价。

### 10. 回报尺度处理是否改善实际控制？

未执行，不作结论；V3 loss更小不是独立归一化试验。

### 11. 三个种子是否一致？

不一致。V3相对classic每种子只在一个最终任务同时改善两个IAE；任务和改善方向因种子而异。

### 12. 是否仍有加载改善/卸载恶化？

有。例如V3 seed2026在T8_combined50加载改善而卸载恶化；开发seed42同样存在取舍。

### 13. 卸载后是否有残差偏置？

有。V3卸载后0.1～1s沿外载方向请求均值在-0.004863～0.008814N·m之间；非零偏置不自动等于所有状态都有害。

### 14. 残差是否方向正确？

部分状态有恢复方向，部分相反；方向要结合误差、速度和基线补偿。不能据相关性宣布准确估计未知负载。

### 15. 共享预算是否显著限制权限？

事件瞬间存在投影/限速，独立命令重算已证实；多数最终物理步未投影。不能证明权限是唯一或主要瓶颈，也不应放宽预算。

### 16. V3是否优于冻结模型辅助控制？

最终均值加载改善1.005%、卸载改善3.489%，有局部/平均收益，但不形成跨种子任务稳定优越性。

### 17. 是否优于同权限classic？

最终均值加载改善-0.597%、卸载改善1.762%；一项改善、一项退化，未通过成功标准。

### 18. 是否改善未见扰动方向？

新组合任务存在局部收益与退化；同一试验同时改变方向/时间/参考，不能单独归因方向泛化，也不声称每个方向分量此前都未出现。

### 19. 是否改善未见加载时刻？

最终事件时刻未用于调参，但无稳定改善；没有独立的仅改变时刻因子实验。

### 20. 是否改善动态Z+Roll/Pitch？

T8_combined50确实完成50mm和±0.25度真实动态运动，收益有加载/卸载取舍。未训练Stage C，结果不能外推为完成全课程训练后的上限。

### 21. 是否值得研究历史观测？

可作为以后单独、有限预算假设，先验证历史误差是否提供可识别信息；本轮没有证明增加历史一定解决问题，不自动新增网络。

### 22. 是否值得研究未知执行器滞后？

有物理动机，但需冻结新的未知滞后分布、强控制基线与可控性/安全试验。本轮没有做未知滞后的单因素辨识验证。

### 23. 主要瓶颈是什么？

已观察到目标不完全一致、价值拟合偏差和持续残差偏置；动作受限是系统事实，历史信息不足尚未经消融。证据不足给出唯一主因。

### 24. 是否证明额外未知扰动适应？

没有。局部改进不等于因果在线辨识，未稳定超过classic，也没有足够跨分布证据。

### 25. 是否进入更复杂RL架构？

不建议立即升级LSTM/Transformer或扩大训练。先审核此有限负/部分正结果，明确是否值得新的受控诊断；本轮停止。



## 未执行项及下一步

未执行：Stage C追加24576决策、每组累计32768、ReturnNorm PPO、2.4N·m专项、4.8N·m正常探索、LSTM/Transformer/SAC/TD3、新硬件实验。未创建虚构checkpoint或归一化结果目录。

当前建议保留强经典控制作为主基线，接受此有限Pilot的局部/负结论。以后若获授权，优先验证信息可辨识性、奖励窗口取舍和未知滞后的独立控制问题，再决定是否值得新RL架构；不凭Reward或最佳种子追加预算。

报告72～76包含协议、奖励、训练、Critic和逐任务行为证据。可机读总判定为 `results/phase2c8_final_decision.json`，安全/预算/测试汇总为 `results/phase2c8_safety_audit.json`，源文件清单随最终artifact manifest归档。

**本阶段停止，等待用户审核。**
