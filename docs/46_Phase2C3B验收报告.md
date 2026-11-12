# Phase 2C-3B 验收报告

日期：2026-11-05；工程`__PROJECT_ROOT__`；独立实验目录`__PROJECT_ROOT__\results\runs\phase2c3b_20261105_v1`。

## 结论与实际交付

本轮完成参考动态FF审计、9组开发、24组正式、6组滞后、3组联合前缀、3组完整负载和4组固定动作权限，共49组研究执行。其中46条完整轨迹完成，3条预检从未冒充完成任务；没有PPO/SAC/TD3/BC训练。历史6135个文件SHA逐一保持，9条冻结基线完整状态与历史逐位相同。参数和方法选择在正式评价前冻结。

| rise_s | C0 | C4 | FF-V | FF-VA | FF-VA-Lead | fourDlead |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.248844659 | 0.248002758 | 0.0238192288 | 0.013300423 | 0.0133105836 | 0.247990913 |
| 6 | 0.173702389 | 0.173124875 | 0.0134051912 | 0.00931867982 | 0.00932654019 | 0.173116045 |
| 8 | 0.133583023 | 0.133143502 | 0.00931594766 | 0.00721068468 | 0.00721689992 | 0.133136549 |
| 10 | 0.108565511 | 0.10821165 | 0.00719226686 | 0.00590745551 | 0.00591254349 | 0.108205937 |


按开发集Z误差选FF-VA，不是全维最优结论。T4下Z-RMSE从0.248003降到0.013300mm，但XY峰值由0.018800增至0.264924mm，三维RMSE从0.248195降到0.124887mm；FF后非Z误差占剩余三维平方误差约98.9%。不能把Z改善概括成全维精度提升。

FF主要解决名义参考动态误差，并不能据此证明RL比经典控制有收益。新任务设计判断：**LARGE-MOTION RL PILOT CONDITIONALLY BLOCKED**，先诊断新增非目标耦合，再决定是否进入有限PPO，见45号报告的待执行门禁。

## 测试、数据复算与视频

完整测试实际输出：`303 passed, 1 xfailed, 2 warnings in 330.34s (0:05:30)`。历史xfail仍单列，未计普通通过。新增测试包含解析参考导数、J力映射、零FF等价、共享幅值/限速、真动态、历史冻结、T8、RGB/状态一致、GUI开关一致性及数据复算。完整日志`results/phase2c3b_pytest.log`；预检9测试的先行日志独立保留。

49条保存的全采样率1ms物理日志重新算Z/3D/SO(3)、力峰值/变化率/平方积分和每链`Fcommand−FC0`预算，结果`__PROJECT_ROOT__\results\runs\phase2c3b_20261105_v1\independent_audit.json`，通过=True。输入异常与提前结束分开处理，预检没有下降阶段就存空值，不虚构成功。

视频`assets/videos/stewart_phase2c3b_dynamic_feedforward.mp4`，1920×1080/25fps、1040帧、41.60秒、2倍播放；5条现场RGB源均逐帧校验qpos SHA与积分状态一致，真实Z变化>49mm。不是重新写qpos生成的运动学录像，也不是PPO训练。视频显示六链行程/力、Z误差、参考v/a、FF请求、预算占用和结果。

GUI对照见`__PROJECT_ROOT__\results\runs\phase2c3b_20261105_v1\live_viewer_validation.json`。CPU墙钟受渲染影响，不改变实际积分。完整实验物理步807000，GUI两次额外16000，测试另计；本轮PPO决策数0、学习更新数0。

## 可直接运行的真实窗口命令

```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\miniconda\envs\stewart_rl\python.exe'
& $py scripts/view_phase2c3b_live.py --method FF-VA --rise 6
```
这是新建独立日志目录的真实动力学执行；不是PPO采样。Pause / Resume暂停物理时钟，Camera left/right只改变相机，Quit关闭并归档。默认完整15秒物理轨迹后停留显示，不再积分；墙钟上限300秒。不会无限重复运动，也不修改原有持续运行控制面板。

下列有界开/关对照已实际执行两次（第二次包含六支链曲线）：
```powershell
& $py scripts/view_phase2c3b_live.py --method FF-VA --rise 6 --validate-steps 4000 --compare-headless
```
每次开启与关闭显示各4000个物理步，完整积分状态逐位一致。GUI验收只走到4秒、约25mm；全50mm往返的证据来自正式物理日志和现场RGB录像。窗口采用Windows Qt＋MuJoCo Renderer单线程所有权，不是原生GLFW Viewer；没有从GUI线程修改live状态。首次尝试ProcessPool被Windows IPC权限拒绝、未执行该批动力学，错误日志保留，随后改成无IPC的独立CLI进程。


只复算，不进行新物理执行：
```powershell
& $py scripts/analyze_phase2c3b_results.py
& $py scripts/analyze_phase2c3b_rl_task.py
& $py scripts/validate_phase2c3b.py
& $py -m pytest -q -p phase2a_pytest_temp_plugin -p phase2c1_pytest_policy
```
重新生成视频会读取归档RGB，非重新控制或训练：`& $py scripts/render_phase2c3b_video.py`。原始研究runner为有意拒绝覆盖目录的实验入口，复现新实验需显式新run，不能复制命令覆盖当前证据。单一窗口每次使用time_ns独立目录。报告生成不修改任何旧模型或旧数据。

## 图表索引

全部位于`assets/plots/`：历史Z误差/速度/加速度联合图`phase2c3b_historical_T4/T6/T10.png`；实际/参考及V/A分量`phase2c3b_T4/T8/T10_comparison.png`；速度与XY代价`phase2c3b_speed_comparison.png`；六链行程和实际力`phase2c3b_six_legs.png`；请求/接受/物理广义力`phase2c3b_command_drive_wrench.png`；预算与原始1ms附加变化率`phase2c3b_budget_and_force_rate.png`；滞后/负载响应`phase2c3b_lag_load.png`；XY及姿态退化`phase2c3b_coupling_tradeoff.png`；Jacobian/weld/几何间隙`phase2c3b_safety_curves.png`；RL候选任务`phase2c3b_rl_candidates.png`；固定探针分阶段利弊`phase2c3b_action_phase_effects.png`。图可降采样显示，主要指标全部来自1ms原始日志。

## 25项问题逐项回答

### 1. 50mm误差与哪些变量相关？

完整1ms分解显示移动方向、参考速度和加减速过渡均相关，CSV保留分段系数。

### 2. 是否有速度特征？

上升实际−参考以负值为主、下降反向，速度FF配对明显降低误差。

### 3. 是否有加速度特征？

VA比V进一步降低残差；这是本模型的干预证据，不是所有误差来源的唯一证明。

### 4. 原C4是否有动态补偿？

已有含参考速度的PD、静态平衡及含M/D/K的观测补偿，缺少此次显式D*vref+M*aref。

### 5. kv来源？

冻结名义Dzz=570.453024115 N·s/m。

### 6. ka来源？

约束切空间等效Mzz=50.185328795kg，并非20kg平台质量。

### 7. 是否重复补偿？

不增加重力；明确区别参考动态FF与速度误差PD，实际配对及静止归零验收通过。

### 8. FF-V优于C4？

本轮T4/6/8/10的Z-RMSE均更低，其他维度另列，不能概括成全维更优。

### 9. VA优于V？

四个正式时间均进一步降低Z-RMSE。

### 10. VA-Lead优于原fourDlead？

Z误差明显更低；VA-Lead不稳定优于VA，开发按既定规则选VA。

### 11. 三档改善一致？

T4/T6/T10一致，独立T8同方向。

### 12. 预算不足？

完整实验最大投影占比0.00133333333；本矩阵没有明显预算瓶颈，不代表所有未来状态。

### 13. X/Y偏差？

会增加。T4峰值由0.018800增到0.264924mm，约14.1倍；三维RMSE仍下降，但非目标退化必须诊断。

### 14. 姿态漂移？

SO(3)与Roll也增加到千分之几度量级，未触保护；不是所有方向精度都提升。

### 15. 实际力合理？

峰值91.639229N，未触±400N饱和；全力变化与附加20N/s分开。

### 16. 5ms下有效？

见严格同T6对照，FF-VA仍低于原C4/fourDlead，无安全失败。

### 17. 10ms下有效？

同T6无载通过；只支持这条已验证工况，未测试10ms叠加载荷。

### 18. −5.5N下能保持？

tau5ms联合7秒预检后3条完整15秒均完成，未降低门槛。

### 19. 哪些误差已解释？

大部分名义参考速度/加速度跟踪误差可由经典FF显著减少。

### 20. 未知部分？

严格配对时序差分保留滞后和负载的闭环额外影响；几何变参、非目标方向也未全部解决。

### 21. 最适合PPO任务？

先固定L50-T6 tau5ms及受限−5.5N负载，不一开始全参数随机。

### 22. 一维Fz？

仍适合首先验证主要Z残差，固定探针证明有限权限；不保证能学习或处理XY。

### 23. 归一化？

50mm宏观位移/slide与微米级误差分尺度，候选53维因果单帧，训练统计冻结。

### 24. 哪些随机化？

先已验证无载tau5/10和tau5负载，再逐项联合预检；10ms＋负载/质量增益未执行。

### 25. 是否值得下一阶段？

LARGE-MOTION RL PILOT CONDITIONALLY BLOCKED；物理动作筛选通过，但先诊断FF的XY/姿态代价及完整任务强基线，再决定是否启动有限PPO。


## 剩余限制与未执行事项

本轮只验证MuJoCo纯世界Z正向50mm任务。887个独立网格凸包查询配对、103个连接组件排除仍属PARTIAL覆盖；30个被动铰链没有原生机械硬止挡。无原生接触不等于全几何无碰撞，更不等于实机安全认证。

未实施未来50mm Gym环境、新Actor、观测归一化/奖励验收、PPO训练、20/50Hz PPO对照、跨种子学习评价、tau10ms＋负载、随机质量/增益、全6DoF大位移或实机测试。固定探针窗口改善不是学习成功，确定性研究完成率不是随机鲁棒成功率。旧3mm PPO权重保留，未在50mm上直接部署。

本轮到此暂停，等待审核，不自动进入下一阶段。
