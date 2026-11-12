# Stewart Phase 2C-0：未知执行器动态、突变负载与任务可行性

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 正式决策

**PHASE 2C PILOT CONDITIONALLY BLOCKED**。

本轮取得正证据：未知10ms滞后和5ms平台延迟均形成安全可重复退化；有限因果四维补偿在两个新冻结评价中分别改善10.82%和17.16%，通过事前工程筛选门槛，不能忽略这些结果。

但相同收益已经由廉价经典规则实现。尚未建立PPO在强C4+4Dlead之上的额外收益依据；两类Home任务本来均在严格精度带内，原大横向误差又超出现有四维直接权限。综合判断是**物理可行性已有，额外学习价值仍需确认**，不把最低工程筛选门槛自动等同必须训练。没有声称PPO无效，也没有为了路线预设GO。

如果用户目标仅为当前容差和抗扰改善，可直接保留经典方案，暂不投入训练。若希望研究更高精度和跨未知动态适应，应先冻结新独立条件、历史观测和强经典基线，再由用户审核是否值得一次明确小预算Pilot；本轮不自动封装或训练。

## 实际运行与预算

| stage | trials | successes | physics_steps |
| --- | --- | --- | --- |
| authority_diagnostic | 1 | 1 | 6000 |
| authority_preflight | 12 | 12 | 72000 |
| delay_development | 9 | 9 | 54000 |
| differential_lag | 6 | 6 | 36000 |
| frozen_evaluation | 6 | 6 | 36000 |
| lag_development | 40 | 40 | 240000 |
| lag_safety | 3 | 3 | 18000 |
| noise_development | 9 | 9 | 54000 |
| paired_decomposition | 3 | 3 | 18000 |
| privileged_diagnostic | 1 | 1 | 6000 |
| zero_equivalence | 4 | 4 | 24000 |

研究合计94×6秒=564000个1ms积分步（564仿真秒），94回合均完整完成，0新保护失败；预先上限96，未凑满。全部失败/退化均保留；无新失败不能人为制造失败视频。旧已知失败和5803历史文件SHA256均不变。

附加成本单列：GUI off/on各2048步，共4096；两轮日志吞吐量基准各4000步，共8000（第一轮与主研究并行，第二轮无其他本代理模拟/渲染负载）。已知非自动测试实际积分合计576096。解析ZOH4000数组采样不计MuJoCo步；自动测试内短物理运行另计，不伪称RL样本。

## 冻结评价及预先门槛

开发筛选只看相对同任务τ0 C4的加载位置IAE，完整成功且退化≥10%才入选，最多2类。实际入选10ms统一lag（开发退化32.16%）、5ms部分观测delay（12.03%）。新评价载荷改为−5N、1.55～3.35s，所有参数、载荷、方法、门槛在新评价前写evaluation_frozen.json；没有用最终结果调参。

C4参数不变；C4_fourD为20Hz等效扰动差分lead，.02s系数，Z/R/P/Yaw额外限±[.5N,.025Nm,.025Nm,.015Nm]，总Yaw请求仍±.30Nm。C4_fixedLead为独立经典对照：3ms当前/过去C4**请求命令**导数超前，导数滤波4ms、逐腿额外限±.1N；它不是读取真实τ，也不是actual drive。两种新增补偿都与C3/C4共用±5N、20N/s。

事前收益门槛：加载位置IAE至少降低10%；无新增失败；实际总力峰值变化率不增>5%、力平方积分不增>2%、非目标XY峰值不增>5%。原完成标准/严格恢复/安全完全不变。

| task | variant | load_position_iae_mm_s | load_peak_position_mm | unload_reverse_z_mm | max_force_rate_N_s | force_square_integral_N2_s | xy_peak_mm | success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval_actuator_lag | C4 | 0.014179 | 0.066700 | 0.066197 | 31.838856 | 272066.726163 | 0.008241 | True |
| eval_actuator_lag | C4_fixedLead | 0.014590 | 0.066223 | 0.065720 | 31.846398 | 272066.723288 | 0.008638 | True |
| eval_actuator_lag | C4_fourD | 0.012645 | 0.066158 | 0.065654 | 31.838856 | 272066.748672 | 0.007414 | True |
| eval_platform_delay | C4 | 0.012395 | 0.054879 | 0.054370 | 38.401611 | 272066.650870 | 0.006885 | True |
| eval_platform_delay | C4_fixedLead | 0.011741 | 0.050663 | 0.050153 | 37.396856 | 272066.635819 | 0.006618 | True |
| eval_platform_delay | C4_fourD | 0.010268 | 0.053203 | 0.052692 | 38.401609 | 272066.660841 | 0.005710 | True |

| task | method | primary_error_reduction | force_slew_ratio | force_square_integral_ratio | xy_peak_ratio | passed |
| --- | --- | --- | --- | --- | --- | --- |
| eval_actuator_lag | C4_fixedLead | -0.028980 | 1.000237 | 1.000000 | 1.048139 | False |
| eval_actuator_lag | C4_fourD | 0.108222 | 1.000000 | 1.000000 | 0.899578 | True |
| eval_platform_delay | C4_fixedLead | 0.052779 | 0.973836 | 1.000000 | 0.961286 | False |
| eval_platform_delay | C4_fourD | 0.171626 | 1.000000 | 1.000000 | 0.829357 | True |

工程筛选2/2通过（C4_fourD）。3ms经典超前在lag中IAE**恶化约2.90%**，在delay中仅改善5.28%，两者均不通过10%门槛；未隐藏。四维lead自身也是经典因果控制，不是学习结果。未来即使训练，必须与C4+fourD这条强基线比较；仅重现该规则不能叫RL额外控制收益。

## 特权物理诊断

| task | variant | load_position_iae_mm_s | load_peak_position_mm | max_force_rate_N_s | success |
| --- | --- | --- | --- | --- | --- |
| eval_actuator_lag | privileged | 0.012913 | 0.062836 | 31.758335 | True |

仅对10ms候选，使用真实τ/内部drive/actual gain/当前外载作受限逆响应，标记PRIVILEGED PHYSICAL DIAGNOSTIC；不进入正式方法排行、不喂Actor。其峰值比C4小，但IAE不及非特权fourDlead。预算投影和所选逆响应不保证最优，不能叫严格理论上界或PPO可达成绩。信息多不自动保证控制更优。

## Gate A～E

| 门禁 | 证据与判断 |
| --- | --- |
| A物理可信 | 通过：ZOH解析核对、实际ctrl/force响应、关闭/τ0逐位等价、无隐藏驱动力泄漏 |
| B任务可控 | Home竖向候选通过；原uneven_dynamic的四维横向任务仍阻塞，不能靠改奖励补权限 |
| C信息可观测 | 对聚合误差的因果补偿有证据；未证明6τ独立可辨识或完整MDP，保留历史/部分观测限制 |
| D研究价值 | 预设工程筛选通过；PPO超越经典fourDlead的额外价值仍未验证，因此保留条件阻塞 |
| E训练成本 | 支持明确有限的物理预算估计；没有实测PPO优化/重置/评价总开销，不能据GPU推算 |

## 吞吐量实测

| journal_stride | physical_steps | construction_seconds | run_and_archive_seconds | physical_steps_per_wall_second | append_pipeline_seconds | recorded_rows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2000 | 0.167366 | 6.476009 | 308.832203 | 1.529528 | 2000 |
| 20 | 2000 | 0.134687 | 4.798320 | 416.812508 | 0.078586 | 100 |

隔离运行的2秒Home基准：完整逐步日志约308.83物理步/s；每20步保留日志约416.81步/s，控制、滞后、Jacobian、安全仍每原周期执行。保存样本的完整积分状态逐位一致。append计时只含该管线，不含全部原始行构造和最终NPZ压缩，不能伪称它是全部日志开销；稀疏benchmark文件的旧force-rate列不能用于科学指标，正式研究全部1ms原始日志不变。

每回合真实loop墙钟及含构建/归档/指标复算总墙钟在run_timing.csv，含RGB回合单列，不拿渲染成本伪称算法开销。前一并行负载基准也保留。未根据GPU型号估算速度。

若未来20Hz每决策50物理步，1024决策=51200物理步，仅按当前完整日志Home基准约166秒；2048决策=102400步约332秒。这些不是训练已运行时间，未含Reset/保持、Actor推理、PPO优化和评价，动态工况也可能不同。本轮没有训练。

## 验收、数据与视频

238项测试实际通过，含全部原212项及新增17项单元/9项验收证据测试。保留原2条pandas弃用警告，无测试被删/改来通过。94回合原始CSV与NPZ完整qpos/qvel/time逐位一致；每步ZOH、三力关系、400N限制、共享预算、observer已知命令记忆、传感器时间与相同噪声独立复核；实际几何映射与scratch核对最大差0。

Windows原生MuJoCo+Qt+RGB开/关各2048步，完整积分逐位相同；额外对命令、drive、drive_prev、原observer diag逐位相同。窗口正常关闭，无录像错误。视频18秒、1920×1080/25fps，450帧全解码，源RGB各有真实图像变化，实际qpos位移非零，每帧状态哈希/时间/输入步/载荷与日志匹配，曲线不显示未来；不是重新积分或静态图伪装。

视频标题ACTUATOR DYNAMICS AND DISTURBANCE ROBUSTNESS STUDY，比较C4/fourD/固定lead，显示真实XYZ/姿态、六腿行程、六腿命令/drive/actual、载荷、预算和状态。没有PPO Training标签。源路径和完整帧映射在视频.validation.json。

## 25个问题逐项回答

1. **.338143mm来自哪轴？** 同一3.585s主要Y84.18%、X15.82%、Z约.00169%，不是Z误差。
2. **何时发生？** 持续加载晚期，卸载前65ms；严格配对表明大部分无外载也存在，不能归因加载瞬间。
3. **四维权限够吗？** 对选定Home竖向候选够；对原横向主问题没有充分闭环解决证据，必须单独阻塞。
4. **τ0复现？** disabled C0/C3/C4和enabled τ0 C4完整6000步逐位一致，含观测diag/事件/终止。
5. **模型正确吗？** 声明的一阶离散状态与解析ZOH一致，三信号与原mj_step对应；不是已标定真实驱动器。
6. **内部状态泄漏？** 无。原observer记已下发命令，与hidden drive的差异独立验证；特权诊断单列。
7. **安全τ？** 本轮0/2/5/10ms以及所列4～6ms小差异向量均完成，不外推其他范围/组合。
8. **统一lag敏感？** Z阶跃IAE在10ms约增加32.16%，有真实可测退化，但仍满足原完成标准。
9. **差异lag敏感？** 该小范围差异影响有限，未引入明显新Yaw偏置；不能声称任意6τ都鲁棒。
10. **动态跟随退化？** τ0加载峰约.252246、10ms约.258447mm，变化小于原gain导致的.338143；横向主因分开分析。
11. **1/2/5ms平台延迟？** C4加载IAE约增加2.42/4.85/12.03%，均完整完成。
12. **噪声稳定？** 三个已知小量级、seed42合成单因素均完成；全链q/dq理想，非完整传感器或统计可靠性验证。
13. **可恢复重复误差？** 两类冻结Home候选均无保护且未越严格精度带；相同因果规则在两类均有可重复的配对改善，非随机统计证明。
14. **哪些超能力？** 本轮未发现新安全失败；原横向任务四维权限/价值仍阻塞，更大τ/长延迟/混合工况未执行，不作安全推断。
15. **合法补偿方向？** 竖向四维方向已实际有效；X/Y局部直接力反馈可改善，但六维随机箱未验证。
16. **改善强C4吗？** fourD在两类加载IAE降低10.82/17.16%，工程筛选通过；这是经典因果诊断，不是PPO成绩。
17. **代价增加吗？** fourD两个评价的实际总力峰值变化率基本相同、平方积分几乎不变、XY峰降低；完整预算/非目标项均通过，不能推广所有状态。
18. **历史有信息吗？** 有，当前/过去等效估计和命令足以构造有效因果lead；不等于识别每个隐藏参数。
19. **哪些不可可靠辨识？** 6个独立τ、未知外载和测量延迟/模型残差的唯一分离未证实；Home缺激励，数值秩不能单独证明可辨识。
20. **强传统基线剩什么？** 原横向跟随和很小的瞬态/积分误差；新增fourDlead已是强基线，尚无RL额外价值证明。
21. **哪些可作Gym候选？** 仅两类安全Home竖向条件可作为后续提案；当前已看数据是筛选/验证资料，不能再次称未来全新最终测试。
22. **3/4/6维？** 选定竖向候选优先保留四维；原XY问题需要另行权限研究，不直接放开六维。初期可锁定不必要轴，随机联合安全仍需验证。
23. **20还是50Hz？** 先20Hz，因为已有20Hz因果正证据、底层仍1kHz；50Hz尚未做同预算比较，不宣称必优。快τ不等于Actor可直接1ms响应。
24. **吞吐量可接受？** 当前完整日志约309步/s支持有界物理预算估计；不是GPU推断，不含PPO/Reset总成本。
25. **能立即进入训练吗？** 条件阻塞：物理、因果改善已具备，但先确认相对强经典方案的研究目标、观测/随机安全与新独立验证，再由用户审核；本轮不启动。

## 已实际运行的PowerShell入口

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/analyze_phase2c0_dynamic_error.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2c0_actuator_lag.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2c0_observer.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2c0.py --run-root results/runs/phase2c0_20261029_v1
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests -p phase2a_pytest_temp_plugin
```

主研究run_phase2c0_robustness.py和run_phase2c0_action_feasibility.py已真实运行，必须使用新目录复现，不覆盖已有目录。benchmark两轮原始数据保留。回放入口watch_phase2c0.py使用实际记录显示命令/drive差异，独立标记ARCHIVE REPLAY；具体测试命令见下方补充。长期回放去掉自动关闭参数，不把回放称新动力学。

## 仍未执行与阶段暂停

正式Gymnasium、PPO/SAC/BC、网络、随机联合4D/6D动作箱、完整硬件观测、真实驱动器辨识、更大/混合τ-gain-noise-delay、跨多噪声种子与实机安全认证均未执行。没有训练checkpoint。下一阶段若被批准，必须把C4+fourDlead列为强传统对照；若PPO只学会相同规则或无稳定额外收益，应停止扩大训练，而不是追加长预算。

本轮完成后暂停，不自动推进训练。

## 已实际测试的归档回放命令

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/watch_phase2c0.py --trial results/runs/phase2c0_20261029_v1/frozen_evaluation/eval_actuator_lag/C4_fourD --validate-seconds 2
```

该命令实际打开并正常关闭。持续查看时去掉--validate-seconds 2；窗口显示ARCHIVE REPLAY和命令/drive差异，不重新训练或积分。原生真实动态窗口的有界命令为上文validate_phase2c0_observer.py，已实际运行。所有未执行项仍按前文标记。
