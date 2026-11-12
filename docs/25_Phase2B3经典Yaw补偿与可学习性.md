# Stewart Phase 2B-3：执行器增益失配、Yaw补偿与任务筛选

日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\miniconda\envs\stewart_rl\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。

## 强传统控制器

**C4-PD**：冻结C3+受限Yaw反馈。世界SO(3)误差e=log(R_actual R_ref.T)，Mz=clip(−Kp e_z−Kd(ω_actual−ω_ref)_z,±.30Nm)。不使用欧拉角差构造控制，不读取gain、外载或未来状态。开发仅Kp=100/300Nm/rad，Kd=2√(名义Mzz Kp)；名义Mzz=4.584301714kgm²，对应Kd约42.821/74.170Nms/rad。这个公式只是附加阻尼的有限候选依据，原C0也含反馈，不能声称整体系统已精确临界阻尼。实际选Kp300/Kd74.169819。

**C4-Observer**：冻结C3+clip(−g·dhat_Mz,±.30Nm)。候选g=.5/1，最终选1。读取原C3当前因果六维局部估计，没有将真实gain或实际plant广义力当估计。已知输入记忆由共享包装层正确更新，避免把自身Yaw补偿误认作外扰。

12次开发：四候选×名义Home、交替±.5%Home、交替±.5%periodic（均无外载）。只在完整成功候选中按平均Yaw显示误差IAE选择；显示角度用于局部评分，控制一直SO(3)。两个家族分别选优，再选开发得分更低的Observer作为最强C4。所有参数在独立最终工况运行前冻结，没有根据最终误差修改。

**C3_4d20**：选定PD的Yaw反馈20Hz，加原Phase2B2可解释Z/R/P估计差分lead（.02s系数，±[.5N,.025Nm,.025Nm]）。

**C4_4d20**：最强C4仍1kHz，再加20Hz四维估计差分lead；额外Yaw lead±.015Nm，合成Yaw请求仍±.30Nm，总支链预算仍共享。两者是手工经典因果诊断，不是神经网络、学习策略或PPO成绩。

## 新独立测试与统计边界

三个新gain组合在protocol中先冻结：reverse=1+[+.004,−.004,+.004,−.004,+.004,−.004]；uneven=1+[−.004,+.002,−.001,+.004,−.003,+.001]；skew=1+[+.001,−.003,+.004,−.002,+.002,−.004]。先C3无外载Home安全预检，通过才进入对应外载测试。预检不调参。

六任务：reverse_home、uneven_home、reverse_z（−6N，1.70～3.45s）、uneven_roll（+.3Nm，1.45～3.75s）、skew_pitch（−.3Nm，1.90～4.05s）、uneven_dynamic（periodic×.85与−4N/+.15Nm组合外载，1.60～3.65s）。各方法使用相同实际gain、参考、外载与初始状态。是固定可重复的确定性评价，无随机种子筛选，不能称统计显著泛化或独立随机场景成功率。

30回合全部完成。下表为六个冻结工况的等权描述均值，不跨历史测试集合并。

| variant | yaw_peak_deg | yaw_IAE_deg_s | position_rmse_mm | orientation_rmse_deg | max_xy_error_mm | max_force_rate_N_s | shared_add_peak_N | shared_add_rate_N_s | force_square_integral_N2_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | 0.072464 | 0.304304 | 0.099751 | 0.051715 | 0.151762 | 34.276039 | 0.462415 | 16.139786 | 271000.243472 |
| C3_4d20 | 0.017787 | 0.103389 | 0.099053 | 0.017753 | 0.150342 | 34.259655 | 0.695178 | 20.000000 | 270999.357287 |
| C4-Observer | 0.006965 | 0.004136 | 0.098903 | 0.003127 | 0.149833 | 37.568773 | 0.723465 | 19.232843 | 270999.210701 |
| C4-PD | 0.017787 | 0.102979 | 0.099116 | 0.017716 | 0.150294 | 36.887400 | 0.651467 | 18.562608 | 270999.353986 |
| C4_4d20 | 0.005268 | 0.003084 | 0.098845 | 0.002849 | 0.149892 | 37.565361 | 0.760772 | 20.000000 | 270999.197677 |

## 每个工况与恢复状态

| task | variant | preload_yaw_bias_deg | yaw_peak_deg | load_peak_position_mm | load_peak_orientation_deg | load_recovery_status | unload_recovery_status | success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reverse_home | C3 | 0.067291 | 0.095923 | 0.015405 | 0.072683 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_home | C3_4d20 | 0.023450 | 0.023451 | 0.012419 | 0.023451 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_home | C4-Observer | 0.000978 | 0.009228 | 0.011245 | 0.000606 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| reverse_home | C4-PD | 0.023451 | 0.023451 | 0.012406 | 0.023451 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_home | C4_4d20 | 0.000740 | 0.007289 | 0.011257 | 0.000486 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| reverse_z | C3 | 0.067291 | 0.095923 | 0.067896 | 0.072683 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_z | C3_4d20 | 0.023450 | 0.023732 | 0.065076 | 0.023732 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_z | C4-Observer | 0.000978 | 0.009228 | 0.066862 | 0.000434 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| reverse_z | C4-PD | 0.023451 | 0.023733 | 0.067197 | 0.023733 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| reverse_z | C4_4d20 | 0.000740 | 0.007289 | 0.064773 | 0.000374 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| skew_pitch | C3 | 0.047647 | 0.064326 | 0.147960 | 0.046856 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| skew_pitch | C3_4d20 | 0.015713 | 0.015744 | 0.149874 | 0.016178 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| skew_pitch | C4-Observer | 0.000194 | 0.006211 | 0.149754 | 0.004087 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| skew_pitch | C4-PD | 0.015713 | 0.015742 | 0.149158 | 0.016206 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| skew_pitch | C4_4d20 | 0.000066 | 0.004586 | 0.150481 | 0.003967 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_home | C3 | -0.041768 | 0.059537 | 0.124258 | 0.045160 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| uneven_home | C3_4d20 | -0.014557 | 0.014558 | 0.124254 | 0.014668 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_home | C4-Observer | -0.000619 | 0.005708 | 0.124126 | 0.001737 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_home | C4-PD | -0.014558 | 0.014558 | 0.124150 | 0.014668 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_home | C4_4d20 | -0.000429 | 0.004149 | 0.124236 | 0.001739 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_roll | C3 | -0.037637 | 0.059537 | 0.148631 | 0.045182 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| uneven_roll | C3_4d20 | -0.014554 | 0.014559 | 0.147468 | 0.014747 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_roll | C4-Observer | -0.000069 | 0.005708 | 0.148415 | 0.002512 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_roll | C4-PD | -0.014557 | 0.014560 | 0.148463 | 0.014760 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_roll | C4_4d20 | -0.000170 | 0.004149 | 0.147261 | 0.002384 | NO EXCURSION / WITHIN TOLERANCE | NO EXCURSION / WITHIN TOLERANCE | True |
| uneven_dynamic | C3 | -0.039848 | 0.059537 | 0.339423 | 0.045227 | NOT RECOVERED DURING LOAD | NOT RECOVERED | True |
| uneven_dynamic | C3_4d20 | -0.014557 | 0.014680 | 0.337364 | 0.021191 | NOT RECOVERED DURING LOAD | RECOVERED | True |
| uneven_dynamic | C4-Observer | -0.000583 | 0.005708 | 0.338143 | 0.015295 | NOT RECOVERED DURING LOAD | RECOVERED | True |
| uneven_dynamic | C4-PD | -0.014559 | 0.014679 | 0.338572 | 0.021202 | NOT RECOVERED DURING LOAD | RECOVERED | True |
| uneven_dynamic | C4_4d20 | -0.000455 | 0.004149 | 0.336915 | 0.015277 | NOT RECOVERED DURING LOAD | RECOVERED | True |

原恢复标准是位置≤.25mm且SO3≤.02deg连续.3s，并非原任务成功标准（最后.5s位置<.75mm、SO3<.1deg且无保护）。本轮全部保留，不能因任务成功就宣称严格精度处处满足。无外载Home中的load/unload只是固定时段窗口，不代表真的加了外载。另列Yaw专项SO(3)世界Z分量误差、±.02deg、.3s保持；安全门槛仍±.2deg。未越带写NO EXCURSION/N/A，不填恢复时间0。

## 强传统控制已解决多少

平均Yaw峰值C3约.072464°、C4-PD .017787°、C4-Observer .006965°；平均Yaw IAE约.304304/.102979/.004136deg·s。Observer优势符合历史中已经准确估计到Mz的机制，不需要猜actual gain。C4-Observer六个工况的Yaw峰值都在.02°专项带内。

C4没有完美解决所有轴：skew_pitch加载位置峰值C3约.147960、Observer .149754mm，略有代价；uneven_dynamic约.338143mm，严格位置.25mm仍有越带，卸载后恢复。这些是保留的XYZRP/XY耦合问题，不能把Yaw改善包装成所有自由度最优。

平均实际总力峰值变化率C3约34.276、Observer37.569N/s，约增加9.6%；共享附加预算仍≤20N/s，总力变化率包含原C0，不混为一项。力平方积分约271000N²s，家族间差值很小，包含重力支撑，不单凭这个接近数声称节能。按每工况而非只看均值评价取舍。

额外lead把Observer平均Yaw峰值约.006965→.005268°，绝对约.001697°；平均IAE .004136→.003084deg·s。百分比看似明显，但经典C4已满足Yaw精度目标；平均位置RMSE仅约.098903→.098845mm，差约.000058mm。共享附加变化率更常用到20N/s。对真实硬件噪声尚无验证，不据此宣称值得承担PPO成本或有统计显著优势。

## 历史观测携带的信息

| task | gain_Mz_mean_Nm | observer_Mz_mean_Nm | observer_gain_Mz_MAE_Nm | estimate_first_63percent_s | yaw_excitation_effective_rank | yaw_excitation_condition |
| --- | --- | --- | --- | --- | --- | --- |
| reverse_home | 0.187279 | 0.187210 | 0.000112 | 0.035000 | 2 | 77124860.268756 |
| reverse_z | 0.187279 | 0.187210 | 0.000112 | 0.035000 | 2 | 77124860.268756 |
| skew_pitch | 0.125620 | 0.125577 | 0.000069 | 0.035000 | 2 | 10476162.300158 |
| uneven_home | -0.116210 | -0.116168 | 0.000069 | 0.035000 | 2 | 4841541.433492 |
| uneven_roll | -0.116210 | -0.116164 | 0.000079 | 0.035000 | 2 | 4660952.552809 |
| uneven_dynamic | -0.116211 | -0.116167 | 0.000073 | 0.035000 | 3 | 4070939.547247 |

加载前约35ms达到实际偏置63%量级，与30rad/s观测带宽相符；这里用离线真值确定目标阈值，只是诊断，不向在线控制器输入。正反gain方向都可从名义模型、过去已下发力、速度/姿态历史估计**聚合Yaw偏置**。这是可因果补偿的证据，不证明能可靠识别六个单独gain。

Home局部Yaw投影激励矩阵条件数约10⁶～10⁸，按相对奇异值1e-4的有效秩仅2～3；说明当前静态/小幅轨迹对六未知量的辨识激励有限。该矩阵使用实际几何作离线诊断，未作为策略观测。相关性不能代替闭环控制验收，也不能据此承诺历史神经网络可辨识全部参数。

## 特权物理诊断（单独，不参与方法排名）

| task | variant | privileged | yaw_peak_deg | yaw_IAE_deg_s | load_peak_position_mm | max_force_rate_N_s | shared_add_peak_N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reverse_home | oracle | True | 0.002233 | 0.001321 | 0.000306 | 38.030795 | 0.358679 |
| reverse_z | oracle | True | 0.002233 | 0.001383 | 0.056481 | 42.938069 | 1.433241 |

Oracle使用actual gain、当前实际切空间映射及真实当前外载，解实际力实现名义C0广义力减外载，1kHz且仍共享预算。严格标签PRIVILEGED ORACLE DIAGNOSTIC；不能叫自主策略、最优理论上界或PPO可达成绩。它表明还有物理抵消空间，不能证明额外收益可从非特权信息及时学出；原C4已经利用可获得估计有效解决主要Yaw问题。

## 研究门禁

Gate1通过：小幅Yaw可控、±.5%可完整保持，原安全与预算未变。Gate2通过：C4公平有限整定并在未见gain/外载对照，充分报告控制代价及非目标轴。Gate3不通过：没有建立在强C4之上、收益足够且值得PPO的剩余任务。

因此当前任务 **PHASE 2C BLOCKED**，理由是额外学习价值不足，不是四维动作无效。传统Yaw观测补偿已解决主要问题；剩余动态XY/ZRP任务还需独立可控性、强基线和信息边界定义。不能为了RL人为削弱C4、扩大预算、放宽保护或重跑已知±1%失败。结论仅限本轮范围，不排除其他任务的RL价值。
