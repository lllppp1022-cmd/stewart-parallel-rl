"""Build reports using raw-derived metrics and completed test evidence only."""
import _bootstrap,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha
from build_phase2b2_reports import table

def main():
    root=ROOT/'results/runs/phase2b3_20261029_v1';v=json.loads((ROOT/'results/phase2b3_validation.json').read_text(encoding='utf-8'));f=pd.read_csv(ROOT/'results/phase2b3_controller_comparison.csv');all_=pd.read_csv(ROOT/'results/phase2b3_all_trials.csv');history=pd.read_csv(ROOT/'results/phase2b3_history_observability.csv');hist=json.loads((ROOT/'results/phase2b3_historical_audit.json').read_text(encoding='utf-8'));frozen=json.loads((root/'controllers_frozen.json').read_text(encoding='utf-8'));p=json.loads((ROOT/'configs/phase2b3_protocol.json').read_text(encoding='utf-8'));test=ET.parse(ROOT/'results/phase2b3_pytest_acceptance_v1.xml').getroot().find('testsuite');assert test is not None and test.attrib['failures']=='0' and test.attrib['errors']=='0';tests=int(test.attrib['tests']);decision='PHASE 2C BLOCKED'
    head='# Stewart Phase 2B-3：执行器增益失配、Yaw补偿与任务筛选\n\n日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\\miniconda\\envs\\stewart_rl\\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。\n\n'
    audit=head+'''## 冻结与数据来源

历史Phase0～2B-2共4814文件已冻结，原控制器、模型、实验数据及旧失败记录逐项SHA256核对一致。`results/phase2b3_frozen_history.json`为清单。首先只读分析Phase2B2的名义C3、交替±.5%及已失败±1%日志，没有重跑±1%。隐藏actual gain只用于本文件离线归因，控制器不可访问。

## 六腿增益如何形成Yaw力矩

世界upper_center处采用原支链Jacobian J_l。每腿名义gain为1，实际力为native clip(g_i F_cmd_i)。力偏差ΔF=F_actual−F_cmd，瞬时六维广义力偏差ΔW=A(q).T ΔF，其中A是独立约束切空间的dl/d世界平台速度映射，与原J_l在同一物理定义下核对。这里保留真实状态下的映射，不把笛卡尔弦长当作实际移动副长度。

六腿的Yaw力臂符号交替。原正支撑力约80～90N，交替±.5%使实际力差约±.4N；gain符号与力臂符号相乘后，六腿Yaw贡献朝同一方向叠加，而不是互相抵消。原CSV新增逐腿yaw_lever_arm_j与yaw_moment_contribution_j，可逐项求和得到下面的ΔMz。对其他轴也完整保留六维ΔW，不把所有运动都归给Yaw。

'''+table(pd.DataFrame(hist['cases']),['case','preload_gain_Mz_mean_Nm','preload_observer_Mz_mean_Nm','local_unexplained_Mz_RMS_Nm','preload_yaw_rate_peak_rad_s','failed','first_failure_s'])
    audit+='''
名义无失配ΔMz=0，观测偏置约-4.76×10⁻⁷Nm。±.5%时真实增益力矩约-.234104Nm、原C3内部估计约-.234345Nm；±1%保护前约-.468238/-.471408Nm。加倍失配近似加倍Yaw力矩与早期角速度，而不是随机颜色或可视化变化。角速度方向为负，Yaw随之偏负，支撑了“力偏差→Yaw力矩→角速度→姿态→保护”的因果链。

原C3确实有完整六维估计器！它主动补偿仅选择[2,3,4]三个分量，未把Mz用于控制；不能说Yaw未进入观测器。C3的等效估计还包含建模残差，并非精确外力传感器。局部方程M0 vdot=Wcmd−W0−K0x−D0v+d_eq，离线用速度增量和名义矩阵复算。±.5%剩余Mz局部平衡RMS约5.55×10⁻⁵Nm，±1%保护前约1.31×10⁻⁴Nm；约束柔顺、变量质量/几何与数值微分仍存在，未宣称全部精确解释。此处没有把constraint反力读入控制器。

## 历史失败时间线

±1%：.200s约Yaw=-.085917°、世界ωz=-.011872rad/s；.384s达到Yaw=-.200308935°，触发原±.2°保护。发生在2s外载前，因此是支撑状态下固定增益失配问题。原0.2s受限保持到.584s，Yaw继续到约-.395282473°：Hold是受限关节保持而非瞬间消除增益偏差/动量。不能称“已恢复”。

保护后原控制器reset，日志中的观测器零值不表示真实扰动消失。所有偏置统计只用保护前有效段。load/unload未发生，指标右删失/N/A，不填0、不用短RMSE与6秒完成轨迹排序。原±1%失败文件未删、未移、未改，也未重新执行。

图：`phase2b3_gain_force_wrench.png`、`phase2b3_yaw_failure_chain.png`；数据`phase2b3_gain_yaw_mapping.csv`，离线脚本`analyze_phase2b3_yaw_failure.py`不调用mj_step。
'''
    (ROOT/'docs/23_Phase2B3增益失配与Yaw失稳分析.md').write_text(audit,encoding='utf-8')
    design=head+'''## 四维接口及零动作

输入顺序[Fz(N),Mx(Nm),My(Nm),Mz(Nm)]，世界坐标、upper_center力矩参考点；W=[0,0,Fz,Mx,My,Mz]，解原J_l.T F=W。不新建另一套控制Jacobian。`task_residual_4d.py`把保持的任务请求在每1ms用原当前J映射，再交给**冻结的**ResidualController和SharedBudgetGovernor。内层PD/Observer为1kHz，因果残差20Hz；适配器只负责转换，不复制C0/C3。

四维全零分别在原force10任务完整6000步对比旧C0/C3：完整458维mjSTATE_INTEGRATION、qpos/qvel、actuator状态、12ctrl、实际力、原C3全部diag观测状态逐位一致。保护事件、终止原因相同，不通过放宽阈值获得等价。

'''+table(pd.read_csv(ROOT/'results/phase2b3_zero_equivalence.csv'),['base','complete_state_bitwise_equal','maximum_difference','signals_and_observer_equal','protection_and_termination_equal'])
    design+='''
## 共享预算与已知控制命令

F_add_total=F_cmd−F_C0，包含原C3全部补偿和新增Yaw/四维请求。每腿总附加±5N、每1ms变化≤.02N即20N/s；还与原±400N命令余量取交集。不是给C4再加一套±5N预算。原生实际力限制±400N不变。预算日志只在budget_applicable有效段解释；保护保持有独立语义。

ResidualController在最终命令与原C3输出不同时，用原remember_applied_command更新下一步**已下发合成命令**记忆；不额外调用observer.update，不改原C3 previous_compensation更新律。它知道下发命令，不知道actual gain。误差仍通过反馈被因果估计。零动作不增加观测更新或随机数，不额外积分。

F_residual_requested、F_residual_applied、F_add_total分别为请求、相对原C3的施加量、相对C0的总附加量。W_requested为名义目标，W_realized为名义Jacobian下的投影结果；W_physical_residual是**仅供离线**的实际增益/真实力映射反事实增量，不冒充请求精确实现。Yaw预算占用会与C3的Z/R/P补偿竞争。

## 小幅Yaw预检

先验证历史成功的±.5%无外载Home；随后名义Home按±.005、±.025、±.1、±.25Nm逐级预检，各施加.8～1.8s，提前声明的时钟不读取真实外载。8个回合全完成，正负Yaw实际响应符号正确。每次仍6s完整运行，记录撤销后的恢复及非目标轴变化。

'''+table(all_[all_.stage=='action_preflight'],['variant','success','yaw_peak_deg','yaw_rate_peak_rad_s','max_xy_error_mm','roll_error_peak_deg','pitch_error_peak_deg','shared_add_peak_N','shared_add_rate_N_s','max_weld_translation_m','max_weld_angle_rad'])
    design+='''
这是单独Yaw方向测试，不是整个4维动作箱的安全证明。控制器Mz请求上限.30Nm，在有限闭环开发/评价中验证；不能把它推广到任意组合持续输入。无失配同时任意Fz/Mx/My/Mz的随机组合未扫描。

六维接口仅使用Phase2B2既有真实单腿±.1N与Leg1+.5N证据作为物理耦合参考，不重复自由6维大扫描。Leg1+.5N额外Y约.482mm，任务Yaw映射则把请求直接限定到Mz；两者量纲/权限不同，旧单腿底座为C0、本轮Yaw探针底座为C3，也不是同基线的受控性能对照，不能仅凭大小宣称四维PPO优于六维PPO。四维减少无意义瞬时Fx/Fy请求，但真实闭环仍有XY耦合，且缺少直接Fx/Fy权限。

## 冻结安全与初态

原1ms积分/1kHz传统控制/100Hz控制Jacobian不变；条件数≤50；weld平移/转动1e-4m/rad；Yaw±.2°、Roll/Pitch±.5°、相对Home XY±2mm/Z±5mm；代理间隙5mm、20ms查询；原行程与.2s保持逻辑保留。碰撞仍为PARTIAL PROXY QUERY。所有真实运动通过原data.ctrl和mj_step完成，没有动态写live qpos。

增益只在每次创建actual模型副本、首次积分前配置，nominal控制模型保持原样；无质量、COM、噪声或延迟混入。本轮固定未知gain，运行中只切换外载，不突然切换gain。原前三维/ZRP动作代码及MJCF均未修改。

## 参考适配器的真实问题

新参考为已允许periodic的0.85时间缩放，q/p/R取原参考.85t，dq与twist乘.85，姿态空间和行程未扩大。首个最终动态工况在首次积分前因适配器没有duration属性被调度器拒绝；0步异常和空日志保留。修复duration=原duration/.85、添加回归测试后，仅在独立continuation_v2目录执行缺失工况。最终6秒预算不变，未改控制参数或障碍。不能把首次0步异常报成动力学成功。
'''
    (ROOT/'docs/24_Phase2B3四维任务空间补偿设计.md').write_text(design,encoding='utf-8')
    study=head+'''## 强传统控制器

**C4-PD**：冻结C3+受限Yaw反馈。世界SO(3)误差e=log(R_actual R_ref.T)，Mz=clip(−Kp e_z−Kd(ω_actual−ω_ref)_z,±.30Nm)。不使用欧拉角差构造控制，不读取gain、外载或未来状态。开发仅Kp=100/300Nm/rad，Kd=2√(名义Mzz Kp)；名义Mzz=4.584301714kgm²，对应Kd约42.821/74.170Nms/rad。这个公式只是附加阻尼的有限候选依据，原C0也含反馈，不能声称整体系统已精确临界阻尼。实际选Kp300/Kd74.169819。

**C4-Observer**：冻结C3+clip(−g·dhat_Mz,±.30Nm)。候选g=.5/1，最终选1。读取原C3当前因果六维局部估计，没有将真实gain或实际plant广义力当估计。已知输入记忆由共享包装层正确更新，避免把自身Yaw补偿误认作外扰。

12次开发：四候选×名义Home、交替±.5%Home、交替±.5%periodic（均无外载）。只在完整成功候选中按平均Yaw显示误差IAE选择；显示角度用于局部评分，控制一直SO(3)。两个家族分别选优，再选开发得分更低的Observer作为最强C4。所有参数在独立最终工况运行前冻结，没有根据最终误差修改。

**C3_4d20**：选定PD的Yaw反馈20Hz，加原Phase2B2可解释Z/R/P估计差分lead（.02s系数，±[.5N,.025Nm,.025Nm]）。

**C4_4d20**：最强C4仍1kHz，再加20Hz四维估计差分lead；额外Yaw lead±.015Nm，合成Yaw请求仍±.30Nm，总支链预算仍共享。两者是手工经典因果诊断，不是神经网络、学习策略或PPO成绩。

## 新独立测试与统计边界

三个新gain组合在protocol中先冻结：reverse=1+[+.004,−.004,+.004,−.004,+.004,−.004]；uneven=1+[−.004,+.002,−.001,+.004,−.003,+.001]；skew=1+[+.001,−.003,+.004,−.002,+.002,−.004]。先C3无外载Home安全预检，通过才进入对应外载测试。预检不调参。

六任务：reverse_home、uneven_home、reverse_z（−6N，1.70～3.45s）、uneven_roll（+.3Nm，1.45～3.75s）、skew_pitch（−.3Nm，1.90～4.05s）、uneven_dynamic（periodic×.85与−4N/+.15Nm组合外载，1.60～3.65s）。各方法使用相同实际gain、参考、外载与初始状态。是固定可重复的确定性评价，无随机种子筛选，不能称统计显著泛化或独立随机场景成功率。

30回合全部完成。下表为六个冻结工况的等权描述均值，不跨历史测试集合并。

'''
    cols=['yaw_peak_deg','yaw_IAE_deg_s','position_rmse_mm','orientation_rmse_deg','max_xy_error_mm','max_force_rate_N_s','shared_add_peak_N','shared_add_rate_N_s','force_square_integral_N2_s'];avg=f.groupby('variant',as_index=False)[cols].mean();study+=table(avg,['variant']+cols)
    study+='\n## 每个工况与恢复状态\n\n'+table(f,['task','variant','preload_yaw_bias_deg','yaw_peak_deg','load_peak_position_mm','load_peak_orientation_deg','load_recovery_status','unload_recovery_status','success'])
    study+='''
原恢复标准是位置≤.25mm且SO3≤.02deg连续.3s，并非原任务成功标准（最后.5s位置<.75mm、SO3<.1deg且无保护）。本轮全部保留，不能因任务成功就宣称严格精度处处满足。无外载Home中的load/unload只是固定时段窗口，不代表真的加了外载。另列Yaw专项SO(3)世界Z分量误差、±.02deg、.3s保持；安全门槛仍±.2deg。未越带写NO EXCURSION/N/A，不填恢复时间0。

## 强传统控制已解决多少

平均Yaw峰值C3约.072464°、C4-PD .017787°、C4-Observer .006965°；平均Yaw IAE约.304304/.102979/.004136deg·s。Observer优势符合历史中已经准确估计到Mz的机制，不需要猜actual gain。C4-Observer六个工况的Yaw峰值都在.02°专项带内。

C4没有完美解决所有轴：skew_pitch加载位置峰值C3约.147960、Observer .149754mm，略有代价；uneven_dynamic约.338143mm，严格位置.25mm仍有越带，卸载后恢复。这些是保留的XYZRP/XY耦合问题，不能把Yaw改善包装成所有自由度最优。

平均实际总力峰值变化率C3约34.276、Observer37.569N/s，约增加9.6%；共享附加预算仍≤20N/s，总力变化率包含原C0，不混为一项。力平方积分约271000N²s，家族间差值很小，包含重力支撑，不单凭这个接近数声称节能。按每工况而非只看均值评价取舍。

额外lead把Observer平均Yaw峰值约.006965→.005268°，绝对约.001697°；平均IAE .004136→.003084deg·s。百分比看似明显，但经典C4已满足Yaw精度目标；平均位置RMSE仅约.098903→.098845mm，差约.000058mm。共享附加变化率更常用到20N/s。对真实硬件噪声尚无验证，不据此宣称值得承担PPO成本或有统计显著优势。

## 历史观测携带的信息

'''+table(history[(history.stage=='frozen_evaluation')&(history.variant=='C4-Observer')],['task','gain_Mz_mean_Nm','observer_Mz_mean_Nm','observer_gain_Mz_MAE_Nm','estimate_first_63percent_s','yaw_excitation_effective_rank','yaw_excitation_condition'])
    study+='''
加载前约35ms达到实际偏置63%量级，与30rad/s观测带宽相符；这里用离线真值确定目标阈值，只是诊断，不向在线控制器输入。正反gain方向都可从名义模型、过去已下发力、速度/姿态历史估计**聚合Yaw偏置**。这是可因果补偿的证据，不证明能可靠识别六个单独gain。

Home局部Yaw投影激励矩阵条件数约10⁶～10⁸，按相对奇异值1e-4的有效秩仅2～3；说明当前静态/小幅轨迹对六未知量的辨识激励有限。该矩阵使用实际几何作离线诊断，未作为策略观测。相关性不能代替闭环控制验收，也不能据此承诺历史神经网络可辨识全部参数。

## 特权物理诊断（单独，不参与方法排名）

'''+table(all_[all_.stage=='oracle_diagnostic'],['task','variant','privileged','yaw_peak_deg','yaw_IAE_deg_s','load_peak_position_mm','max_force_rate_N_s','shared_add_peak_N'])
    study+='''
Oracle使用actual gain、当前实际切空间映射及真实当前外载，解实际力实现名义C0广义力减外载，1kHz且仍共享预算。严格标签PRIVILEGED ORACLE DIAGNOSTIC；不能叫自主策略、最优理论上界或PPO可达成绩。它表明还有物理抵消空间，不能证明额外收益可从非特权信息及时学出；原C4已经利用可获得估计有效解决主要Yaw问题。

## 研究门禁

Gate1通过：小幅Yaw可控、±.5%可完整保持，原安全与预算未变。Gate2通过：C4公平有限整定并在未见gain/外载对照，充分报告控制代价及非目标轴。Gate3不通过：没有建立在强C4之上、收益足够且值得PPO的剩余任务。

因此当前任务 **PHASE 2C BLOCKED**，理由是额外学习价值不足，不是四维动作无效。传统Yaw观测补偿已解决主要问题；剩余动态XY/ZRP任务还需独立可控性、强基线和信息边界定义。不能为了RL人为削弱C4、扩大预算、放宽保护或重跑已知±1%失败。结论仅限本轮范围，不排除其他任务的RL价值。
'''
    (ROOT/'docs/25_Phase2B3经典Yaw补偿与可学习性.md').write_text(study,encoding='utf-8')
    final=head+f'''## 正式结论

**{decision}**。四维Yaw通道有效，C4-Observer已用允许的因果信息解决主要增益Yaw偏置。额外因果残差仍有小数值改善，但没有证明值得在最强传统基线上增加PPO训练。完成本轮后暂停，不创建Gymnasium或训练任何网络。

## 实际运行清单与成本

'''+table(all_.groupby('stage',as_index=False).agg(trials=('success','size'),successes=('success','sum'),physics_steps=('physical_steps','sum')),['stage','trials','successes','physics_steps'])+f'''
有效动力学共{v['trials']}回合，{v['physical_steps']}个1ms积分步（348仿真秒），全部按原任务完成。另1次工程尝试在0步失败，共59次尝试，未超过预先66回合最大研究预算。0步异常不混入动力学成功率分母，也没有删除。原±1%安全失败只做离线分析，无重新执行。

主目录：`results/runs/phase2b3_20261029_v1`；缺失工况续跑：`results/runs/phase2b3_20261029_continuation_v2`；索引`phase2b3_run_index.json`。两者是同一冻结矩阵，不是两轮调参或择优重跑。每个有效回合6000步，保存完整CSV.gz、NPZ、配置、plant、action_profile、原metrics及独立metrics_recomputed、事件、summary。没有神经网络checkpoint，因为没有训练。

额外观察器验收：独立off/on各2048步（共4096），原生MuJoCo显示镜像+Qt+现场RGB开启/关闭的完整458维integration逐位一致，最大差0、无中断、正常退出。自动测试里的短动力学归测试成本，不混作科学实验。

## 全量测试与独立复核

**{tests}项测试全部通过**，包含原189项，新增17项单元与6项实际证据测试。原代码/模型/全部4814历史文件SHA256不变。保留原pandas弃用警告2条。第一轮新增单测15过2失败（错误假设世界Z外载的局部Fx/Fy必须零、错误字段dt），修正新测试后17通过，未改控制实现或旧测试来通过；首轮失败日志保留。后发现参考调度接口duration缺失，修复并回归验证，0步异常完整保留。

`phase2b3_validation.json`复核58回合：原始CSV与NPZ完整qpos/qvel/time逐位一致，状态有限；actual force=gain×cmd（按原native clip）；共享±5N/20N/s和原±400N通过；后六ctrl只来自外载，未混入残差；无额外噪声、延迟、质量、COM；独立scratch重算预积分映射最大差{v['max_scratch_mapping_error']:.3g}；全部物理回合无MuJoCo警告。安全指标按回合详见完整CSV，不以自动测试替代物理实验。

## 关键结果

'''+table(avg,['variant','yaw_peak_deg','yaw_IAE_deg_s','position_rmse_mm','max_force_rate_N_s','shared_add_peak_N','shared_add_rate_N_s'])+'''
上表是6个确定性冻结工况的描述平均，非统计显著结论。全部正式方法各6/6完成且无保护；这不等于全程都在严格.25mm/.02deg精度带，也不等于全机构碰撞或实机认证。

## 20个研究问题

1. **增益为何Yaw偏移？** 正支撑力乘交替gain误差，乘交替Yaw力臂后同号叠加；±.5%约-.234Nm，角速度/姿态方向与记录一致。
2. **原C3估计准确吗？** 局部小幅范围明显响应，±.5%均值约-.234345Nm，与离线真值近似；非精确传感器，仍有建模残差。
3. **四维能控制Yaw？** 八个正负探针真实积分方向正确，完整零动作等价，非可视化假动作。
4. **非目标耦合多少？** 逐回合CSV有XY/Z/R/P峰值；C4平均XY约.150mm，单个动态工况约.338mm，不能声称无耦合。
5. **PD还是Observer？** 本模型理想观测下Observer更好；开发与最终一致。未做新噪声/延迟下C4鲁棒性，不能外推硬件。
6. **共享预算保持？** 是，C3+Yaw+额外lead合计每腿±5N、20N/s，未加第二预算。
7. **±.5%能改善？** 开发无外载Home/periodic中C4明显降低Yaw；最终不同组合亦有效。
8. **±1%能解释？** 离线真实力矩约加倍，负Yaw越界，观测器有响应但未被C3主动补偿；局部方程留有残差，不做全局证明。
9. **需立即重跑±1%？** 不需要、未重跑；当前安全权限和任务选择不要求把已失败案例变成探索环境。
10. **加载前已出现？** 是，历史失败.384s早于2s外载；新无载保持和前载指标单列。
11. **加外载仍有效？** 独立Z/Roll/Pitch/动态组合均完成，Yaw压低；各轴代价详表，没有筛掉退化项。
12. **其他轴是否更差？** 部分工况有微小位置退化，例如skew_pitch，已报告。C4未承诺所有轴最优。
13. **控制代价取舍？** Observer平均实际总力峰值变化率比C3约增9.6%；共享限速仍合法；力平方积分近似，不能包装节能。
14. **哪些特权？** Oracle读真实gain/当前外载/实际映射，仅诊断；C3、PD、Observer、两个20Hz补偿均无真值/未来输入。
15. **经典Yaw能解决主要问题？** 是，当前理想反馈及小幅未知gain范围，Observer六个最终工况Yaw峰值都在.02°带内。
16. **残差超过C4的价值？** 有小数值改善，但未形成足够额外收益；平均位置RMSE改善约.000058mm，不能直接据此启动PPO。
17. **四维优于六维？** 对任务结构更直接、避免任意Fx/Fy请求；未训练、未做等权限广泛扫描，不宣称普遍优越。
18. **哪些历史有用？** 过去已下发命令、速度增量、SO3误差及因果动量估计可判断聚合Mz偏置；Yaw激励不足以稳定辨识所有6个gain。
19. **未来哪些工况适合？** 若重新立项，已安全的±.25/±.5%开发、最终≤.4%组合可作候选；但需先定义相对强C4确有价值的新目标，不能直接当获准训练集。
20. **可进入Phase2C吗？** 当前BLOCKED，主要因学习价值证据不足；不自动进入Gym/PPO。

## 实际视频与窗口验收

`assets/videos/stewart_phase2b3_yaw_compensation.mp4`：18秒1920×1080/25fps，450帧全部解码、450不同帧；3段C3/C4-PD/C4-Observer均来自reverse_z**现场live RGB**。每帧qpos哈希、时间、步数与CSV匹配，无未来曲线、无重新积分。另验证实际记录帧的支链位移非零。标题ACTUATOR MISMATCH AND YAW COMPENSATION STUDY，显示gain类型（仅标签）、Yaw/目标、Z/R/P、六腿行程/实际力、载荷、共享预算、保护/运行状态。没有PPO标签，也没有把毫米级动作夸大。

观看器是历史归档回放，明确ARCHIVE REPLAY；不把回放当新动力学。真实原生显示使用独立状态镜像，由实际owner发布，无GUI线程写live状态。GUI采集开/关完整积分相同，已真实Windows运行。

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts/watch_phase2b3.py --trial results/runs/phase2b3_20261029_v1/frozen_evaluation/reverse_z/C4-Observer --validate-seconds 2
& D:\miniconda\envs\stewart_rl\python.exe scripts/validate_phase2b3_observer.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/analyze_phase2b3_yaw_failure.py
& D:\miniconda\envs\stewart_rl\python.exe scripts/compare_phase2b3_controllers.py --run-root results/runs/phase2b3_20261029_v1 --continuation-root results/runs/phase2b3_20261029_continuation_v2
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests -p phase2a_pytest_temp_plugin
```

上述入口及参数均已实际运行（测试运行带XML输出）。持续查看回放可去掉--validate-seconds 2，关闭窗口结束；本轮只对2秒自动关闭做了有界GUI验收，不虚称进行了无界时长验收。无需为看视频重跑实验。

重现实验的主入口是run_phase2b3_yaw_compensation.py（已运行），必须指定全新run-root，已有目录拒绝覆盖。首次原运行、修复后的续跑入口及全部日志保留。代码已修复参考适配，未来全新运行可直接完成冻结矩阵；本轮未为了证明命令再重复58回合。单独validate_phase2b3_yaw_actions.py的零动作函数已通过主流程执行，独立CLI未另跑，不列作额外实际运行成本。

## 未执行、限制与暂停

未执行：±1%失败重跑；任意四维/六维动作组合安全穷举；C4新噪声/延迟或更大失配/组合质量测试；实机、硬实时调度/传感器标定；全机构碰撞/形式化安全认证；Gym/PPO/SAC/BC。没有这些结果，更无训练checkpoint。原观察器是Home局部模型，不能外推大姿态。

本阶段交付源码、原始物理数据、图、18秒视频、全部失败/退化和报告。按用户要求暂停，等待审核。
'''
    (ROOT/'docs/26_Phase2B3验收报告.md').write_text(final,encoding='utf-8')
    write_json(ROOT/'results/phase2b3_decision.json',dict(decision=decision,gate1='PASS: local safe yaw authority',gate2='PASS: strong frozen classical comparison',gate3='NOT MET: meaningful extra learning benefit beyond C4 not demonstrated',tests_passed=tests,rl_training=False,physical_trials=58,engineering_attempts=1,physics_steps=348000))
    print('Reports generated, decision',decision,'tests',tests)

if __name__=='__main__':main()
