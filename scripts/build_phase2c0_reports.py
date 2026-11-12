"""Chinese feasibility reports from raw-derived artifacts; no simulation/training."""
import _bootstrap,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd
from src.model import ROOT,write_json
from build_phase2b2_reports import table

def main():
    root=ROOT/'results/runs/phase2c0_20261029_v1';f=pd.read_csv(ROOT/'results/phase2c0_all_trials.csv');decomp=pd.read_csv(ROOT/'results/phase2c0_error_decomposition.csv');pairs=pd.read_csv(ROOT/'results/phase2c0_paired_decomposition.csv');gate=pd.read_csv(ROOT/'results/phase2c0_benefit_gate.csv');v=json.loads((ROOT/'results/phase2c0_validation.json').read_text(encoding='utf-8'));bench=json.loads((ROOT/'results/phase2c0_simulation_benchmark.json').read_text(encoding='utf-8'));plan=json.loads((ROOT/'configs/phase2c0_protocol.json').read_text(encoding='utf-8'));vector=json.loads((ROOT/'results/phase2c0_paired_vector_attribution.json').read_text(encoding='utf-8'));tests=ET.parse(ROOT/'results/phase2c0_pytest_acceptance_v1.xml').getroot().find('testsuite');assert tests is not None and tests.attrib['failures']=='0' and tests.attrib['errors']=='0';nt=int(tests.attrib['tests']);decision='PHASE 2C PILOT CONDITIONALLY BLOCKED'
    head='# Stewart Phase 2C-0：未知执行器动态、突变负载与任务可行性\n\n日期：2026-10-29。项目：`__PROJECT_ROOT__`。Python：`D:\\miniconda\\envs\\stewart_rl\\python.exe`。本轮无神经网络、无Gymnasium、无PPO/SAC/BC。\n\n'
    selected=decomp[decomp.window=='load_all'];report=head+'''## 首要结论：主要是横向动态跟随，不是Z抗扰

先只读原Phase2B3 uneven_dynamic四种方法的原始physics.csv.gz。C4-Observer在**3.585s、卸载前65ms的持续加载阶段**达到加载位置范数峰值.3381432959mm。同一时刻向量为[+.134482,−.310248,+.001390]mm。平方误差贡献X15.8171%、Y84.1812%、Z.001689%，不能把该误差归作竖直力误差。这个分解没有把不同时间的各轴峰值相加。

'''+table(selected,['method','peak_time_s','peak_norm_mm','x_at_norm_peak_mm','y_at_norm_peak_mm','z_at_norm_peak_mm','x_squared_contribution_percent','y_squared_contribution_percent','z_squared_contribution_percent'])
    report+='''
完整表还单列各轴独立峰值、RMSE、加载前/加载前50ms/持续加载/卸载50ms/恢复期，不能与“同一时刻向量”混用。姿态按世界SO(3)计算；源日志保留真实平台速度、六支链行程/速度/实际力、参考、外载及F_C0/共享补偿。

## 原峰值时刻的六支链

'''
    peak=selected[selected.method=='C4-Observer'].iloc[0];d=pd.read_csv(Path(peak.source_folder)/'physics.csv.gz');i=int(np.argmin(abs(d.time_s-peak.peak_time_s)));leg=[]
    for j in range(6):leg.append(dict(leg=j+1,command_N=d[f'f_cmd_{j}'].iloc[i],actual_N=d[f'force_{j}'].iloc[i],slide_mm=d[f'q_{j}'].iloc[i]*1000,velocity_mm_s=d[f'dq_{j}'].iloc[i]*1000,shared_additional_N=d[f'F_add_total_{j}'].iloc[i]))
    report+=table(pd.DataFrame(leg),['leg','command_N','actual_N','slide_mm','velocity_mm_s','shared_additional_N'])
    report+='''
## 三个明确标记的新配对实验

旧记录没有同一.85时间缩放参考、同一模型、无外载的完整严格配对，不能拿另一历史轨迹相减。本轮新跑：名义gain无载、名义gain有载、原uneven gain无载；第四项uneven有载使用原历史记录，不重跑旧实验。四项参考q/p/R逐位一致，有/无载各自加载前qpos/qvel/force逐位一致，才计算增量。

'''+table(pairs,['case','new_experiment','load_peak_time_s','load_peak_norm_mm','same_historical_peak_time_s','x_at_same_peak_mm','y_at_same_peak_mm','z_at_same_peak_mm'])
    report+='\n同一3.585s的向量差（mm）：\n\n'+table(pd.DataFrame([dict(component=k,x_mm=x[0],y_mm=x[1],z_mm=x[2]) for k,x in vector['xyz_mm'].items()]),['component','x_mm','y_mm','z_mm'])
    report+='''
在该时刻，名义无载本来就有Y=-.238211mm；轻微gain带来无载Y增量-.068420mm；加入外载的Y增量仅-.003616mm。故大部分横向误差在同轨迹无载时已存在，gain进一步加重，而这次突加载荷不是主因。这里是严格匹配向量差，不是范数的可加因果百分比，也未完整分离惯性、被动阻尼和PD静差各自贡献。

## 四维动作权限

四维[Fz,Mx,My,Mz]无直接Fx/Fy，但动态耦合意味着不能仅凭此断言X/Y绝对不可控。本轮在同一名义动态无载轨迹，2.2～3.2s施加四维各轴±[.2N,.01Nm,.01Nm,.005Nm]，另对Fx/Fy各±.02N作**单方向局部**诊断。共12回合；探针前2200步与名义无载基线qpos逐位一致。局部灵敏度和同一3s状态增量在phase2c0_authority_sensitivity.csv；归一化输出容差和已测幅值后，4维响应矩阵秩4、6维矩阵用于物理诊断，不能解释为任意随机动作箱安全。

四维可以间接改变X/Y，但伴随姿态/其他轴取舍；本轮有限探针与原4Dlead没有证明能充分解决原横向问题。新增一个允许的局部6D权限诊断：在冻结C4上加世界XY反馈（300N/m、10Ns/m，Fx/Fy各限±.04N），其余C4不变且共享同一预算。原加载峰值.338143→.232482mm，卸载峰值.245854mm；本次加载/卸载均未越原严格带，说明直接横向权限有实际作用，也说明简单传统反馈可以解决主要横向问题的一部分。

这不是自动新建C5、不是六维随机探索、不是神经网络。整个六维动作箱未验证，不能据此直接开六维PPO。原uneven_dynamic必须列为**当前四维RL训练任务的阻塞项**：需要先解决权限和强传统基线，不能只增加奖励惩罚。

输出：error_decomposition.csv、paired_decomposition.csv、paired_vector_attribution.json、authority_sensitivity.csv及带局部线性近似警告的authority_linear_diagnostic.json；图xyz_historical/so3_historical/strict_pairs，原始数据源路径均可追溯。
'''
    (ROOT/'docs/27_Phase2C0动态轨迹误差归因.md').write_text(report,encoding='utf-8')
    design=head+'''## 可关闭的合成一阶模型

模型仅表示SYNTHETIC FIRST-ORDER FORCE RESPONSE：τ_i dF_drive_i/dt+F_drive_i=F_command_i。τ为研究候选，不是液压阀、电液伺服、摩擦或实机辨识模型。原MJCF、质量/惯量、原生执行器及C3/C4均未改。

dt=.001s，α_i=−expm1(−dt/τ_i)，F_drive_next=clip(F_drive_previous+α(F_command−F_drive_previous),±400N)。τ=0和disabled原样直通。正负阶跃、保持、卸载共4000个数组采样与解析解核对；这4000不是MuJoCo物理步。

## 采样顺序和三个信号

1. 在物理步开始，原控制器读取允许测量与原当前参考，C0/C3/C4生成经共享预算的F_command。
2. 原观测器记忆的是这个已下发F_command，而不是隐藏F_drive/F_actual。
3. plant模块用当前已知命令和内部旧状态更新一次F_drive_next；无未来物理状态。
4. 前六native ctrl写F_drive_next，后六ctrl仍只由原外扰profile写入；actual force=native clip(actual gain×drive)。
5. 原mj_step恰好一次、原mj_forward、安全检查、原.2s受限保持及记录；CSV状态时间为积分后t(k+1)，输入/控制步为k。

ZOH对执行器离散状态在固定命令下解析精确；将更新端点驱动力在一个1ms数值积分步内保持，是明确的离散plant接口约定，不宣称连续力波形和整个机器人积分也被解析精确求解。没有额外物理步，也未把一阶响应重复应用两次。1kHz指仿真离散控制周期，不是Windows硬实时保证；视频按物理时间25fps播放，不冒充原墙钟采集速度。

## 如何复用原执行循环

execution_hook.py从**只读冻结**DisturbanceOwner.step构造同一执行函数，只在原self.last_force赋值之前插入force=self.apply_actuator_command(force)。不全局替换mujoco函数，不用mjcb_control（避免mj_forward重复触发状态更新），不改历史源文件。AST删除这唯一插入后与原函数完全相同，单测验证且保留原唯一mj_step。这样保护保持的原力命令也经过同一动态模块；Hold时不能重置隐藏驱动状态或跳过滞后掩盖惯性。

## 初态与严格关闭等价

隐藏驱动力初值使用冻结名义Home参考平衡支链力ff，六缸约80～90N，保存actuator_dynamics.json初始化依据。所有方法初始qpos/qvel完全相同，没有从不合理的零驱动力起步，也未写live qpos制造稳定。disabled与enabled+τ0均不增加隐含一步延迟。

'''+table(pd.read_csv(ROOT/'results/phase2c0_zero_equivalence.csv'),['method','enabled','tau_s','bitwise_equal','maximum_difference','signals_and_observer_equal','protection_termination_equal'])
    design+='''
原C0/C3完整6000步对比Phase2B2 force10；C4完整6000步对比Phase2B3 reverse_z（含相同旧gain、外载和参考）。全部458维integration、物理信号、原观测器diag及保护/终止逐位一致。仅等价对照保留旧静态gain；新滞后研究全部nominal gain，不混入gain/质量/噪声/延迟。

## 信息边界

| 信号 | C4可读 | 独立安全/离线可读 | 说明 |
| --- | --- | --- | --- |
| 平台p/R/v测量 | 是 | 是 | nominal为真值，噪声/延迟工况只扰此通道 |
| 全链q/dq | 是 | 是 | 保持原理想反馈，非完整硬件传感器模型 |
| 原参考、名义模型、当前/过去下发请求 | 是 | 是 | 原C3/C4不改 |
| C3等效扰动与历史 | 是 | 是 | 不是实际外力传感器 |
| actual τ、hidden drive、actual force | 否 | 是 | 未定义实际力传感器，不得输入Actor |
| 外载真值、实际隐藏plant参数 | 否 | 是 | 仅单独特权诊断可用 |
| 未来状态 | 否 | 否 | 没有未来状态提供给控制器 |

每步记录observer_memory_wrench和known_command_wrench，独立复算一致；非零τ下它们与nominal_drive_wrench确实不同，证明未把内部输出力偷换成已知命令。原C3仍面对未知执行器模型误差，dhat同时含外载、动态误差与局部近似误差。

## 日志与共享预算

F_command为最终有界请求；F_drive_previous/drive为内部驱动；force为MuJoCo真实力；f_cmd历史别名保持“命令”语义，ctrl前六在新变体中是drive。F_add_total=F_command−F_C0仍合计C3、C4及候选补偿，±5N、20N/s不变；实际drive/force各±400N。不能把原命令总变化率误认为附加限速。

458维NPZ是完整MuJoCo integration状态，不包含Python额外状态。隐藏执行器drive/prev逐步保存在CSV，τ/初值在JSON；若未来要中途恢复执行，还需恢复原控制器内部状态或从前缀重建，不能仅凭NPZ宣称完整Python系统断点续跑。本轮观看使用已记录RGB/状态，不重新积分。

lagged_residual_wrench/lagged_additional_wrench通过独立平行影子滤波（同初值/τ，输入为实际访问状态上的C3/C0命令历史）提供离线实际增量。它不是另一条完整闭环反事实轨迹，也不输入控制器。旧W_physical_*保留原“瞬时命令反事实”定义，有lag时不能当作纯新增补偿作用；新分析用明确的三信号和lagged_*字段。

所有原行程、weld1e-4m/rad、条件数50、Yaw±.2°、Roll/Pitch±.5°、相对Home XY±2mm/Z±5mm、部分代理5mm及.2s保持不变。代理仍PARTIAL PROXY QUERY，不是整机或连续安全认证。
'''
    (ROOT/'docs/28_Phase2C0执行器动态模型与因果信息审计.md').write_text(design,encoding='utf-8')
    robust=head+'''## 先小幅预检，再冻结单因素评价

统一τ=0/2/5/10ms；每个非零τ先C4无载Home六秒预检，再运行C0/C3/C4的Home、Z阶跃、动态参考+小外载，另C4 Roll阶跃。若出现保护/不稳定则不升更大τ。本轮四档全部完成。之后才测试小差异τ=[4,4.4,4.8,5.2,5.6,6]ms的Home/动态三方法，共6回合。所有实际τ固定在回合开始，没有运行中突改。

下表为冻结C4-Observer，未重新整定观测带宽30rad/s、Yaw gain1或共享补偿参数。

'''+table(f[(f.stage=='lag_development')&(f.method=='C4')],['task','tau_s','load_peak_position_mm','load_position_iae_mm_s','unload_reverse_z_mm','yaw_peak_deg','maximum_command_drive_difference_N','max_force_rate_N_s'])
    robust+='''
Z阶跃（−6N）下，2/5/10ms的加载位置IAE相对同任务τ0约增加4.47%/14.01%/32.16%；峰值约.06609→.06856/.07462/.08818mm。Home保持基本不变，未发生零驱动重力假瞬态。Roll和动态任务也有变化，但后者仍主要是横向跟随问题，不能仅依据全回合RMSE给τ优劣定性。

'''+table(f[(f.stage=='differential_lag')&(f.method=='C4')],['task','tau_s','load_peak_position_mm','load_position_iae_mm_s','yaw_peak_deg','maximum_command_drive_difference_N'])
    robust+='''
差异τ候选均完成，未见明显新Yaw偏置；不代表更大差异、静态gain混合或运行中突变τ也安全。实际命令/驱动的最佳相关延迟仅是离线相位指标，不等于精确τ辨识；驱动力仅用于该离线核验。

## 独立平台观测延迟与噪声

延迟1/2/5ms，τ关闭，不与任何gain/质量失配混合。平台位置/旋转/速度噪声分别为2μm、2×10⁻⁵rad、2×10⁻⁵相应m/s或rad/s，逐因素。numpy.default_rng的PCG64、seed42，预生成6400×12标准高斯数组，各C0/C3/C4使用同序列。三类噪声原始标准样本的配对逐位核验，无挑选种子。旋转噪声以小旋转向量左乘世界姿态，非直接改欧拉角；没有改真实qpos/qvel。

capture/read/control/physical step分别记录；队列仅提供过去样本，启动保持首样本，capture≤read，差为固定1/2/5步；不通过多做mj_step制造延迟。真实安全与离线误差仍用实际状态。

'''+table(f[(f.stage.isin(['delay_development','noise_development']))&(f.method=='C4')],['stage','variant','measurement_json','load_peak_position_mm','load_position_iae_mm_s','max_force_rate_N_s','success'])
    robust+='''
1/2/5ms延迟下C4加载IAE约增加2.42%/4.85%/12.03%；三类小噪声影响有限、均完成。本轮没有真实传感器规格依据，严格标记SYNTHETIC SENSOR NOISE。只有p/R/v被扰动，计算Jacobian/关节PD的全链q/dq仍理想，属于PARTIAL OBSERVATION CHANNEL ROBUSTNESS。C0几乎不变不等于其真实全传感器性能优于C4。单seed、单量级不能给出噪声统计可靠性或硬件保证。

## 哪些信息有用，哪些仍不可辨识

已下发请求、六支链响应、平台速度与C4等效扰动的历史包含相位变化；20Hz有限因果lead在两个独立冻结工况能改善实际误差，表明允许反馈确有补偿信息。但等效扰动混合外载、执行器误差和测量延迟，不能直接据其值宣称已识别6个τ。激励矩阵的数值秩与命令能量已存all_trials.csv；即便数值满秩，也不解决未知外载与潜在参数的结构歧义。Home恒定支撑尤其缺乏辨识激励。

本轮未做完整系统辨识或训练历史网络，未证明部分观测可直接成为Markov状态。不能因为未来网络更大就忽略这些限制；若后续研究需先定义历史窗口和跨τ验证。
'''
    (ROOT/'docs/29_Phase2C0观测延迟噪声及C4鲁棒性.md').write_text(robust,encoding='utf-8')
    final=head+f'''## 正式决策

**{decision}**。

本轮取得正证据：未知10ms滞后和5ms平台延迟均形成安全可重复退化；有限因果四维补偿在两个新冻结评价中分别改善10.82%和17.16%，通过事前工程筛选门槛，不能忽略这些结果。

但相同收益已经由廉价经典规则实现。尚未建立PPO在强C4+4Dlead之上的额外收益依据；两类Home任务本来均在严格精度带内，原大横向误差又超出现有四维直接权限。综合判断是**物理可行性已有，额外学习价值仍需确认**，不把最低工程筛选门槛自动等同必须训练。没有声称PPO无效，也没有为了路线预设GO。

如果用户目标仅为当前容差和抗扰改善，可直接保留经典方案，暂不投入训练。若希望研究更高精度和跨未知动态适应，应先冻结新独立条件、历史观测和强经典基线，再由用户审核是否值得一次明确小预算Pilot；本轮不自动封装或训练。

## 实际运行与预算

'''+table(f.groupby('stage',as_index=False).agg(trials=('success','size'),successes=('success','sum'),physics_steps=('physical_steps','sum')),['stage','trials','successes','physics_steps'])+f'''
研究合计{v['research_trials']}×6秒={v['physical_steps']}个1ms积分步（564仿真秒），94回合均完整完成，0新保护失败；预先上限96，未凑满。全部失败/退化均保留；无新失败不能人为制造失败视频。旧已知失败和5803历史文件SHA256均不变。

附加成本单列：GUI off/on各2048步，共4096；两轮日志吞吐量基准各4000步，共8000（第一轮与主研究并行，第二轮无其他本代理模拟/渲染负载）。已知非自动测试实际积分合计576096。解析ZOH4000数组采样不计MuJoCo步；自动测试内短物理运行另计，不伪称RL样本。

## 冻结评价及预先门槛

开发筛选只看相对同任务τ0 C4的加载位置IAE，完整成功且退化≥10%才入选，最多2类。实际入选10ms统一lag（开发退化32.16%）、5ms部分观测delay（12.03%）。新评价载荷改为−5N、1.55～3.35s，所有参数、载荷、方法、门槛在新评价前写evaluation_frozen.json；没有用最终结果调参。

C4参数不变；C4_fourD为20Hz等效扰动差分lead，.02s系数，Z/R/P/Yaw额外限±[.5N,.025Nm,.025Nm,.015Nm]，总Yaw请求仍±.30Nm。C4_fixedLead为独立经典对照：3ms当前/过去C4**请求命令**导数超前，导数滤波4ms、逐腿额外限±.1N；它不是读取真实τ，也不是actual drive。两种新增补偿都与C3/C4共用±5N、20N/s。

事前收益门槛：加载位置IAE至少降低10%；无新增失败；实际总力峰值变化率不增>5%、力平方积分不增>2%、非目标XY峰值不增>5%。原完成标准/严格恢复/安全完全不变。

'''+table(f[f.stage=='frozen_evaluation'],['task','variant','load_position_iae_mm_s','load_peak_position_mm','unload_reverse_z_mm','max_force_rate_N_s','force_square_integral_N2_s','xy_peak_mm','success'])+'\n'+table(gate,['task','method','primary_error_reduction','force_slew_ratio','force_square_integral_ratio','xy_peak_ratio','passed'])+'''
工程筛选2/2通过（C4_fourD）。3ms经典超前在lag中IAE**恶化约2.90%**，在delay中仅改善5.28%，两者均不通过10%门槛；未隐藏。四维lead自身也是经典因果控制，不是学习结果。未来即使训练，必须与C4+fourD这条强基线比较；仅重现该规则不能叫RL额外控制收益。

## 特权物理诊断

'''+table(f[f.stage=='privileged_diagnostic'],['task','variant','load_position_iae_mm_s','load_peak_position_mm','max_force_rate_N_s','success'])+'''
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

'''+table(pd.DataFrame(bench['records']),['journal_stride','physical_steps','construction_seconds','run_and_archive_seconds','physical_steps_per_wall_second','append_pipeline_seconds','recorded_rows'])+'''
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
'''
    final+='''
## 已实际测试的归档回放命令

```powershell
Set-Location __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/watch_phase2c0.py --trial results/runs/phase2c0_20261029_v1/frozen_evaluation/eval_actuator_lag/C4_fourD --validate-seconds 2
```

该命令实际打开并正常关闭。持续查看时去掉--validate-seconds 2；窗口显示ARCHIVE REPLAY和命令/drive差异，不重新训练或积分。原生真实动态窗口的有界命令为上文validate_phase2c0_observer.py，已实际运行。所有未执行项仍按前文标记。
'''
    (ROOT/'docs/30_Phase2C0强化学习任务可行性与验收报告.md').write_text(final,encoding='utf-8')
    write_json(ROOT/'results/phase2c0_decision.json',dict(decision=decision,engineering_benefit_filter_passed=True,classes_with_passed_filter=2,reason='Causal classical4Dlead already provides the measured benefit; extra PPO value and future independent uncertainty validation remain unproven',gate_A='PASS',gate_B='PASS_HOME_Z_ONLY; original XY task blocked',gate_C='causal aggregate feedback demonstrated, individual hidden parameters not proven identifiable',gate_D='ENGINEERING_FILTER_PASS / extra_RL_value_unproven',gate_E='bounded_cost_estimate_available',tests_passed=nt,physical_steps=v['physical_steps'],no_training=True))
    write_json(ROOT/'configs/phase2c0_recommended_task.json',dict(status='PROPOSAL_ONLY_CONDITIONALLY_BLOCKED',training_enabled=False,task_candidates=['Home Z step with uniform10ms synthetic lag','Home Z step with5ms partial platform delay, no actuator lag'],excluded_task='uneven_dynamic with dominant XY under unvalidated four-dimensional permissions',reference_controller='Frozen C4-Observer',mandatory_classical_baselines=['C4+fourD20Hz lead','C4+fixed3ms command lead'],candidate_action_order=['Fz_N','Mx_Nm','My_Nm','Mz_Nm'],initial_single_direction_verified_scale=[.2,.01,.01,.005],random_joint_action_box_verified=False,decision_frequency_first_test_Hz=20,not_tested_alternative_Hz=50,observation_whitelist=['measured p/R/v','permitted chain q/dq','current reference','known issued commands','causal C4 equivalent estimates','past residual/actions and errors'],forbidden_actor_inputs=['actual tau','hidden drive','actual force without validated force-sensor protocol','true disturbance','future state'],pilot_budget_proposal_only=dict(first_smoke_decisions=1024,conditional_max_decisions=2048,physical_steps_at20Hz_without_reset=[51200,102400]),new_final_test_required=True,current_data_are_seen_screening_data=True,remaining_preconditions=['Clarify extra value beyond classical4Dlead','Freeze unseen heldout conditions','Validate reset/MDP or history state','Validate random action safety before any learning']))
    print('Reports generated:',decision,'tests',nt)

if __name__=='__main__':main()
