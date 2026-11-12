"""Generate Chinese reports only from archived observations and frozen design."""
import _bootstrap,json
from pathlib import Path
import pandas as pd,numpy as np
from src.model import ROOT,write_json,sha
from src.workspace.feedforward_study import RUN

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def table(d):
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.9g}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def put(name,text):(ROOT/'docs'/name).write_text(text,encoding='utf-8')

def main():
    d=pd.read_csv(ROOT/'results/phase2c3b_controller_comparison.csv');formal=d[d.stage=='formal'];full=d[d.stage!='joint_preflight'];p=read(ROOT/'results/phase2c3b_feedforward_parameters.json');s=read(ROOT/'results/phase2c3b_safety_validation.json');rl=read(ROOT/'results/phase2c3b_rl_task_recommendation.json');frozen=read(RUN/'methods_frozen.json');strong=frozen['strong_method'];hist=pd.read_csv(ROOT/'results/phase2c3b_error_phase_decomposition.csv');nom=pd.read_csv(ROOT/'results/phase2c3b_nominal_position_audit.csv');paired=pd.read_csv(ROOT/'results/phase2c3b_paired_effects.csv');actions=pd.read_csv(ROOT/'results/phase2c3b_action_authority.csv');testfile=ROOT/'results/phase2c3b_pytest.log';testtext=testfile.read_text(encoding='utf-8',errors='replace') if testfile.exists() else '未执行完整套件；本报告不可作为最终验收。';testline=next((x.strip() for x in reversed(testtext.splitlines()) if ' passed' in x),testtext[-400:]);v=s['video'];columns=['method','rise_s','z_rmse_mm','peak_z_error_mm','xy_peak_mm','so3_rmse_deg','final_home_error_mm','max_force_N','max_force_rate_N_s','force_energy_N2_s','additional_force_peak_N','budget_projection_fraction'];summary=formal.pivot(index='rise_s',columns='method',values='z_rmse_mm').reset_index();comparison=formal.merge(formal[formal.method=='C4'][['rise_s','z_rmse_mm']].rename(columns={'z_rmse_mm':'C4_z_rmse_mm'}),on='rise_s');comparison['gain_vs_C4_percent']=(1-comparison.z_rmse_mm/comparison.C4_z_rmse_mm)*100
    scope='本轮只验证MuJoCo纯世界Z正向50mm任务。887个独立网格凸包查询配对、103个连接组件排除仍属PARTIAL覆盖；30个被动铰链没有原生机械硬止挡。无原生接触不等于全几何无碰撞，更不等于实机安全认证。'
    command="""```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\\miniconda\\envs\\stewart_rl\\python.exe'
& $py scripts/view_phase2c3b_live.py --method FF-VA --rise 6
```
这是新建独立日志目录的真实动力学执行；不是PPO采样。Pause / Resume暂停物理时钟，Camera left/right只改变相机，Quit关闭并归档。默认完整15秒物理轨迹后停留显示，不再积分；墙钟上限300秒。不会无限重复运动，也不修改原有持续运行控制面板。

下列有界开/关对照已实际执行两次（第二次包含六支链曲线）：
```powershell
& $py scripts/view_phase2c3b_live.py --method FF-VA --rise 6 --validate-steps 4000 --compare-headless
```
每次开启与关闭显示各4000个物理步，完整积分状态逐位一致。GUI验收只走到4秒、约25mm；全50mm往返的证据来自正式物理日志和现场RGB录像。窗口采用Windows Qt＋MuJoCo Renderer单线程所有权，不是原生GLFW Viewer；没有从GUI线程修改live状态。首次尝试ProcessPool被Windows IPC权限拒绝、未执行该批动力学，错误日志保留，随后改成无IPC的独立CLI进程。
"""
    put('42_Phase2C3B动态误差与前馈建模.md',f'''# Phase 2C-3B：动态误差与前馈建模

日期：2026-11-05。所有数值来自已有3A的1ms物理日志或本轮真实动力学；没有训练网络。

## 历史误差审计

历史C0、C4、fourDlead各T4/T6/T10共9条轨迹，按初始保持、上升加速/减速、顶部保持、下降加速/减速、最终保持分为63行。正误差定义为实际Z减参考Z。

{table(hist[hist.method=='C4'][['rise_s','phase','z_rmse_mm','z_mean_mm','velocity_error_correlation','acceleration_error_correlation','observer_mean_Fz_N']])}

移动段上升主要为负误差、下降为正，具有速度滞后特征；加速度项与过渡误差相关。分段速度、加速度及误差的相关系数和联合回归残差保存在CSV，保持段零方差的相关系数为空，不填0冒充估计。上升/下降过渡和保持恢复共同影响全程RMSE，不能以单一系数解释全部误差。

图见`assets/plots/phase2c3b_historical_T4.png`、T6、T10。CSV为`results/phase2c3b_error_phase_decomposition.csv`。本轮6种控制的误差分解见`phase2c3b_control_phase_errors.csv`。

## 冻结控制律审计

C0源码`src/phase1/controllers.py`明确为`reference.ff + Kp*(q_ref-q) + Kd*(dq_ref-dq)`：已有参考速度参与速度误差反馈，`reference.ff`是沿参考构型计算的静态平衡力。不能称C0完全没有速度项。C3/C4的观测器使用名义M、D、K估计闭环等效扰动；C4还补偿Yaw；fourDlead对因果扰动估计作20Hz受限超前。它们没有单独下发本次新增的`Dzz*v_ref + Mzz*a_ref`显式参考动态广义力。

新增项意在补偿名义被动阻尼及运动惯性需求，区别于Kd速度误差反馈；不新增重力。它仍是局部Z等效模型，不是Stewart全逆动力学；支链耦合、柔性闭链、几何变参和控制离散效应仍有残差。

## 参数来源与单位

`Wff_z = kv*v_ref + ka*a_ref`，kv={p['kv_N_s_m']:.12g} N·s/m、ka={p['ka_kg']:.12g} kg。来源：冻结Home名义约束切空间模型`{p['source']}`，SHA256 `{p['source_sha256']}`。通过`E*T=0`、`Jsite*T=I`计算`Mtask=T.T*Mfull*T`和`Dtask=T.T*diag(damping)*T`。平台20kg不是包含支链运动贡献的50.185kg任务空间等效质量。

0～50mm的26个名义位置审计：Mzz范围{nom.effective_mass_kg.min():.9f}～{nom.effective_mass_kg.max():.9f} kg，Dzz范围{nom.effective_damping_N_s_m.min():.9f}～{nom.effective_damping_N_s_m.max():.9f} N·s/m。位置变化存在，但这条纯Z路径上幅度有限；没有宣称Home值在全工作空间精确。

## 因果解释的边界

相关系数属于统计描述，切空间M/D属于简化名义解释。只有相同轨迹/物理模型/预算、只改变前馈结构的配对实验才支持控制干预效应。FF-V减少误差，FF-VA进一步减少，支持名义参考动态补偿不足是本组快速Z误差的重要来源；并不证明所有误差都是阻尼，也不能把无滞后名义误差说成未知执行器造成。

{scope}
''')
    dev=d[d.stage=='dev'];put('43_Phase2C3B强传统控制器设计.md',f'''# Phase 2C-3B：强传统控制器设计

## 三个独立版本

FF-V=C4+kv*v_ref；FF-VA=C4+kv*v_ref+ka*a_ref；FF-VA-Lead=原C4+fourDlead+同一VA前馈。旧控制器与配置未改，新增包装在`dynamic_reference_feedforward.py`和`large_motion_feedforward_wrapper.py`。

使用世界系、平台中心力矩参考点，wrench顺序`Fx,Fy,Fz,Mx,My,Mz`，`Wff=[0,0,Fz,0,0,0]`，解`J.T*Fleg=Wff`。采用原模型可观测构型计算的完整链式Jacobian，六支链编号保持原序，禁止简单均分Z力。控制1kHz，原Jacobian刷新周期保留；原fourDlead的20Hz更新保留。前六真实执行器与后六外扰通道隔离。

## 一个真实附加预算

`F_add_total=F_command-F_C0`的每链幅值≤5N、变化率≤20N/s；总力仍由原生±400N限制。原C3本身有限制，新包装不放宽它；C4/fourDlead/FF/诊断探针最终一起进入原SharedBudgetGovernor一次。反事实counter governor只对同一历史状态计算日志，不下发、不增加独立权限、不再次调用物理积分。

插入发生在原final-command observer记忆修正之前，观测器因此记住真实最终命令，避免观测器把新增前馈误当外扰抵消。零前馈的C4与fourDlead均做了逐物理步完整状态等价测试；正式C0/C4/fourDlead的9条轨迹也与旧3A保存状态逐位一致。

`F_command`为最终命令，`F_drive`为一阶合成驱动状态，`force`为MuJoCo执行器实际输出。`ff_realized_wrench`准确含义是J转置映射的**合成附加命令**（经过预算、驱动滞后之前），不是纯FF实际物理贡献。`ff_effect_command_N`与`ff_probe_effect_N`是同一状态下的命令差分，不是另一次完整闭环轨迹的因果差分；实际闭环效果用配对实验。

`lagged_additional_wrench`另用真实当前几何映射实际力相对于同状态C0命令的滞后影子驱动差，供离线诊断，不输入控制器；它同样不是另一个完整C0闭环。请求/接受命令/物理驱动力对照图见`phase2c3b_command_drive_wrench.png`。原日志约定保留：每行实际状态为步后time_s，控制参考/命令对应步前约time_s−0.001；独立大工作区的解析加速度诊断按行时刻计算。FF自身分量记录的是实际计算所用步前参考，二者至多相差1ms，不将其混用为精确同步的控制输入。

静止保持的参考v/a均为0，FF请求归零；C4/驱动可能保留因果恢复过程，不能要求整个附加控制瞬时为0。日志分别记录C4、原lead、FF请求、合成请求、接受值、限制原因及实际力。

## 有限开发与冻结

参数只取名义来源的一对候选，倍率1.0，无搜索网格、无测试集拟合。FF-V实际ka=0；每试验参数文件也保存公共来源候选ka，方法名和分量日志确定实际开关，不能把FF-V解释为也执行了惯性前馈。

{table(dev[['method','amplitude_m','rise_s','z_rmse_mm','additional_force_peak_N','success']])}

开发集平均Z-RMSE：{json.dumps(frozen['development_scores'],ensure_ascii=False)} mm。按预先冻结的最低平均误差规则选{strong}，随后才运行正式T8及鲁棒性。VA与VA-Lead差异很小，不宣称统计显著；没有事后给Lead重调参数。冻结文件`{RUN/'methods_frozen.json'}`。

共享预算最大实际占用{s['maxima']['additional_force_peak_N']:.6f} N/链，最大变化率{s['maxima']['additional_force_rate_peak_N_s']:.6f} N/s。完整结果中预算投影占比最大{full.budget_projection_fraction.max():.9g}；这批任务没有因扩大权限获得改善。原总力变化率不受附加20N/s规则等同约束，二者分别报告。

{scope}
''')
    put('44_Phase2C3B不同速度与外载对照.md',f'''# Phase 2C-3B：不同速度与外载对照

## 实际执行清单

预先上限49组：开发9、正式24、无载滞后6、联合7秒前缀预检3、完整联合负载3、固定动作权限4。实际49组，完整46条全部完成，3条前缀只标预检通过、未完成且success=False，不计入完成率。它们不是49个独立随机场景；这是冻结参数下的确定性配对实验。独立CLI双进程只隔离物理数据，未共享MjData。

实际研究积分{s['research_physical_steps']}步、{s['research_sim_seconds']:.3f}秒；额外GUI一致性诊断两次共16000步，不混入正式任务数。自动测试动力学另计测试成本，不是训练样本。

## 全程Z误差（mm）

{table(summary)}

T8没有用于开发方法选择；T6/T10与开发重复条件只能称复现对照，不能称盲测。每条轨迹均完整Home→上升→顶部保持→下降→Home，单程T不是总时长，总时长2T+3秒。

## 全指标与代价

{table(formal[columns])}

FF-VA使名义Z误差显著按数值幅度下降，但X/Y误差没有被一维Z前馈解决。完整指标包括3D位置误差、SO(3)、最终Home、各链行程、实际力平方积分、最大变化率和保护。对照表不可仅抽一条最佳视频。采用的名义系数与参数权限相同；无训练随机种子或显著性检验。

必须强调，XY不只是“仍有误差”，而是比原C4增大：

{table(pd.DataFrame(rl['non_Z_tradeoffs']))}

T4峰值0.018800→0.264924mm，约14.1倍；T10约24.4倍。三维RMSE仍约减半，但FF后非Z误差占三维平方误差98%以上。SO(3)也从极小量增加到约千分之几度，未触保护但存在代价。故“Z改善”与“非目标方向退化”必须同时呈现。

相对C4的Z改善：

{table(comparison[comparison.method=='FF-VA'][['rise_s','C4_z_rmse_mm','z_rmse_mm','gain_vs_C4_percent']])}

## 滞后与负载配对

{table(d[d.stage.isin(['lag','load'])][['stage','method','tau_s','external_Fz_N','z_rmse_mm','peak_z_error_mm','xy_peak_mm','additional_force_peak_N','success']])}

合成tau0/5/10ms无载任务先通过，再执行tau5ms＋世界Z−5.5N。首次联合条件各做7秒前缀预检，通过后才做完整15秒。加载5.2秒、卸载11秒，都是该T6轨迹冻结时序；未执行tau10ms＋负载，也未扩展强度。它们不是经过辨识的真实液压参数。

{table(paired)}

表中RMSE相减只描述配对全程指标变化，不能作为可加的误差分量。`paired_time_series.csv`按同一1ms时刻保存真实位置差，区分参考动态、滞后干预、给定滞后下外载干预，包含闭环反馈响应。

## 控制权限探针

{table(actions)}

这是固定±0.25N/1.3Hz正弦Z力请求，20/50Hz仅在正弦探针中比较相同15秒任务；不是Actor，不是PPO训练。相位级改善/恶化全部保存在`phase2c3b_phase_action_effects.csv`。不能因某一窗口改善就声称固定探针整体更优。

## 保护与失效

本轮没有动力学保护失败或原生力饱和；没有删除性能退化的探针。力峰值{s['maxima']['max_force_N']:.6f}N，weld平移峰值{s['maxima']['max_weld_translation_m']:.9g}m、转角峰值{s['maxima']['max_weld_angle_rad']:.9g}rad，条件数最大{s['maxima']['max_jacobian_condition']:.6f}，网格查询最小{s['minima']['expanded_mesh_gap_m']*1000:.6f}mm。所有历史失败和IPC启动错误均保留。

{scope}
''')
    scales=pd.read_csv(ROOT/'results/phase2c3b_observation_scale_audit.csv')
    put('45_Phase2C3B大位移RL任务设计.md',f'''# Phase 2C-3B：大位移残差RL任务设计（未训练）

## 基于实际数据的判断

**{rl['decision']}**。范围仅限已验证MuJoCo纯Z仿真。检查明细：`{json.dumps(rl['checks'],ensure_ascii=False)}`。

预先冻结的有限安全/动作权限筛选通过={rl['physical_action_filters_pass']}，但它不是最终研究GO的充分条件。完整任务指标发现FF引入了更大的XY偏差、非Z分量占剩余三维平方误差98%以上，一维Fz并未证明能解决它。最终因此有条件阻断PPO启动。最初只检查物理/动作过滤的GO草稿保存在`rl_physical_filter_initial_assessment.json`；最终结论加入用户要求的非目标方向审查，没有据此重调参数、删掉结果或放宽门槛。

下一步最少应先诊断纯Z标量FF与完整任务空间耦合：名义M/D的Z列含非零X/Y及力矩分量，使用仅Z的对角项未补偿它们，这是待验证模型解释，不是已证明的全部原因；Jacobian保持、闭链柔性和观测补偿交互也可能有贡献。本轮不新增全维控制算法。需单独验证完整名义动态需求、非目标误差及原预算，再决定保留纯Z学习目标或验证其他动作通道。不能因当前XY退化就直接扩Actor权限。

严格配对下FF-VA的5ms滞后额外Z位置差峰值仅约{paired[paired.method==strong].paired_lag_position_difference_peak_mm.iloc[0]:.9f}mm，而同滞后加载的额外位置差峰值约{paired[paired.method==strong].paired_load_position_difference_peak_mm.iloc[0]:.6f}mm。当前证据不支持把所有剩余误差归因于tau；若继续研究，受限加载/卸载瞬态比纯tau适应更有可测量信号。

经典FF已解决大部分名义参考动态误差，不能继续将其包装成RL必须解决的问题。残差学习仅研究未知驱动与突加载荷下的可观测控制误差；还必须以FF-VA为强基线。当前固定探针证明有限动作通道有效、特定阶段可改善，但并未证明可学习性、泛化、稳定收益或训练成本值得。若未来PPO收益不足，应停止扩大预算而不是削弱FF。

## 第一个任务与分布

优先L50-T6、tau5ms、−5.5N固定外载5.2～11秒；先固定一个工况建立零动作等价。50mm为主，30mm只作为后续较易/泛化任务。T4更快但参考FF已能良好解释，不宜一开始叠加所有扰动。tau5/10ms无载已验证，但tau10ms＋负载未验证，不允许将分项通过拼成联合安全证据。随后仅逐项引入预检后的tau/加载时刻小范围随机化，模型质量/增益、负Z、大姿态和真实液压均未执行。

本轮T8现在已经开启，只能作为本轮独立时序，不能再次称未来RL全新最终测试。未来需另冻结新时序或未见任务组合，在机械/数值预检后封存；评价在配置冻结后开启。当前未生成也未执行新的RL最终测试分布。

## 一维动作及可用权限

`delta_Fz`归一化[-1,1]，候选初始±0.1N；仅为小于已验证±0.25N探针的初始建议，不是全状态安全认证。映射原J转置，共享每链±5N/20N/s，不能另开预算。FF-VA在全部已完成条件的最小剩余每链峰值余量{rl['checks']['remaining_shared_headroom_N']:.6f}N。真实接受比例仍要在未来训练逐步记录，尤其限速及原C4竞争时。

一维Z通道不处理当前非目标X/Y漂移，禁止未经审计扩成六维。20Hz与50Hz先做同物理时长对照，按时间换算折扣`gamma(dt)=0.99**(dt/0.05)`，GAE和rollout同样匹配物理时间；不能只按相同决策数声称样本公平。

## 候选因果观测与尺度（仅设计）

候选单帧53维：位置误差3、SO(3)误差3、实际twist6、参考twist6、参考Z加速度1、相对slide6、slide速度6、已知基础命令6、因果扰动估计6、参考FFz1、接受的总附加命令6、上次动作1、实际/目标Z各1。3帧历史159维只是候选，不是已实现环境。不得输入真实tau、真实外载、内部F_drive、未来实际状态；参考当前解析速度/加速度是合法已知量。命令和扰动估计来自控制器可得信号。

大位移Z/目标尺度0.05m、slide0.06m、速度0.03m/s、参考加速度0.03m/s²；误差独立按Z约1e−5m、XY5e−5m、角度约0.000349rad初始候选，已知命令减Home再按5N缩放，接受附加命令按5N缩放。按训练/开发统计冻结，报告验证饱和率，不以最终测试拟合。宏观位移与微米级误差必须分开归一化，不能把旧3mm/1mm量程直接裁剪复用；实际加速度若使用须因果估计及滤波审计，当前候选仅用已知参考加速度。

以下范围审计只用开发集FF-VA L50T6；不是完整85维旧观测或候选53维环境运行。旧slide特征约60%元素超过原±10裁剪范围，说明不能直接复用旧归一化。参考Z在50mm处恰好处在旧裁剪边界，浮点误差也计入越界统计，不把它等同于明显信息丢失；主要饱和问题是slide。

{table(scales)}

## 候选奖励（仅设计、未冻结为可训练版本）

每物理时间积分归一化代价，初始项可为`Zerror² +0.1*XYerror² +0.05*angleerror² +0.01*requestedResidual² +0.005*acceptedAddSlew²`，上述变量先物理尺度归一化，按dt/0.05累计。加载/卸载IAE、峰值、全程RMSE及非目标姿态作为共同评价，不依据哪个窗口最有利于RL事后改权重。全程失败成本需足够覆盖少走轨迹的潜在收益，必须先离线检验成功/失败/保护/超时的折扣回报排序，当前没有完成这个未来环境奖励验收。

完整往返及尾部保持后才允许成功，不能在顶部先给终止奖励逃避下降卸载。安全失败terminated并执行原200ms真实保护保持，预算不足truncated且incomplete，奖励分别记录请求/接受幅值、真实力变化和失败项。不可把早停的较低RMSE当改善。

## 进入下一阶段的顺序

先实现新的环境/归一化与reward审计，零RL必须逐物理步复现FF-VA（包括final-command observer及共享governor），录制开关不改随机数/积分；然后才考虑1024个决策烟雾。通过独立安全/有效性门禁后再另冻结两个种子2048→4096的有限预算。以上均未在本轮执行。对照至少C4、fourDlead、FF-VA、FF-VA+PPO，同一物理任务；不继承旧3mm策略权重直接部署50mm。

{scope}
''')
    q=[
    ('50mm误差与哪些变量相关？','完整1ms分解显示移动方向、参考速度和加减速过渡均相关，CSV保留分段系数。'),
    ('是否有速度特征？','上升实际−参考以负值为主、下降反向，速度FF配对明显降低误差。'),
    ('是否有加速度特征？','VA比V进一步降低残差；这是本模型的干预证据，不是所有误差来源的唯一证明。'),
    ('原C4是否有动态补偿？','已有含参考速度的PD、静态平衡及含M/D/K的观测补偿，缺少此次显式D*vref+M*aref。'),
    ('kv来源？',f'冻结名义Dzz={p["kv_N_s_m"]:.9f} N·s/m。'),
    ('ka来源？',f'约束切空间等效Mzz={p["ka_kg"]:.9f}kg，并非20kg平台质量。'),
    ('是否重复补偿？','不增加重力；明确区别参考动态FF与速度误差PD，实际配对及静止归零验收通过。'),
    ('FF-V优于C4？','本轮T4/6/8/10的Z-RMSE均更低，其他维度另列，不能概括成全维更优。'),
    ('VA优于V？','四个正式时间均进一步降低Z-RMSE。'),
    ('VA-Lead优于原fourDlead？','Z误差明显更低；VA-Lead不稳定优于VA，开发按既定规则选VA。'),
    ('三档改善一致？','T4/T6/T10一致，独立T8同方向。'),
    ('预算不足？',f'完整实验最大投影占比{full.budget_projection_fraction.max():.9g}；本矩阵没有明显预算瓶颈，不代表所有未来状态。'),
    ('X/Y偏差？','会增加。T4峰值由0.018800增到0.264924mm，约14.1倍；三维RMSE仍下降，但非目标退化必须诊断。'),
    ('姿态漂移？','SO(3)与Roll也增加到千分之几度量级，未触保护；不是所有方向精度都提升。'),
    ('实际力合理？',f'峰值{s["maxima"]["max_force_N"]:.6f}N，未触±400N饱和；全力变化与附加20N/s分开。'),
    ('5ms下有效？','见严格同T6对照，FF-VA仍低于原C4/fourDlead，无安全失败。'),
    ('10ms下有效？','同T6无载通过；只支持这条已验证工况，未测试10ms叠加载荷。'),
    ('−5.5N下能保持？','tau5ms联合7秒预检后3条完整15秒均完成，未降低门槛。'),
    ('哪些误差已解释？','大部分名义参考速度/加速度跟踪误差可由经典FF显著减少。'),
    ('未知部分？','严格配对时序差分保留滞后和负载的闭环额外影响；几何变参、非目标方向也未全部解决。'),
    ('最适合PPO任务？','先固定L50-T6 tau5ms及受限−5.5N负载，不一开始全参数随机。'),
    ('一维Fz？','仍适合首先验证主要Z残差，固定探针证明有限权限；不保证能学习或处理XY。'),
    ('归一化？','50mm宏观位移/slide与微米级误差分尺度，候选53维因果单帧，训练统计冻结。'),
    ('哪些随机化？','先已验证无载tau5/10和tau5负载，再逐项联合预检；10ms＋负载/质量增益未执行。'),
    ('是否值得下一阶段？',rl['decision']+'；物理动作筛选通过，但先诊断FF的XY/姿态代价及完整任务强基线，再决定是否启动有限PPO。')]
    qa='\n'.join(f'### {i}. {question}\n\n{answer}\n' for i,(question,answer) in enumerate(q,1))
    put('46_Phase2C3B验收报告.md',f'''# Phase 2C-3B 验收报告

日期：2026-11-05；工程`{ROOT}`；独立实验目录`{RUN}`。

## 结论与实际交付

本轮完成参考动态FF审计、9组开发、24组正式、6组滞后、3组联合前缀、3组完整负载和4组固定动作权限，共49组研究执行。其中46条完整轨迹完成，3条预检从未冒充完成任务；没有PPO/SAC/TD3/BC训练。历史{s['history_files_checked']}个文件SHA逐一保持，9条冻结基线完整状态与历史逐位相同。参数和方法选择在正式评价前冻结。

{table(summary)}

按开发集Z误差选{strong}，不是全维最优结论。T4下Z-RMSE从0.248003降到0.013300mm，但XY峰值由0.018800增至0.264924mm，三维RMSE从0.248195降到0.124887mm；FF后非Z误差占剩余三维平方误差约98.9%。不能把Z改善概括成全维精度提升。

FF主要解决名义参考动态误差，并不能据此证明RL比经典控制有收益。新任务设计判断：**{rl['decision']}**，先诊断新增非目标耦合，再决定是否进入有限PPO，见45号报告的待执行门禁。

## 测试、数据复算与视频

完整测试实际输出：`{testline}`。历史xfail仍单列，未计普通通过。新增测试包含解析参考导数、J力映射、零FF等价、共享幅值/限速、真动态、历史冻结、T8、RGB/状态一致、GUI开关一致性及数据复算。完整日志`results/phase2c3b_pytest.log`；预检9测试的先行日志独立保留。

49条保存的全采样率1ms物理日志重新算Z/3D/SO(3)、力峰值/变化率/平方积分和每链`Fcommand−FC0`预算，结果`{RUN/'independent_audit.json'}`，通过={s['independent_audit']}。输入异常与提前结束分开处理，预检没有下降阶段就存空值，不虚构成功。

视频`assets/videos/stewart_phase2c3b_dynamic_feedforward.mp4`，1920×1080/25fps、{v['decoded_frames']}帧、{v['seconds']:.2f}秒、2倍播放；5条现场RGB源均逐帧校验qpos SHA与积分状态一致，真实Z变化>49mm。不是重新写qpos生成的运动学录像，也不是PPO训练。视频显示六链行程/力、Z误差、参考v/a、FF请求、预算占用和结果。

GUI对照见`{RUN/'live_viewer_validation.json'}`。CPU墙钟受渲染影响，不改变实际积分。完整实验物理步{s['research_physical_steps']}，GUI两次额外16000，测试另计；本轮PPO决策数0、学习更新数0。

## 可直接运行的真实窗口命令

{command}

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

{qa}

## 剩余限制与未执行事项

{scope}

未实施未来50mm Gym环境、新Actor、观测归一化/奖励验收、PPO训练、20/50Hz PPO对照、跨种子学习评价、tau10ms＋负载、随机质量/增益、全6DoF大位移或实机测试。固定探针窗口改善不是学习成功，确定性研究完成率不是随机鲁棒成功率。旧3mm PPO权重保留，未在50mm上直接部署。

本轮到此暂停，等待审核，不自动进入下一阶段。
''')
    files={}
    paths=[ROOT/'configs/phase2c3b_feedforward_study.json']+list((ROOT/'src').rglob('*phase2c3b*'))+list((ROOT/'scripts').glob('*phase2c3b*.py'))+list((ROOT/'tests').glob('*phase2c3b*.py'))+list((ROOT/'results').glob('phase2c3b*.*'))+list((ROOT/'assets/plots').glob('phase2c3b*.png'))+[ROOT/'assets/videos/stewart_phase2c3b_dynamic_feedforward.mp4']+list((ROOT/'docs').glob('4[2-6]_Phase2C3B*.md'))
    paths += [ROOT/'src/control/dynamic_reference_feedforward.py',ROOT/'src/control/large_motion_feedforward_wrapper.py',ROOT/'src/workspace/feedforward_study.py',ROOT/'src/workspace/feedforward_metrics.py']
    paths+=list(RUN.rglob('*'))
    for path in sorted(set(paths)):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='phase2c3b_artifact_manifest.json':files[str(path.relative_to(ROOT))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    write_json(ROOT/'results/phase2c3b_artifact_manifest.json',dict(stage='phase2c3b',run=str(RUN),files=files,history_unchanged=s['changed_history_files']==[],no_learning=True));print('Reports42-46 and manifest generated',len(files),'files')

if __name__=='__main__':main()
