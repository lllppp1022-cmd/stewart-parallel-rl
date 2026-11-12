"""Final Chinese reports generated exclusively from completed experiment artifacts."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from phase2c9b_protocol import ROOT,RUN,SEEDS,METHODS,cfg,read,write_json,sha


def table(d,precision=7):
    def cell(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.{precision}g}'
        return str(v).replace('|','/').replace('\n',' ')
    rows=[d.columns,['---']*len(d.columns),*d.itertuples(index=False,name=None)]
    return '\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in rows)


def doc(n,title,text):
    (ROOT/f'docs/{n}_Phase2C9B{title}.md').write_text(f'# Phase 2C-9B {title}\n\n'+text.strip()+'\n',encoding='utf8')


def main():
    selection=read(RUN/'final_selection.json');n=selection['common_budget']
    summary=pd.read_csv(ROOT/'results/phase2c9b_training_summary.csv')
    final=pd.read_csv(ROOT/'results/phase2c9b_generalization.csv')
    updates=pd.read_csv(ROOT/'results/phase2c9b_update_events.csv')
    episodes=pd.read_csv(ROOT/'results/phase2c9b_episode_returns.csv')
    effects=pd.read_csv(ROOT/'results/phase2c9b_paired_effects.csv')
    actions=pd.read_csv(ROOT/'results/phase2c9b_action_audit.csv')
    authority_by_mass=actions.groupby(['method','mass_kg'])[['request_rms_Nm','accepted_rms_Nm','projection_fraction','budget_peak_N']].mean().reset_index()
    sensitivity=pd.read_csv(ROOT/'results/phase2c9b_history_ablation.csv').groupby('variant')[['normalized_action_rms_change','maximum_action_change']].agg(['mean','max'])
    sensitivity.columns=['_'.join(c) for c in sensitivity.columns];sensitivity=sensitivity.reset_index()
    authority=pd.read_csv(ROOT/'results/phase2c9b_authority_audit.csv')
    result=read(ROOT/'results/phase2c9b_final_decision.json')
    critic_summary=updates.groupby(['method','seed']).agg(
        value_loss_max=('train/value_loss','max'),explained_variance_last=('train/explained_variance','last'),
        target_std_max=('value_target_std','max'),bias_last=('prediction_minus_target_bias','last')).reset_index()
    gae_error=pd.read_csv(ROOT/'results/phase2c9b_gae_recomputation.csv').max_error.max()
    preflight=pd.DataFrame([read(p) for p in (RUN/'preflight').rglob('metrics.json')]+[read(p) for p in (RUN/'preflight_z50').rglob('metrics.json')])
    preflight.to_csv(ROOT/'results/phase2c9b_mass_preflight.csv',index=False)
    totals=summary.groupby(['method','seed']).agg(decisions=('new_decisions','sum'),physics=('physical_steps','sum'),complete=('complete_episodes','sum'),fragments=('fragments','sum'),failures=('failures','sum'),wall_s=('wall_s','sum')).reset_index()
    optimization=updates.groupby(['method','seed']).agg(ppo_updates=('decisions','size'),adam_steps=('adam_steps','sum')).reset_index()
    totals=totals.merge(optimization,on=['method','seed'])
    assert len(totals)==6 and totals.decisions.eq(n).all() and totals.physics.eq(n*20).all()
    networks=[]
    for m in METHODS:
        for seed in SEEDS:networks.append(dict(method=m,seed=seed,**read(RUN/f'training/{m}/seed{seed}/from_000000/network.json')))
    keys=['load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','omega_peak_rad_s','z_rmse_mm','xyz_rmse_mm','xy_peak_mm','force_rate_N_s','force_squared_integral_N2_s']
    means=final.groupby('method')[keys].mean().reset_index()
    completion=final.groupby('method').agg(trials=('success','size'),complete_success=('success','sum')).reset_index()
    means=means.merge(completion,on='method')
    indexed=means.set_index('method');h=indexed.loc['history_5frame'];s=indexed.loc['single_frame'];a=indexed.loc['adaptive']
    comparison_text=(f"全任务等权均值中，历史相对单帧的加载IAE变化为{100*(h.load_iae_deg_s/s.load_iae_deg_s-1):+.3f}%"
        f"，卸载IAE变化为{100*(h.unload_iae_deg_s/s.unload_iae_deg_s-1):+.3f}%"
        f"，XYZ-RMSE变化为{100*(h.xyz_rmse_mm/s.xyz_rmse_mm-1):+.3f}%（正号代表误差增加）。"
        f"历史相对经典适应的加载IAE变化为{100*(h.load_iae_deg_s/a.load_iae_deg_s-1):+.3f}%"
        f"，卸载IAE变化为{100*(h.unload_iae_deg_s/a.unload_iae_deg_s-1):+.3f}%。"
        "均值存在加载/卸载取舍，不能用一个较低的均值证明逐任务一致收益。")
    per_seed=final[final.method.isin(METHODS)].groupby(['method','seed'])[keys].mean().reset_index()
    means.to_csv(ROOT/'results/phase2c9b_method_metrics.csv',index=False)
    per_seed.to_csv(ROOT/'results/phase2c9b_seed_metrics.csv',index=False)
    tests={}
    for label in ('historical_tests','historical_tests_unrestricted','new_tests'):
        path=RUN/(label+'.xml')
        if path.exists():
            cases=list(ET.parse(path).getroot().iter('testcase'))
            fail=[dict(name=x.attrib['name'],message=(x.find('failure') if x.find('failure') is not None else x.find('error')).attrib.get('message','')) for x in cases if x.find('failure') is not None or x.find('error') is not None]
            tests[label]=dict(total=len(cases),passed=len(cases)-len(fail),failures=fail)
    history=read(RUN/'history_verification.json')
    safety=dict(training=read(RUN/'training_audit.json'),preflight_trials=len(preflight),preflight_success=int(preflight.success.sum()),
        formal_training_failures=int(totals.failures.sum()),final_trials=len(final),final_complete_success=int(final.success.sum()),
        max_actual_force_N=float(final.force_peak_N.max()),max_shared_add_N=float(final.add_peak_N.max()),
        max_shared_slew_N_s=float(final.add_rate_N_s.max()),min_proxy_gap_m=float(final.clearance_m.min()),
        max_weld_translation_m=float(final.weld_translation_m.max()),max_weld_angle_rad=float(final.weld_angle_rad.max()),
        physical_metrics=read(RUN/'physical_validation.json'),history=history,tests=tests,
        partial_geometry_only=True,no_continuous_or_hardware_safety_claim=True)
    native=pd.read_csv(ROOT/'results/phase2c9b_native_contact_audit.csv')
    safety.update(native_contact_peak=int(native.native_contact_peak.max()),warning_peak=int(native.warning_peak.max()),
        native_force_saturation_steps=int(native.native_force_saturation_steps.sum()))
    if (RUN/'abandoned_thread_evaluation/attempt.json').exists():
        safety['incomplete_engineering_attempt']=read(RUN/'abandoned_thread_evaluation/attempt.json')
    write_json(ROOT/'results/phase2c9b_safety_audit.json',safety)
    prefix=f'日期2026-11-12，Asia/Shanghai。独立归档 `{RUN}`。数值从实际CSV/JSON/NPZ重算。'
    doc(89,'未知质量安全与控制权限审计',f'''
{prefix}

68次分层及有界随机预检，完成{int(preflight.success.sum())}/68。质量20、20.3、20.6、21.2、21.5、21.6、21.8、22 kg的Home、纯Z和联合姿态均分别检查。初始纯Z层取自B0为30 mm，随后在训练前单独补了8次纯50 mm B1检查，原记录完整保留。训练、开发和最终外载组合还分别使用零残差及固定种子有界随机序列验证，随机动作包含正负、双轴和连续保持；这仍是有限序列验证。

名义质量始终20 kg。实际Plant按固定形状密度比例缩放质量与主惯量，质心、执行器增益、5 ms合成滞后等参数固定。`phase2c9_plant.json`是每回合修改后实际与名义的参数证据，通用早期plant配置只表示创建时状态。不存在运行中通过qpos模拟质量变化。

{table(preflight[['task_id','mass_kg','method','success','so3_peak_deg','load_iae_deg_s','unload_iae_deg_s','xyz_rmse_mm','add_peak_N','add_rate_N_s']])}

动作因果探针采用相同确定性前缀回放至4.4 s，再执行0或±0.0125 N·m轴向及复合动作。各分支前4400步完整native积分状态逐位相同，控制历史一致。18个干预结果如下，原生驱动力与平台角速度均直接从真实积分日志读取。

{table(authority.drop(columns=['source']))}

存在可测物理作用不保证该方向能改善整个任务。接受力矩表示同状态共享预算的边际输出，不等于独立零RL轨迹之差。实际力变化率包含变化的C0，单独统计附加补偿20 N/s。所有原工作区、Jacobian、weld和几何代理保护保留。有限代理与离散仿真不构成完整碰撞或实机认证。
''')
    doc(90,'单帧与历史观测PPO环境',f'''
{prefix}

复用StewartAdaptiveEnv和HistoryObservation，58维当前帧逐字段沿用原归一化，历史为[t−4,t−3,t−2,t−1,t]、290维，20 ms间隔、首尾80 ms。初始重复当前合法帧，reset清空缓冲。当前观测包含因果观测器内部反馈和前一动作，因此“单帧”仍具有传统控制器的内部记忆。历史策略增加的是Actor可见的4个过去帧。

零动作、非零动作、reset历史顺序及22 kg Plant和20 kg控制模型隔离均完成真实测试。恢复验证在显式预算边界保存优化器、任务RNG和全局随机状态，重复重载分支的动作、物理轨迹与最终参数逐位相等。测试用1536决策单列，不计正式训练。任意回合中途崩溃恢复尚不支持。

两种Actor/Critic隐藏层64×64，原始V2奖励，原512 rollout、64 batch、10 epoch、lr0.0003及历史gamma/GAE。实际参数量如下。输入维度不同造成参数量差异，不称完全同容量。

由初始checkpoint反读的有效设置还包括vf_coef=0.5、normalize_advantage=True、clip_range_vf=None；完整实际设置见 effective_ppo_settings.json。单帧16069个可训练参数，五帧45765个。

{table(pd.DataFrame(networks))}

训练质量与任务来自独立PCG64采样器。所有种子两方法任务序列逐条核对，Box裁剪和真实随机动作由原回调验证。质量标签与实际驱动力隐状态只用于日志，不进入58维帧。随机策略训练和确定性评价分开。
''')
    doc(91,'真实PPO训练与学习曲线',f'''
{prefix}

{table(totals)}

正式合计{int(totals.decisions.sum())}决策、{int(totals.physics.sum())}物理步、{len(updates)}次PPO更新、{int(updates.adam_steps.sum())}次Adam。完整回合{int(totals.complete.sum())}，预算片段{int(totals.fragments.sum())}，保护失败{int(totals.failures.sum())}。各回合任务、质量、原始动作、实际接受补偿与1 ms状态保留。预算片段不计作完整成功或失败。

两方法优化器配置与最大epoch预算相同。原target_kl=0.03保留，部分更新触发既有KL提前退出，因此实际Adam步数不是每次都达到80；表中记录真实执行量，没有把最大预算当作实际优化次数。

8192决策Pilot结束时，当前未结束回合按预算片段归档；共同门禁通过后，两方法都以新回合reset续训，并恢复优化器、全局随机状态及独立任务采样器。阶段内跨512步rollout的回合不被截断。这里是匹配的显式阶段边界协议，不称为任意回合中途无缝继续。

Pilot判定原件如下。

`{read(RUN/'pilot_gate.json')}`

最终共同预算为{n}。每个种子纳入全部统计，终端模型统一选择，没有用最终测试挑模型。0/8192以及实际执行到的16384/24576/32768均保存；未运行的阶段不会创建checkpoint。原生模型SHA见 final_selection.json。

![实际训练曲线](../assets/plots/phase2c9b_learning_curves.png)

![固定开发任务](../assets/plots/phase2c9b_fixed_task_learning.png)

训练回报因质量/任务变化而不能视为相同任务性能曲线。固定开发集在相同checkpoint上确定性执行，逐个任务等权。全部正式奖励分量逐物理步复算，所有rollout GAE和bootstrap独立核验。Value Loss大小与物理控制收益分开解释。

{table(critic_summary)}

上述Value Loss是SB3记录的实际优化批次损失，不能单凭最大值判断发散；应结合不同任务组成、价值目标的尺度、预测偏差和Explained Variance。每次更新的目标/预测/优势5%、50%、95%分位数和实际任务组成保存在update_events.csv。按SB3 float32运算独立复算全部GAE的最大绝对误差为{gae_error:.9g}。网络更新和价值拟合不等于闭环控制优于经典补偿，最终按物理IAE及代价护栏判断。
''')
    effect_table=effects[effects.comparator.isin(['single_frame','adaptive'])]
    recovery=pd.read_csv(ROOT/'results/phase2c9b_recovery_status.csv')
    doc(92,'经典控制与未知质量泛化评价',f'''
{prefix}

最终8个任务×9策略，共{len(final)}次真实闭环，完成{int(final.success.sum())}次。每任务传统控制3个方法各运行一次，两个PPO方法各3个独立训练种子。评价参考、质量、外载和控制限制配对；独立配置训练前冻结，基线安全预检已执行，不能称完全未观察过的物理场景。未见质量20.3、21.5、21.8 kg，20 kg作名义回归。参考几何缓存沿用已审核资产，最终参考/外载组合和时序与训练不同。

全部测试质量位于训练质量范围内，是未见数值的插值评价。质量在reset时设置并整回合固定，外扰在运行中突加与卸载。本轮恢复时间对应未知固定质量下的外力矩事件，不能解释为回合中质量突然改变后的辨识或适应速度。

经典适应补偿在开发任务选定 `{read(RUN/'adaptive_selection.json')['gains']}`。候选积分增益0、10、30，按加载+卸载IAE以及XYZ非劣护栏选择。积分变化在50 Hz处理，控制命令仍经原物理预算器。未读取隐藏质量。

各方法均值（PPO再对三个种子等权，传统不重复成三次）如下。

![全部质量与训练种子](../assets/plots/phase2c9b_generalization_curves.png)

{table(means)}

逐训练种子平均如下。

{table(per_seed)}

历史策略配对改善百分比，正值改善、负值退化。

{table(effect_table)}

SO(3)误差用旋转矩阵相对旋转的对数模长，加载/卸载各取事件后1000个1 ms样本。未完整覆盖窗口时明确记录截断；提前失败的低误差不计改善。XYZ_RMSE是三维误差范数均方根，力平方积分不等于能耗。恢复指标保留未越阈值与右删失状态，不强制写0秒。

卸载后0.01度阈值恢复状态统计如下。只有两个配对方法均为recovered时才计算恢复时间差，负毫秒代表历史策略更快；其余情况不排序，见 paired_effects.csv 的对应列。

{table(recovery)}
''')
    doc(93,'历史信息消融与策略行为',f'''
{prefix}

阶段结果为 **{result['status']}**。匹配训练后历史相对单帧的收益支持状态为 `{result['evidence']['single_frame']}`，相对经典适应为 `{result['evidence']['adaptive']}`。规则要求至少两个种子各在至少两个未见质量任务同时改善加载与卸载IAE各1%，并满足全任务安全和代价护栏。

{comparison_text}

满足整套逐任务条件的种子：相对单帧{result['qualified_seeds']['single_frame']}，相对经典适应{result['qualified_seeds']['adaptive']}。单个种子的局部表现不能升级为跨种子适应收益。

{table(actions)}

按实际质量汇总控制权限：

{table(authority_by_mass)}

21.5及21.8 kg达到既有5 N共享预算，投影比例比20 kg更高；最高质量下的加载IAE也明显增加。这些日志支持“基础补偿占用与受限残差权限需要一起研究”的诊断，但尚不能认定预算投影是唯一原因。本轮没有扩大动作范围或放宽预算，也不能把请求力矩全部当成已施加力矩。

事件20 ms、100 ms和1秒内的请求偏置及SO(3)IAE见 phase2c9b_action_events.csv。1 ms行为图位于实验目录behavior_plots。请求与接受之差必须结合瞬态限速和共享附加力解释，RMS比值不能作为统一权限系数。不根据相关性宣称准确辨识质量或外扰。

对全部冻结历史策略的真实评价观测进行了只读敏感性诊断，比较原顺序、重复当前帧和反序历史，见 phase2c9b_history_ablation.csv。原始顺序的预测先与执行动作核对，全部诊断没有新增物理步、没有更新模型。重复和反序历史可能是离分布输入，动作变化不能当成闭环收益或物理因果证明。本轮该项是离线敏感性，闭环历史替换如未执行就保留为未完成项。

{table(sensitivity)}

策略输出对历史替换敏感，说明网络确实使用了历史特征；实际闭环却未满足历史收益门槛，因此“使用历史”与“历史改善控制”必须分别判断。

匹配容量对照状态：{'已触发，必须结合capacity_control_validation.json补充审核' if result['matched_capacity_control_required'] else '未执行：最终结果未满足一致历史收益门槛，因此没有追加训练'}。候选58维[124,123]网络45756参数与历史网络45765参数只差9个，作为条件协议在最终结果汇总前冻结。本轮不声称排除了网络参数量差异。
''')
    tests_table=pd.DataFrame([dict(run=k,total=v['total'],passed=v['passed'],failed=len(v['failures'])) for k,v in tests.items()])
    videos=[]
    for kind in ('unknown_mass','history_training','single_vs_history'):
        p=RUN/f'video_{kind}.json'
        if p.exists():
            v=read(p);videos.append(dict(file=Path(v['path']).name,frames=v['frames'],seconds=v['seconds'],sha=v['sha']))
    doc(94,'最终验收报告',f'''
{prefix}

**{result['status']}**。工程训练完成与历史控制收益分别判定。正式完成{int(totals.decisions.sum())}决策、{int(totals.physics.sum())}真实物理步，最终评价{len(final)}次，其中{int(final.success.sum())}次完整。历史是否比单帧与经典适应更好的结论，见报告92/93逐任务及逐种子结果。

{comparison_text}

{table(means)}

{table(totals)}

历史冻结核验{history['checked']}文件、变化{len(history['changes'])}个。测试执行统计如下，失败详情和首次沙箱错误均保留。

{table(tests_table)}

原Phase 0“环境不得安装Torch/SB3”断言与后续训练环境冲突，保留失败。首次沙箱运行pytest临时目录产生访问错误，使用独立目录重跑，并明确区分执行环境错误与代码失败。新增测试对真实模型、物理日志、GAE和全种子预算作核验，没有删除历史失败。

全量审计首次因float64标量GAE递推与SB3 float32向量递推的舍入差异失败（最大超差约8.6×10⁻⁵）；失败日志full_audit.log保留。随后按训练库实际float32逐步运算重算，未修改任何训练权重、奖励或已发生的样本。

提前检查点评价工具最初启动了三条同进程线程试验，随后终止并改为独立进程执行。三条未完成启动记录保存在 abandoned_thread_evaluation，未纳入开发指标或策略选择。该工程尝试的全部实际步数不能从终止记录精确恢复，因此不声称包含这部分的全工程积分总数精确已知。正式PPO、完成的预检和评价均有独立完整账本。另一次ProcessPool尝试受Windows沙箱命名管道访问限制而未开始物理积分，日志保留；最终使用独立Python子进程。

{table(pd.DataFrame(videos)) if videos else '视频验收尚未归档。'}

视频来自训练现场或冻结策略真实积分，逐帧时间和qpos SHA与原日志对应。训练视频标记ACTUAL STOCHASTIC TRAINING，评价标记FROZEN DETERMINISTIC EVALUATION。机器人保持真实物理角度，曲线纵轴辅助显示微小误差。全部种子在最终比较表及比较视频中保留。

PowerShell打开冻结策略GUI，两个方法均在1000步自动验收模式检查与headless前缀一致。任务选择绑定实际质量、方向及加载卸载时序，只允许已通过评价的组合。

GUI顶部先选任务、控制器和训练种子，然后点击Reset new episode，选择才会作用于新的物理回合；状态栏ACTIVE始终显示正在执行的方法、种子和真实质量。Pause/resume暂停继续，拖动机器人画面调整相机，Close and archive或关闭窗口正常归档。这里是冻结策略实时动力学评价，不是新的训练；训练随机探索请观看现场训练视频或TensorBoard。

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c9b_policy.py --method history_5frame --seed 42 --task Final_m21.5_6
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c9b_policy.py --method single_frame --seed 42 --task Final_m21.5_6
```

TensorBoard实际日志为实验目录training，各方法/种子/预算阶段独立。完整开发曲线和奖励分量还可以从CSV重算。当前服务器地址及HTTP验收见 tensorboard_validation.json，固定端口6037。

访问 http://127.0.0.1:6037/#timeseries 。本轮服务实际读取{len(read(RUN/'tensorboard_validation.json')['runs'])}个run：两阶段训练日志12个及完整课程derived日志6个。推荐先选derived/history_5frame_seed42或对应单帧/其他种子，查看episodes/complete_return、fixed_development/load_iae_deg_s、fixed_development/unload_iae_deg_s及optimization/train/value_loss。五个固定开发点0、8192、16384、24576、32768已通过HTTP读取核验。不要把旧6031端口的其他阶段曲线当成本轮结果。训练已完成停止，图表保持不再增长属于正常现象。

```powershell
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m tensorboard.main --logdir results/runs/phase2c9b_20261112_v1/training --host 127.0.0.1 --port 6037
```

实际复算入口：analyze_phase2c9b.py training、analyze_phase2c9b_history.py、plot_phase2c9b.py。训练原始目录已存在，重复训练须使用新的独立命名空间；既有正式目录不会覆盖。

本轮仍只有有限轨迹、质量和随机种子。评估为固定策略利用历史反馈，评价阶段不更新网络。未执行实机实验，几何代理不完整，离散预检不提供连续安全保证。对任意回合中途崩溃的无缝恢复未作保证，显式预算边界恢复通过。

测量沿用理想位姿与支链状态反馈，没有在本轮增加传感器噪声。驱动滞后固定为5 ms合成一阶模型，质量采用均匀密度等效变化；对偏心附加载荷、未知液压动态或实机噪声的结果尚未验证。

明确未执行/未完成：闭环历史信息替换（本轮完成的是只读输入敏感性）；匹配容量对照{'需要补充，见条件触发状态' if result['matched_capacity_control_required'] else '未触发、未训练'}；任意回合中途的无缝恢复；实机与噪声泛化。历史环境隔离测试仍有1项失败，没有宣称全部历史测试通过。以上均不作为已完成实验。

本阶段结束后暂停，等待审核。
''')


if __name__=='__main__':main()
