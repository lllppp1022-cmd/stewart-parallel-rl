"""Generate Phase8 decisions and Chinese reports from completed real artifacts only."""
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from phase2c8_protocol import ROOT, RUN, SEEDS, cfg, read, write_json, sha
from gate_phase2c8 import KEYS


def table(d, digits=6):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return '—' if not np.isfinite(x) else f'{x:.{digits}f}'
        return str(x).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, d.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(d.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in d.itertuples(index=False, name=None)])


def main():
    train = pd.read_csv(ROOT / 'results/phase2c8_training_summary.csv')
    final = pd.read_csv(ROOT / 'results/phase2c8_controller_comparison.csv')
    dev = pd.read_csv(ROOT / 'results/phase2c8_checkpoint_evaluation.csv')
    critic = pd.read_csv(ROOT / 'results/phase2c8_critic_diagnostics.csv')
    samples = pd.read_csv(ROOT / 'results/phase2c8_training_samples.csv')
    episodes = pd.read_csv(ROOT / 'results/phase2c8_episode_returns.csv')
    actions = pd.read_csv(ROOT / 'results/phase2c8_action_audit.csv')
    events = pd.read_csv(ROOT / 'results/phase2c8_action_events.csv')
    offline = pd.read_csv(ROOT / 'results/phase2c8_reward_offline_audit.csv')
    gate = read(RUN / 'pilot_gate.json')
    selection = read(RUN / 'final_selection.json')
    history = read(RUN / 'history_after.json')
    recompute = read(RUN / 'final_metric_validation.json')
    runtime = read(RUN / 'runtime.json')
    assert history['passed'] and recompute['passed'] and len(final) == 32 and len(train) == 6
    assert not gate['passed'] and selection['budget'] == 8192
    for path, digest in read(RUN / 'config_freeze.json').items():
        assert sha(path) == digest
    for model in selection['models'].values():
        assert sha(model['path']) == model['sha']

    metric_keys = ['load_iae_deg_s', 'unload_iae_deg_s', 'so3_peak_deg', 'omega_peak_rad_s',
                   'xyz_rmse_mm', 'z_rmse_mm', 'xy_peak_mm', 'force_rate_N_s', 'force_squared_integral_N2_s']
    mean = final.groupby('method')[metric_keys].mean()
    per_seed = final[final.method.isin(['v2_control', 'v3'])].groupby(['method', 'seed'])[metric_keys].mean().reset_index()
    pair_rows = []
    for r in final[final.method.isin(['v2_control', 'v3'])].to_dict('records'):
        for base in ['baseline', 'classical50'] + (['v2_control'] if r['method'] == 'v3' else []):
            b = final[(final.task_id == r['task_id']) & (final.method == base)]
            if base == 'v2_control':
                b = b[b.seed == r['seed']]
            b = b.iloc[0]
            row = dict(method=r['method'], seed=r['seed'], task_id=r['task_id'], comparator=base)
            for k in metric_keys:
                row[k + '_improvement_pct'] = 100 * (1 - r[k] / b[k])
            row['all_cost_guards'] = all(r[k] <= b[k] * 1.1 + 1e-12 for k in KEYS)
            row['safe_complete'] = bool(r['success'])
            row['both_iae_improve_1pct'] = bool(row['load_iae_deg_s_improvement_pct'] >= 1 and row['unload_iae_deg_s_improvement_pct'] >= 1)
            pair_rows.append(row)
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(ROOT / 'results/phase2c8_paired_effects.csv', index=False)
    counts = []
    for (method, seed), d in paired[paired.comparator == 'classical50'].groupby(['method', 'seed']):
        counts.append(dict(method=method, seed=int(seed), tasks_both_iae_improve_1pct=int(d.both_iae_improve_1pct.sum()),
                           all_cost_guards=bool(d.all_cost_guards.all()), all_safe=bool(d.safe_complete.all()),
                           qualifies=bool(d.both_iae_improve_1pct.sum() >= 2 and d.all_cost_guards.all() and d.safe_complete.all())))
    counts = pd.DataFrame(counts)
    success = bool(counts[(counts.method == 'v3') & counts.qualifies].shape[0] >= 2)
    assert not success
    partial = bool(paired[(paired.method == 'v3') & (paired.comparator == 'classical50')].both_iae_improve_1pct.any())
    status = 'PHASE 2C-8 PARTIAL SUCCESS' if partial else 'PHASE 2C-8 NEGATIVE'
    counts.to_csv(ROOT / 'results/phase2c8_seed_success_criteria.csv', index=False)
    per_seed.to_csv(ROOT / 'results/phase2c8_seed_metrics.csv', index=False)
    mean.to_csv(ROOT / 'results/phase2c8_method_metrics.csv')
    costs = []
    for method, d in samples.groupby('method'):
        for name in ['orientation', 'omega', 'xyz', 'action', 'slew']:
            a = -d['reward_' + name]
            costs.append(dict(method=method, component=name, mean_cost=float(a.mean()), p95_cost=float(a.quantile(.95)),
                              maximum_cost=float(a.max()), total_cost=float(a.sum()), fraction=float(a.sum() / -d.reward.sum())))
    costs = pd.DataFrame(costs)
    costs.to_csv(ROOT / 'results/phase2c8_reward_component_distribution.csv', index=False)
    progress = []
    for (method, seed), d in dev[dev.method.isin(['v2_control', 'v3'])].groupby(['method', 'seed']):
        d = d.groupby('decisions')[['common_v2_return', 'load_iae_deg_s', 'unload_iae_deg_s', 'xyz_rmse_mm']].mean()
        p = dict(method=method, seed=int(seed), common_v2_return_initial=d.loc[0, 'common_v2_return'],
                 common_v2_return_final=d.loc[8192, 'common_v2_return'])
        for k in ['load_iae_deg_s', 'unload_iae_deg_s', 'xyz_rmse_mm']:
            p[k + '_improvement_pct'] = 100 * (1 - d.loc[8192, k] / d.loc[0, k])
        p['return_up_but_any_iae_worse'] = bool(p['common_v2_return_final'] > p['common_v2_return_initial'] and
                (p['load_iae_deg_s_improvement_pct'] < 0 or p['unload_iae_deg_s_improvement_pct'] < 0))
        progress.append(p)
    progress = pd.DataFrame(progress)
    progress.to_csv(ROOT / 'results/phase2c8_development_progress.csv', index=False)
    cs = critic.groupby(['method', 'seed']).agg(value_loss_max=('value_loss', 'max'), EV_median=('explained_variance', 'median'),
          negative_EV_updates=('explained_variance', lambda x: int((x < 0).sum())),
          mean_prediction_minus_target=('value_bias_prediction_minus_target', 'mean'),
          last_prediction_minus_target=('value_bias_prediction_minus_target', 'last')).reset_index()
    cs.to_csv(ROOT / 'results/phase2c8_critic_summary.csv', index=False)
    peak_rows = critic.loc[critic.groupby(['method', 'seed']).value_loss.idxmax(),
                          ['method', 'seed', 'update', 'stage', 'value_loss', 'target_mean', 'target_std', 'value_mean', 'task_composition']]

    physics = Counter()
    for p in RUN.rglob('episode_*/summary.json'):
        group = p.relative_to(RUN).parts[0]
        if group.startswith('pytest'):
            continue  # Regression tests use several historical log formats, not a uniform budget ledger.
        physics[group] += read(p).get('physical_steps', 0)
    assert physics['training'] == int(train.physical_steps.sum())
    tests = {}
    for name in ['historical', 'new']:
        suites = ET.parse(RUN / f'{name}_tests.xml').getroot()
        cases = list(suites.iter('testcase'))
        failures = [dict(name=c.attrib['name'], classname=c.attrib.get('classname'), text=c.find('failure').text)
                    for c in cases if c.find('failure') is not None]
        tests[name] = dict(total=len(cases), passed=len(cases) - len(failures), failures=failures)
    cap_steps = sum(read(p)['diagnostics']['cap_steps'] for p in (RUN / 'training').rglob('episode_*/summary.json'))
    safety = dict(formal_training=dict(runs=len(train), decisions=int(train.new_decisions.sum()), physics_steps=int(train.physical_steps.sum()),
                                      complete_episodes=int(train.complete_episodes.sum()), budget_fragments=int(train.fragments.sum()),
                                      failures=int(train.failures.sum()), reward_cap_steps=cap_steps),
                  development=dict(trials=len(dev), complete_success=int(dev.success.sum())),
                  final=dict(trials=len(final), complete_success=int(final.success.sum()), protection_failures=int(final.failure.notna().sum()),
                             native_force_saturation_steps=int(final.native_force_saturation_steps.sum()),
                             max_actual_force_N=float(final.force_peak_N.max()), max_shared_add_N=float(final.add_peak_N.max()),
                             max_shared_add_rate_N_s=float(final.add_rate_N_s.max()), max_actual_force_rate_N_s=float(final.force_rate_N_s.max()),
                             min_proxy_clearance_m=float(final.clearance_m.min()), max_weld_translation_m=float(final.weld_translation_m.max()),
                             max_weld_angle_rad=float(final.weld_angle_rad.max()), max_jacobian_condition=float(final.condition_peak.max()),
                             reward_cap_fraction_max=float(final.cap_fraction.max())),
                  declared_limits=dict(rl_torque_per_axis_Nm=.05, shared_add_per_leg_N=5, shared_add_slew_N_s=20, native_force_N=400),
                  physical_budget_by_group=dict(physics), archived_research_physics_total=sum(physics.values()),
                  physics_total_excludes_heterogeneous_historical_regression_tests=True,
                  final_metric_audit=recompute, history_audit=history, tests=tests,
                  limitation='Simulation protection and partial proxy geometry queries; no continuous or hardware safety guarantee')
    write_json(ROOT / 'results/phase2c8_safety_audit.json', safety)
    decision = dict(status=status, scientific_conclusion='局部改善，但无跨种子、跨任务稳定超过强经典控制的证据；V3共同开发门禁失败。',
                    pilot_gate_passed=False, formal_decisions_per_run=8192, formal_training_decisions=int(train.new_decisions.sum()),
                    formal_training_physics_steps=int(train.physical_steps.sum()), final_trials=len(final), final_all_safe=bool(final.success.all()),
                    success_rule=cfg('training')['final_success'], seed_criteria=counts.to_dict('records'),
                    extension_32768='NOT EXECUTED: failed common development gate',
                    return_norm_training='NOT EXECUTED: failed reward admission; wrapper validation only',
                    Stage_C_training='NOT EXECUTED', optional_2p4_Nm='NOT EXECUTED', forbidden_4p8_Nm='NOT EXECUTED',
                    no_lstm_transformer_sac_td3=True, next_step='Pause for review; no automatic budget expansion',
                    checkpoint_selection=selection, test_summary={k:dict(total=v['total'], passed=v['passed'], failed=len(v['failures'])) for k,v in tests.items()},
                    historical_files_unchanged=history['passed'], no_statistical_significance_claim=True)
    write_json(ROOT / 'results/phase2c8_final_decision.json', decision)

    def doc(name, text):
        (ROOT / 'docs' / name).write_text(text.strip() + '\n', encoding='utf-8')
    prefix = f"日期：2026-11-09（Asia/Shanghai）；独立实验目录：`{RUN}`。所有数值由原始CSV/NPZ/JSON生成，不修改历史结果。"
    train_table = table(train[['method', 'seed', 'new_decisions', 'physical_steps', 'complete_episodes', 'fragments', 'updates', 'adam_steps', 'failures', 'wall_s']])
    gate_table = pd.DataFrame(gate['details'])
    for col in ['load_improvement', 'unload_improvement']:
        gate_table[col + '_pct'] = 100 * gate_table.pop(col)
    model_table = table(pd.DataFrame([dict(model=k, checkpoint=v['path'], SHA256=v['sha']) for k,v in selection['models'].items()]))
    mean_table = table(mean.reset_index(), 9)
    comparison = paired[paired.comparator == 'classical50'][['method','seed','task_id','load_iae_deg_s_improvement_pct','unload_iae_deg_s_improvement_pct','all_cost_guards','both_iae_improve_1pct']]
    tasks_table = table(pd.DataFrame([dict(task=t['id'], z_mm=t['reference']['z_m']*1000, roll_deg=t['reference']['roll_deg'],
              pitch_deg=t['reference']['pitch_deg'], rise_s=t['reference']['rise_s'], Mx_Nm=t['torque'][0], My_Nm=t['torque'][1],
              on_s=t['on'], off_s=t['off']) for t in cfg('evaluation')['final']]))
    doc('72_Phase2C8训练协议与随机数审计.md', f"""
# Phase 2C-8 训练协议与随机数审计

{prefix}

## 历史冻结与运行环境

历史冻结清单 `history_before.json` 共 {history['checked']} 个文件，事后逐文件SHA核验 {history['checked']} 个、变化 {len(history['changes'])} 个。旧模型、控制器、场景、Reward-V2、7/7V视频均保留。新四份配置的训练前SHA位于 `config_freeze.json`。实际执行实现SHA另存 `executed_implementation_v1.json`，此源码快照在Pilot结束后登记，不伪称为训练前时间戳。

Python：`{runtime['python']}`；版本 `{runtime['version']}`；系统 `{runtime['platform']}`。

{table(pd.DataFrame(runtime['packages'].items(), columns=['package','actual_version']))}

训练使用CPU、每进程1线程；当前Torch为CPU构建，CUDA available={runtime['cuda_available']}。不把用户历史GPU型号当作本次GPU训练证据。六个独立训练进程没有共享随机数或物理对象。

## 冻结的控制与PPO

物理1ms、传统控制1000Hz、PPO50Hz，每决策20个真实 `mj_step`；58维原因果观测；2维归一化高斯动作经裁剪后映射每轴±0.05N·m。C4、fourDlead和FF-VA未改；共享补偿相对C0每支链±5N、20N/s，原生±400N；原工作区、姿态、Jacobian和weld检查保留。实际驱动力变化率不等于“相对C0附加补偿”的20N/s限速，二者分别记录。

{table(pd.DataFrame([(k,str(v)) for k,v in cfg('training')['ppo'].items()], columns=['PPO设置','实际冻结值']))}

Actor/Critic分别64×64，初始Actor均值为零、log_std=-1；这是零均值随机探索，不是零动作训练。V2/V3同种子初始参数哈希一致，六组均从新初始模型出发。旧7期模型只在离线奖励审计中作为historical42，不能代替新V2匹配对照。

## RNG修复与恢复边界

任务抽样改为独立 `PCG64(SeedSequence([training_seed,8008]))`。Python/NumPy全局/Torch动作RNG与任务RNG分别保存。每episode记录task、外载、参考、阶段、抽样前后状态及Gym收到的seed。`gym.Env.reset(seed=...)`只处理自身RNG，不重置任务采样器。

SB3 load/set_env后清空VecEnv待应用seed，再恢复全局RNG；独立任务采样器也防止未来reset seed传播再次覆盖任务序列。Stage A/B分别4096步，在课程切换处显式结束预算片段并reset。A最后一次抽样后状态等于B首次抽样前状态，V2/V3每个种子的实际任务序列完全相同。

实际连续/恢复试验：A前缀512决策，连续B512与保存重载B512两条分支；总计1536测试决策、30720真实物理步。两分支动作、观测、奖励、rollout目标、物理状态、最终参数逐位相等。详见 `resume_validation.json`。此证明只覆盖声明的课程边界，**不支持任意回合中间崩溃后无缝恢复**；这种失败必须归档重做，不能静默reset冒充连续。

## 数据划分与实际预算

训练复用18个既有精确联合预检任务配置，A/B/C各阶段候选保留；新Dev为D8_roll、D8_pitch及D8_mix，其中D8_mix与旧Dev_mix物理任务一致，明确不是全新开发任务。Final四个任务的完整参考/外载组合与旧任务不同，配置训练前冻结。参考轨迹几何缓存复用旧审核资产，并非所有轨迹几何均首次出现。新Dev和Final先做14次零动作/有界随机联合安全预检；这属于准入检查，不是策略选择。所有最终PPO策略评价在 `final_selection.json` 冻结后才开始。

{tasks_table}

{train_table}

正式共 {int(train.new_decisions.sum())} 决策、{int(train.physical_steps.sum())} 物理步，{int(train.updates.sum())} 次PPO更新、{int(train.adam_steps.sum())} 次Adam。没有将恢复验证1536决策算进正式预算。每组10个完整回合、2个课程/预算片段；片段未被标成失败或完整回合。未发生训练保护失败。

共同开发门禁：至少2/3种子同时改善加载和卸载均值≥1%；每个种子IAE退化不得>10%；各任务SO3峰、角速度、Z/XYZ/XY、实际力变化率及力平方积分不得超过匹配V2的110%；全部安全完成。此门槛训练前冻结。

{table(gate_table)}

结果门禁失败，**六组均在8192停止，Stage C、32768扩展和ReturnNorm正式训练未执行**。没有只追加表现好的种子。保存所有终端checkpoint并进入最终评价，没有从最终测试选择best seed。

## 真实积分账本

{table(pd.DataFrame(physics.items(), columns=['运行类别','已归档物理步']))}

共 {sum(physics.values())} 步（含准入、奖励审计、开发、最终评价、恢复/零动作/归一化验证及两个GUI1000步测试）；历史pytest包含多种日志格式，未将其混入这个统一账本。关闭/渲染不增加隐藏物理步，重置初始化没有计作动态运动。

## 模型证据

{model_table}

每512决策的update模型、0/4096/8192 checkpoint、边界RNG/优化器均保留。96个rollout的GAE及bootstrap独立复算通过；未覆盖任意中途恢复这一限制仍保留。
""")
    offline_selected = offline[(offline.candidate == 'original') | (offline.candidate == "{'alpha': 0.5, 'beta': 0.5}")]
    doc('73_Phase2C8姿态抗扰RewardV3设计.md', f"""
# Phase 2C-8 姿态抗扰Reward-V3设计

{prefix}

## 公式与假设

H1：纯平方姿态误差与加载/卸载IAE的目标不完全一致。H1是可检验假设，不是本轮既定结论。

定义 `e = ||log(R_actual R_ref^T)|| / sR`，先将角误差转换为度，`sR=0.006度`，因此e无量纲。

`orientation_V2 = e²`

`orientation_V3 = 0.5 × (sqrt(e² + 0.01²) − 0.01) + 0.5 × e²`

平滑绝对值的epsilon=0.01是无量纲；零点代价与一阶导数均为零，有限；误差增大单调增加，不区分误差正负。中小误差段增加近似绝对误差贡献，但极接近零时仍是平滑二次形态；大误差保留平方代价。IAE按SO(3)模长积分，所以有目标对应依据；训练仍优化全时域折扣回报，**不等价于只优化加载/卸载各1秒IAE**。

其余代价完全沿用V2：

- `0.2 × sum((角速度误差 / 0.003 rad/s)²)`。
- `0.1 × sum((XYZ误差mm / [0.2,0.2,0.05]mm)²)`。
- `0.001 × sum((边际接受Mx/My / 0.05 N·m)²)`。
- `0.000025 × sum((边际接受力矩变化率 / 2.5 N·m/s)²)`。

五项非负cost总和在每物理步按同比例cap到100，决策reward为20物理步cost总和的负均值；安全失败另加原失败惩罚。未删除平滑代价、未增加控制权限。Reward只读取当前位姿、参考、速度、接受动作及其差分，不读取真实外载幅值、未来加载/卸载时刻。

## 选择规则和真实离线重计分

候选只有(alpha,beta)=(0.5,0.5)与(1,0.25)。训练前预指定等权版本通过有限性、单位、排序及失败检查后使用；第二候选仅离线诊断，不按哪个训练种子或Final Test更好来选权重。

同一旧开发任务真实执行baseline、classical50、historical42、positive/negative两轴±0.025N·m动作探针，共75000物理步。独立重计分V2最大误差为0；原始日志完整保留。

{table(offline_selected[['method','reward_version','episode_return','orientation_cost','omega_cost','xyz_cost','action_cost','slew_cost','cap_steps']])}

正负探针都是真实动态执行，不代表优质策略。五条轨迹均完成，没有伪造失败回合。分量级测试检验姿态、位置、动作及变化率增大时惩罚增长；对剩余0～800决策逐一验证原安全失败惩罚小于最坏未来100/步代价的折扣和，提前失败没有回报捷径。该测试是代数边界验证，不冒充真实失败轨迹。

## 训练中的实际分量

下表是49152真实决策的负reward分量，按每决策统计；fraction是同一奖励版本总cost份额，不能据此直接比较两方法物理性能。

{table(costs)}

正式训练cap触发 {cap_steps} 个物理步。完整72段日志逐物理步五项分量、逐决策回报均复算通过，记录在 `phase2c8_reward_recomputation.csv`。数学排序通过不保证综合多目标轨迹排名完全由IAE决定，存在姿态与XYZ/角速度/动作的真实取舍。

## 实证结论

V3更直接包含绝对姿态误差，但共同开发门禁未通过；最终有局部任务改善，也有加载/卸载互相恶化。不能将V3训练return的绝对值与V2比较。共同物理评价和统一只读V2计分详见报告74、76。本轮未进一步调alpha/beta，也未因Final结果改定义。
""")
    doc('74_Phase2C8奖励消融PPO训练.md', f"""
# Phase 2C-8 奖励消融真实PPO训练

{prefix}

六组真实训练已经完成，不是只创建代码。两方法相同初始化、任务RNG、场景、控制器、动作/观测、PPO配置、预算与安全约束，只改变声明的Reward公式。三种子为42、123、2026。

{train_table}

每组16次PPO更新（512决策一次），每次80次Adam，共7680次Adam；所有Actor/Critic变化范数>0，保存模型重载参数哈希一致。实采原始高斯动作、裁剪动作、接受动作、58维观测、奖励、价值目标均保存。随机训练与确定性评价分开。

## Pilot决策

{table(gate_table)}

没有任何一个V3种子同时达到开发加载和卸载各≥1%的改善，要求至少两个种子。虽然均安全且各代价护栏通过，仍然不晋级。V2和V3公平停在8192；继续32768、16384/24576/32768 checkpoint及ReturnNorm训练模型均**未产生、未执行**。

## 完整回合与课程片段

60个完整回合，12个课程切换/最终预算片段。片段仍保留原始物理日志与部分return，不加入完整episode return均值。一次rollout可跨episode，一个episode可跨512决策更新，`policy_version`逐物理步记录，不假设整个episode权重固定。

开发在0、4096、8192三个checkpoint对三个固定任务分别评价，另有传统对照，共60次真实闭环、900000物理步。所有评价0次梯度更新。统一V2只读计分用于跨奖励对比；这不改变训练reward。

相对各自初始策略的固定开发任务变化（各任务等权，正百分比为误差改善）：

{table(progress)}

其中 {int(progress.return_up_but_any_iae_worse.sum())} 组出现统一V2回报提高、但至少一个IAE恶化。这是实测的目标取舍，不只是理论可能性；V3 seed2026的统一回报和两个IAE也未同步改善。该表与“V3相对匹配V2”的晋级门禁是两个不同对照，不能混用。

## 学习曲线与TensorBoard

![训练奖励与分量](../assets/plots/phase2c8_training_reward.png)

![Critic与优势](../assets/plots/phase2c8_critic_learning.png)

![固定开发任务](../assets/plots/phase2c8_fixed_task_learning.png)

已启动并实际HTTP200验收：[本轮TensorBoard](http://127.0.0.1:6034/)。18个run：12个原始A/B事件流、6个从原始CSV复算的全程derived事件流。derived不是新训练；`actual/`是SB3实际事件。旧6031/6032/6033面板未改。

查看derived的 `development/*` 固定任务曲线，并同时看加载/卸载IAE、SO3/XYZ，不能只看训练return。累计episode reward每决策会累积，但完整episode return只在结束时更新；更新指标每512决策出现。训练已停止，所以刷新页面不会继续产生训练点。

## 已实际执行的命令

下面是六组已运行命令的形式，用于记录来源，不要在已有输出目录上重复运行：

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/train_phase2c8_reward_ablation.py --method v2_control --seed 42 --phase pilot
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/train_phase2c8_reward_ablation.py --method v3 --seed 42 --phase pilot
```

同样命令分别使用seed123、2026，共六组。独立复现实验需要先配置新的RUN和模型输出命名空间并重新冻结配置，当前脚本固定本轮归档路径，不提供会覆盖本轮结果的“一键重复训练”。不运行continue：它的冻结门禁为false。

只读重汇总/指标复算可执行：

```powershell
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/evaluate_phase2c8.py collect
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/audit_phase2c8_final.py
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/audit_phase2c8_gae.py
```

这些命令实际运行通过，只写本轮派生汇总/审计文件，不训练模型。每个种子的原始physics、integration_states、RGB帧、rollout NPZ、task_sequence和TensorBoard位于独立 `training/<method>/seed<seed>/<A或B>`。
""")
    doc('75_Phase2C8_Critic价值学习与回报尺度.md', f"""
# Phase 2C-8 Critic价值学习与回报尺度

{prefix}

## 实际价值拟合

{table(cs)}

`prediction_minus_target>0`表示对负回报预测得过于乐观，即低估未来代价。六组的平均偏差均为正，部分更新Explained Variance为负。Value Loss下降或绝对值较小不等于控制改善，也不是发散的充分条件。没有NaN/Inf、没有模型更新中断。

最大loss所在更新及实际任务：

{table(peak_rows)}

上述峰值需要结合初始Critic预测接近零、实际目标负值及其方差理解。不同任务强度/课程确实产生不同目标分布，但本轮不是仅在后期大位移困难任务出现loss；不能把全部误差归因于扰动强度，更不能仅凭loss宣布Critic发散。完整96行包含target/value/GAE均值、标准差、范围、5/95分位，及该rollout实际task_composition。

## 独立GAE审计

96个512决策rollout逐个使用原reward、value、episode_starts、last_values和最后done重算GAE、return；最大优势误差 {read(RUN/'gae_validation.json')['max_advantage_error']:.12g}、最大return误差 {read(RUN/'gae_validation.json')['max_return_error']:.12g}，符合float32累计复算 `rtol=3e-6, atol=5e-5`。这是数值一致性容差，不是安全门槛。TimeLimit bootstrap保留，真正失败终止不bootstrap；预算尾端是训练片段，不伪造time_limit。

`pre_value_mse`是优化前rollout预测对目标MSE；SB3 `value_loss`是训练中各minibatch/epoch loss平均，两者不必一致。EV基于该rollout优化前预测与目标，不是优化后独立验证EV。三者在CSV分开记录。

## H2与ReturnNorm状态

V3最大Value Loss约为V2的一半，但原始奖励定义和目标尺度也改变；不能将该数字解释成纯Critic稳定性提升，更不能当作姿态抗扰改进。

ReturnNorm准入要求V3先通过共同物理收益门禁；本轮未通过，所以 **V3-ReturnNorm PPO Pilot和长训练均未执行**。没有归一化模型、没有归一化物理优越性结论；H2因果实验未完成。

仅完成工程验证：VecNormalize `norm_obs=False`、`norm_reward=True`、gamma沿用原值、clip_reward=1e6、epsilon=1e-8；1500真实物理步验证原始reward/物理状态不变、统计保存重载一致、评价 `training=False/norm_reward=False` 冻结统计。58维物理观测不做第二次归一化。

VecNormalize改变训练reward、GAE和Critic目标，也影响Actor优化，**不是Critic-only**。未实施PopArt或单独Critic目标缩放。不用封装测试冒充归一化PPO训练。

## 后续合理性

本轮支持“仍存在价值拟合问题”这一观察，不支持“已证明价值尺度是控制瓶颈”。若后续经审核继续，先采用独立、对等预算的训练诊断，并预先规定物理非劣标准；不要只追求Value Loss更小或无限扩大预算。本阶段停止于现有门禁。
""")
    action_stats = actions.groupby('method').agg(requested_rms_mean_Nm=('requested_rms_Nm','mean'),
              projection_fraction_mean=('executed_projection_fraction','mean'), projection_fraction_max=('executed_projection_fraction','max'),
              restoring_sign_fraction_mean=('restoring_sign_fraction','mean')).reset_index()
    evstats = events[events.window.isin(['pre100','post100','recovery'])].groupby(['method','event','window']).agg(
              request_along_load_min_Nm=('requested_along_load_Nm','min'), request_along_load_max_Nm=('requested_along_load_Nm','max')).reset_index()
    doc('76_Phase2C8独立抗扰评价.md', f"""
# Phase 2C-8 独立闭环与策略行为评价

{prefix}

## 冻结评价协议

最终4任务、8控制器，共32次真实闭环、488000物理步。baseline=冻结C4+fourDlead+FF-VA，classical50=在相同二维力矩/共享预算权限内增加50Hz经典姿态反馈；六个PPO均用8192终端checkpoint、deterministic=True，没有选best seed。参考与外载参数训练前冻结；模型集合开发门禁后、最终策略评价前冻结。

{tasks_table}

加载和卸载IAE分别对事件对应的1000个1ms积分结果求和，单位deg·s。姿态用SO(3)旋转误差，不混用欧拉角直接相减；Roll/Pitch/Yaw峰值为世界旋转误差向量各轴绝对值峰值。XYZ_RMSE是三维误差范数的均方根，XY_peak是两个轴分量的最大绝对值。力平方积分不是能耗。

姿态恢复使用0.1度和0.01度两个band、0.5秒持续要求；若没有越band则 `not_triggered`，不是虚构0秒恢复。末尾不满足持续时间则右删失。完整CSV保留这两种状态。

## 真实结果

下表为各方法四任务等权均值；PPO再对三个独立训练种子等权。传统控制每任务执行一次，不把它复制三次当独立试验。IAE越小越好。总均值仅作摘要，结论依据逐任务/种子配对。

{mean_table}

每个种子四任务均值：

{table(per_seed,9)}

相对同权限经典反馈的改善百分比，正值改善、负值退化：

{table(comparison,4)}

冻结成功门槛：每个合格种子在至少两个任务上加载/卸载均改善≥1%，所有任务安全且SO3峰、角速度、Z/XYZ/XY、力变化率、力平方积分≤classic的110%；至少两个种子合格。

{table(counts)}

V3每个种子只有一个任务同时达到两项IAE改善，没有种子达到两个任务，因此不满足SUCCESS。阶段判定为 **{status}**：局部收益存在，但跨种子/任务不一致；不是宣称整体算法成功。V3共同开发门禁本身是负结果。

## 实际安全和复算

32/32完整安全完成，native饱和步数 {safety['final']['native_force_saturation_steps']}；最大实际力 {safety['final']['max_actual_force_N']:.6f}N；相对C0附加力最大 {safety['final']['max_shared_add_N']:.6f}N、变化率最大 {safety['final']['max_shared_add_rate_N_s']:.9f}N/s。实际力变化率最大 {safety['final']['max_actual_force_rate_N_s']:.6f}N/s包含随状态变化的C0，不应误报为附加补偿20N/s越限。

最小实验性几何代理距离 {safety['final']['min_proxy_clearance_m']:.9f}m；最大weld平移残差 {safety['final']['max_weld_translation_m']:.9g}m，角残差 {safety['final']['max_weld_angle_rad']:.9g}rad。代理距离不是完整连续碰撞认证；无保护事件也不等于实机安全认证。

32条原始1ms日志独立重算每条24个指标（共768项）及恢复状态，通过；最大误差 {recompute['max_absolute_error']:.9g}。四任务每个控制器的参考数组逐位相同，真实外载重建差异最大 {recompute['max_input_roundoff']:.9g}N/N·m，来自坐标变换浮点舍入，低于1e-12数值容差。第一次将该重建力矩要求逐位相等的审计失败日志保留为 `phase2c8_final_metric_audit_first_attempt.log`；没有修改物理数据或安全门槛。

## Actor加载/卸载行为

四任务×六PPO策略共24组，每组全时域、加载±100ms、卸载±100ms共72张图，位于本轮 `behavior_plots/`。事件局部保留1ms采样，灰线是20ms动作决策边界；全程图绘图下采样不改变原始日志。

{table(action_stats)}

{table(evstats)}

存在加载前和卸载后持续请求偏置，某些任务减少加载误差却增加卸载误差。`restoring_sign_fraction`只统计接受力矩与姿态误差内积为负的比例，忽略速度/前馈/瞬态，因此不是正确控制比例。不能把残差方向简单等同于识别外载。

首次日志请求变化在事件0～20ms附近；事件恰位于20ms网格时，0ms的Actor动作可能使用外载施加前状态，这是预先更新的请求而非即时感知外载。该列明确 `not_proven_causal_latency=True`，不宣称0ms扰动观测或准确估计未知力矩。

实际命令由原共享预算器处理。独立重算 `C0 + clip(传统附加 + 映射RL, 共享幅度/限速/native上下界)` 与记录一致；无RL同状态反事实也一致，误差≤1e-9N。`marginal accepted`是相同实际历史和预算状态下有无RL的命令差映射，**不等于独立无RL轨迹上的力矩差**。RMS接受/请求比不能当成统一“控制权限利用率”。平均投影步数很少不代表加载瞬间完全无限制，也不能仅凭共享预算存在就归因学习失败。

所有请求/接受、真实六缸力、姿态/角速度、reward分量及共享预算曲线保留。这里有相关行为和局部闭环作用证据，没有证明在线辨识未知外载或未知执行器动态。
""")

    rel = 100 * (1 - mean.loc['v3'] / mean.loc['v2_control'])
    rel_base = 100 * (1 - mean.loc['v3'] / mean.loc['baseline'])
    rel_class = 100 * (1 - mean.loc['v3'] / mean.loc['classical50'])
    recovery_bias = events[(events.method == 'v3') & (events.event == 'unload') & (events.window == 'recovery')]
    questions = [
        ('V3是否更符合姿态抗扰目标？','公式增加平滑绝对误差，结构上更接近IAE；实际并未证明全时域折扣目标等价于两个事件窗口的抗扰目标。'),
        ('相对V2是否改善加载IAE？',f'最终跨种子/任务均值改善{rel.load_iae_deg_s:.3f}%，幅度小且不一致；开发三种子分别一升一降等取舍，没有通过共同门禁。'),
        ('是否改善卸载IAE？',f'最终均值改善{rel.unload_iae_deg_s:.3f}%（负值即恶化），不能说更好。'),
        ('是否降低姿态峰值？',f'最终均值改善{rel.so3_peak_deg:.3f}%，属于微小差异，未宣称统计显著。'),
        ('是否降低最大角速度？',f'最终均值改善{rel.omega_peak_rad_s:.5f}%，近乎不变。'),
        ('是否增加XYZ误差？',f'相对V2最终XYZ_RMSE改善{rel.xyz_rmse_mm:.3f}%（负值为增加），无超过预设10%护栏的全面恶化。'),
        ('奖励与物理评价是否冲突？',f'固定开发集有{int(progress.return_up_but_any_iae_worse.sum())}组出现共同V2计分比初始提高、但至少一个IAE恶化。总回报结合全程姿态、位置与动作代价，不能替代事件窗口IAE；逐组数据见报告74。'),
        ('Critic是否存在持续偏差？','六组平均prediction-minus-target均为正，部分更新EV为负，表明对负回报存在过于乐观的拟合偏差；没有NaN/Inf，不称发散。'),
        ('回报尺度处理是否改善EV？','未执行ReturnNorm PPO，无法判断；只验证封装、统计保存/冻结及物理等价。'),
        ('回报尺度处理是否改善实际控制？','未执行，不作结论；V3 loss更小不是独立归一化试验。'),
        ('三个种子是否一致？','不一致。V3相对classic每种子只在一个最终任务同时改善两个IAE；任务和改善方向因种子而异。'),
        ('是否仍有加载改善/卸载恶化？','有。例如V3 seed2026在T8_combined50加载改善而卸载恶化；开发seed42同样存在取舍。'),
        ('卸载后是否有残差偏置？',f'有。V3卸载后0.1～1s沿外载方向请求均值在{recovery_bias.requested_along_load_Nm.min():.6f}～{recovery_bias.requested_along_load_Nm.max():.6f}N·m之间；非零偏置不自动等于所有状态都有害。'),
        ('残差是否方向正确？','部分状态有恢复方向，部分相反；方向要结合误差、速度和基线补偿。不能据相关性宣布准确估计未知负载。'),
        ('共享预算是否显著限制权限？','事件瞬间存在投影/限速，独立命令重算已证实；多数最终物理步未投影。不能证明权限是唯一或主要瓶颈，也不应放宽预算。'),
        ('V3是否优于冻结模型辅助控制？',f'最终均值加载改善{rel_base.load_iae_deg_s:.3f}%、卸载改善{rel_base.unload_iae_deg_s:.3f}%，有局部/平均收益，但不形成跨种子任务稳定优越性。'),
        ('是否优于同权限classic？',f'最终均值加载改善{rel_class.load_iae_deg_s:.3f}%、卸载改善{rel_class.unload_iae_deg_s:.3f}%；一项改善、一项退化，未通过成功标准。'),
        ('是否改善未见扰动方向？','新组合任务存在局部收益与退化；同一试验同时改变方向/时间/参考，不能单独归因方向泛化，也不声称每个方向分量此前都未出现。'),
        ('是否改善未见加载时刻？','最终事件时刻未用于调参，但无稳定改善；没有独立的仅改变时刻因子实验。'),
        ('是否改善动态Z+Roll/Pitch？','T8_combined50确实完成50mm和±0.25度真实动态运动，收益有加载/卸载取舍。未训练Stage C，结果不能外推为完成全课程训练后的上限。'),
        ('是否值得研究历史观测？','可作为以后单独、有限预算假设，先验证历史误差是否提供可识别信息；本轮没有证明增加历史一定解决问题，不自动新增网络。'),
        ('是否值得研究未知执行器滞后？','有物理动机，但需冻结新的未知滞后分布、强控制基线与可控性/安全试验。本轮没有做未知滞后的单因素辨识验证。'),
        ('主要瓶颈是什么？','已观察到目标不完全一致、价值拟合偏差和持续残差偏置；动作受限是系统事实，历史信息不足尚未经消融。证据不足给出唯一主因。'),
        ('是否证明额外未知扰动适应？','没有。局部改进不等于因果在线辨识，未稳定超过classic，也没有足够跨分布证据。'),
        ('是否进入更复杂RL架构？','不建议立即升级LSTM/Transformer或扩大训练。先审核此有限负/部分正结果，明确是否值得新的受控诊断；本轮停止。'),
    ]
    videos = [read(RUN / f'video_{k}.json') for k in ['training','reward_comparison','attitude_stability']]
    video_table = table(pd.DataFrame([dict(file=Path(v['path']).name, seconds=v['seconds'], frames=v['frames'], fps=v['fps'],
                          size_MB=Path(v['path']).stat().st_size/1e6, sha256=v['sha']) for v in videos]))
    gui = [read(p) for p in (RUN/'viewer').glob('*/viewer_validation.json')]
    assert len(gui) >= 2 and all(g['bitwise_equal'] and g['window_closed'] for g in gui)
    tb = read(RUN/'tensorboard_http_validation.json')
    assert tb['status'] == 200 and len(tb['runs']) == 18
    question_text = ''.join(f'### {i}. {q}\n\n{a}\n\n' for i, (q, a) in enumerate(questions, 1))
    doc('77_Phase2C8最终验收报告.md', f"""
# Phase 2C-8 最终验收报告

{prefix}

**阶段判定：{status}。实际完成六组8192决策匹配PPO Pilot、60次开发评价和32次最终评价；V3未通过训练扩展门禁。局部姿态改善存在，但尚无稳定超过强传统控制的证据。**

这不是32768完整训练成功，也不是归一化实验完成。H1奖励更换的稳定收益假设本次未获支持；H2归一化因果试验未启动。

## 交付与实际结果

{train_table}

正式PPO：49152决策、983040物理步、96次更新、7680次Adam，60个完整回合加12个预算片段。全部Actor/Critic真实变化；原始随机动作与每1ms实际状态完整保存。训练保护失败0。

{mean_table}

32次最终评价全部安全完整结束。微小数值改善不宣称统计显著；仅三个训练种子、四个有限任务。每个种子保留，不筛掉退化种子。详见报告76逐任务表。

{table(gate_table)}

扩展门禁0/3种子满足同时改善要求，因此没有进入Stage C或32768，也没有ReturnNorm PPO。不对V3长期训练或未运行的归一化作预测性优越结论。

## 验收及保留问题

- 历史SHA：{history['checked']}文件，0变化；旧模型、报告、视频未覆盖。
- 原测试：{tests['historical']['passed']}通过、{len(tests['historical']['failures'])}失败。失败为 `tests/test_phase0.py::test_isolated_environment_no_learning_packages`，旧Phase0断言环境没有Torch/SB3；当前按后续研究安装了这两个包，条件冲突。未删除、skip或篡改此失败，原日志及JUnit保留。
- 新验收：{tests['new']['passed']}通过、{len(tests['new']['failures'])}失败；合并本次执行总计{sum(x['passed'] for x in tests.values())}通过、{sum(len(x['failures']) for x in tests.values())}失败，不能报告“所有测试通过”。
- 新测试覆盖RNG、阶段恢复、匹配初始化/预算、58维/2维接口、控制限幅、零动作、reward重算、真实参数变化、模型重载、测试隔离、归一化封装、GUI、视频和负门禁。
- 96个GAE rollout独立复算通过；正式983040物理行reward全部复算通过；最终32条轨迹×24指标复算通过，另核对恢复状态。
- 最终外载逐位比较首次失败由约1e-14量级坐标变换浮点舍入引起，日志保留；改为1e-12数值核验，不改变安全门槛或原始结果。
- 两次Windows Qt真实GUI（V2/V3）分别1000物理步：暂停零积分、单决策20步、前缀与无GUI保存状态逐位一致、关闭正常归档。物理均通过实际ctrl与mj_step执行。
- 归一化1500物理步仅封装测试，不计为归一化PPO训练。任意回合中途崩溃续训尚不支持；声明的课程边界恢复已通过。

原测试命令（本次已执行，输出保留）：

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m pytest tests --ignore=tests/test_phase2c8.py -q --basetemp=results/runs/phase2c8_20261109_v1/pytest_history --junitxml=results/runs/phase2c8_20261109_v1/historical_tests.xml
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m pytest tests/test_phase2c8.py -q --junitxml=results/runs/phase2c8_20261109_v1/new_tests.xml
```

再次运行测试请换独立basetemp/日志名，避免pytest清理既有测试临时归档。模型原始运行目录不作为临时目录。

## 观看真实机器人与训练视频

GUI复用既有MuJoCo+Qt架构，显示冻结策略重新执行的真实动力学；不是重新训练，也不是训练现场。启动命令的两个方法均已用 `--validate-steps 1000` 实际测试。正常观看去掉该验收退出参数：

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c8_policy.py --method v3 --seed 42 --task T8_mix
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c8_policy.py --method v2_control --seed 42 --task T8_mix
```

控制面板可选任务、baseline/classical50/PPO及六个已验收模型，选择后Reset生效。Pause暂停积分，Single decision只推进20个物理步；Reset新建独立运行日志。相机拖拽/滚轮；任务完成后停在终态，Reset可重新执行；关闭窗口归档退出。不能把本GUI标签“LIVE DETERMINISTIC EVALUATION”称为在线训练。

训练过程请看实际采样RGB视频和日志：

{video_table}

三个视频1920×1080、25fps，均经过正常解码、不同帧变化、关节位移及时间/状态SHA对应检查。另已查看三张代表帧，文本、机器人和曲线清晰。训练视频为seed42的V2/V3各A/B真实episode拼接，B含50mm真实升降；来自训练现场RGB，图表后处理，不重新predict生成轨迹。画面显示policy version，回合跨更新时会变化。未录制不存在的Stage C训练。另两视频分别T8_mix八控制器、T8_combined50传统/经典/V3三种子的**独立确定性评价**，各自有明确标签。

训练扰动引起的姿态偏差本来很小；没有放大机器人真实运动。右侧角度/误差曲线使用自适应纵轴帮助看清，标题保留单位和真实范围。50mm升降是真实位移。单条演示不是整体成功率证据。

```powershell
Start-Process .\\assets\\videos\\stewart_phase2c8_training.mp4
Start-Process .\\assets\\videos\\stewart_phase2c8_reward_comparison.mp4
Start-Process .\\assets\\videos\\stewart_phase2c8_attitude_stability.mp4
```

视频已实际解码并检查；上述系统关联播放器命令只是打开同一文件，不声称额外进行过播放器GUI自动测试。

TensorBoard已在 [http://127.0.0.1:6034/](http://127.0.0.1:6034/) 运行，HTTP和18run标量接口验证通过。若以后进程已结束，可以用本次实际启动参数重开：

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m tensorboard.main --logdir_spec "actual:results/runs/phase2c8_20261109_v1/training,derived:results/phase2c8_tensorboard" --host 127.0.0.1 --port 6034
```

## 25项研究问题

{question_text}

## 未执行项及下一步

未执行：Stage C追加24576决策、每组累计32768、ReturnNorm PPO、2.4N·m专项、4.8N·m正常探索、LSTM/Transformer/SAC/TD3、新硬件实验。未创建虚构checkpoint或归一化结果目录。

当前建议保留强经典控制作为主基线，接受此有限Pilot的局部/负结论。以后若获授权，优先验证信息可辨识性、奖励窗口取舍和未知滞后的独立控制问题，再决定是否值得新RL架构；不凭Reward或最佳种子追加预算。

报告72～76包含协议、奖励、训练、Critic和逐任务行为证据。可机读总判定为 `results/phase2c8_final_decision.json`，安全/预算/测试汇总为 `results/phase2c8_safety_audit.json`，源文件清单随最终artifact manifest归档。

**本阶段停止，等待用户审核。**
""")
    print(status)
    print('Reports 72-77 and safety/final decisions written from actual artifacts.')
    print(mean.to_string())


if __name__ == '__main__':
    main()
