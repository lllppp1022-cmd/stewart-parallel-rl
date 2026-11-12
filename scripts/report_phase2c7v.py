"""Generate Chinese reports from computed results, never hard-code new measurements."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from phase2c7v_common import ROOT,RUN,OLD,SEEDS,NAMES,POINTS,read,write_json

def table(df):
    def val(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}' if np.isfinite(x) else '缺失/不适用'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def write(number,name,text):
    path=ROOT/f'docs/{number}_Phase2C7V{name}.md';path.write_text(text,encoding='utf-8')

def main():
    audit=read(RUN/'reward_audit_summary.json');d=pd.read_csv(ROOT/'results/phase2c7v_reward_components.csv');e=pd.read_csv(ROOT/'results/phase2c7v_episode_returns.csv');s=pd.read_csv(ROOT/'results/phase2c7v_training_summary.csv');u=pd.read_csv(RUN/'updates.csv');v=pd.read_csv(ROOT/'results/phase2c7v_checkpoint_evaluation.csv');b=pd.read_csv(ROOT/'results/phase2c7v_policy_behavior.csv');a=pd.read_csv(ROOT/'results/phase2c7v_action_audit.csv');go=pd.read_csv(RUN/'governor_counterfactual_audit.csv');q=pd.read_csv(ROOT/'results/phase2c7v_rollout_diagnostics.csv');value=pd.read_csv(RUN/'value_recomputation.csv')
    peak=u.loc[u['train/value_loss'].idxmax()];peakq=q[(q.seed==peak.seed)&(q['update']==peak['update'])].iloc[0]
    tags=read(RUN/'original_tensorboard_tags.json');cfg=read(ROOT/'configs/phase2c7_ppo.yaml')
    stage_rows=[]
    for (seed,st),g in u.groupby(['seed',pd.cut(u.decisions,[0,4096,8192,32768],labels=['A','B','C'])],observed=True):stage_rows.append(dict(seed=seed,stage=st,negative_EV=int(g['train/explained_variance'].lt(0).sum()),updates=len(g),value_loss_max=g['train/value_loss'].max(),EV_min=g['train/explained_variance'].min()))
    earlylate=[]
    for seed in SEEDS:
        g=e[(e.seed==seed)&e.complete&(e.stage=='C')].sort_values('end_decision');half=len(g)//2
        earlylate.append(dict(seed=seed,first_half_n=half,first_half_return=g.iloc[:half].episode_return.mean(),second_half_n=len(g)-half,second_half_return=g.iloc[half:].episode_return.mean()))
    pd.DataFrame(earlylate).to_csv(RUN/'stage_c_half_summary.csv',index=False)
    write(67,'训练Reward与日志审计',f'''# Phase 2C-7V：训练Reward与日志审计

本轮只分析已完成的Phase 2C-7及只读checkpoint评价。原Reward、环境、Actor/Critic和控制器均未修改。新结果根目录：`{RUN}`。

## 数据规模与完整性

实际扫描三种子pilot/continue的training_samples.csv、135份physics.csv.gz、192份rollout NPZ、updates.json、episodes.json和原TensorBoard。决策{audit['decisions']:,}，物理积分{audit['physical_steps']:,}；126个完整回合、9个边界片段。每种子A/B/C完整回合数分别5/5/32；完整回合实际均15秒。片段按真实长度显示，不能当作完整失败回合。Stage A为1–4096，B为4097–8192，C为8193–32768；图的零点是训练前。

## 四种不同的回报量

1. 环境即时Reward：20个1ms物理代价求和后除以20并取负，加原失败惩罚。
2. 完整Episode Return：完整环境回合内即时Reward之和，不包含SB3 bootstrap。
3. PPO价值目标：实际rollout中`returns=advantages+values`；GAE使用gamma={cfg['gamma']}、lambda={cfg['gae_lambda']}。时间上限终止时SB3给末步奖励加gamma×终态价值，这不属于环境奖励。
4. Evaluation Return：固定Dev_mix上冻结确定性策略的完整15秒真实动力学回报。与训练时随机策略、不同课程任务分开绘图。

135份物理日志独立复算最大Reward绝对差{audit['max_abs_difference']:.6g}，加权平均绝对差{audit['mean_abs_difference']:.6g}，异常{audit['anomalies']}；5项原始物理分量最大差{audit['physical_component_error']:.6g}。cap触发物理步{audit['cap_steps']}。失败终止为0；不能由此推断更强外载也安全。

## 原Reward-V2的真实公式

每1ms：姿态项=(SO3误差deg/0.006)^2；角速度项=0.2Σ((实际−参考)rad/s /0.003)^2；位置项=0.1Σ(误差mm/[0.2,0.2,0.05])^2；动作项=0.001Σ(接受力矩Nm/0.05)^2；变化率项=0.000025Σ(接受力矩差分Nm/s /2.5)^2。五项统一乘min(1,100/总代价)，不是逐项独立cap。

原失败惩罚函数为−(100+100×(1−gamma^(remaining+10))/(1−gamma))，remaining是剩余决策数；本数据没有失败惩罚实例。保护保持如发生也计入物理代价。时间上限用truncated，物理失败用terminated。课程/预算片段是外部停止，不能凭长度假设有一次真实done。

## 三种子分阶段统计

{table(s[['seed','stage','decisions','physical_steps','complete_episodes','return_mean_same_duration','return_std_same_duration','mean_reward','failures']])}

动作RMS、Box裁剪、共享预算标志和五项占比见`results/phase2c7v_training_summary.csv`。这里接受残差RMS来自每20ms接受力矩均值，不等于1ms物理RMS；最终行为审计的RMS使用1ms日志。历史随机高斯动作、裁剪动作、Actor均值与标准差仍完整保存在98304行的reward_components.csv中。

{table(s[['seed','stage']+[n+'_fraction' for n in NAMES]])}

各阶段姿态代价占主导；位置代价占比在C增大。占比增大不等于绝对误差增大，任务参考与扰动分布也变化。A/B/C不能当成同一个学习任务直接比较。

## Critic与优势审计

{table(pd.DataFrame(stage_rows))}

Value Loss最大值{peak['train/value_loss']:.6f}位于seed{int(peak.seed)}、update{int(peak['update'])}、决策{int(peak.decisions)}；实际任务组成`{peakq.task_composition}`。价值目标均值{peak.value_target_mean:.6f}、标准差{peak.value_target_std:.6f}、最小值{peak.value_target_min:.6f}；同rollout Critic预测范围[{peakq.values_min:.6f}, {peakq.values_max:.6f}]。这表明大负价值目标与预测尺度存在明显差距，不能仅凭高loss称为数值发散。

Value Loss是优化过程中各minibatch误差汇总，不能等同更新前一次MSE。`value_recomputation.csv`另给出更新前MSE/EV。192份rollout内部GAE递推最大数值差{value.gae_internal_max_error.max():.6g}；return=V+A最大差{value.return_value_plus_advantage_max_error.max():.6g}。每个rollout最后一个GAE所需的最终bootstrap观测/预测没有完整独立日志，最后一项保留实际NPZ值，不伪称已从原始状态全部重算。时间上限的126个bootstrap差值单独保存在bootstrap_audit.csv。

## 原TensorBoard为什么看起来没有完整曲线

原事件实际存在`attitude/reward`、`attitude/episode_return`、`attitude/reward_orientation`、`attitude/reward_omega`、`attitude/reward_xyz`、`attitude/reward_action`、`attitude/reward_slew`和`rollout/ep_rew_mean`等标签。它们经过SB3 logger按rollout写出，不能当作每决策/每完整episode的完整档案。实际标签及step清单在`original_tensorboard_tags.json`，共扫描{len(tags)}个原事件目录。

旧视频只画当前物理回合曲线和最近一次Reward文字，没有遍历所有训练episode构建Return历史。训练已完成时TensorBoard不会继续产生新点，这是正常现象。

新派生TensorBoard使用原CSV逐决策重建，目录`results/phase2c7v_tensorboard`。原事件文件不覆盖。所有摘要来自实际日志，没有补点或外推学习曲线。
''')
    comparison=[];best=[]
    for seed in SEEDS:
        g=v[v.seed==seed];initial=g[g.decisions==0].iloc[0];final=g[g.decisions==32768].iloc[0]
        comparison.append(dict(seed=seed,initial_return=initial.evaluation_return,final_return=final.evaluation_return,load_improvement_percent=100*(1-final.load_iae_deg_s/initial.load_iae_deg_s),unload_improvement_percent=100*(1-final.unload_iae_deg_s/initial.unload_iae_deg_s),Z_improvement_percent=100*(1-final.z_rmse_mm/initial.z_rmse_mm)))
        for key in ('evaluation_return','load_iae_deg_s','unload_iae_deg_s'):
            row=g.loc[g[key].idxmax() if key=='evaluation_return' else g[key].idxmin()];best.append(dict(seed=seed,criterion=key,decisions=int(row.decisions),value=row[key],scope='best among 9 sampled points only'))
    pd.DataFrame(comparison).to_csv(RUN/'development_initial_final.csv',index=False);pd.DataFrame(best).to_csv(RUN/'development_sampled_best.csv',index=False)
    write(68,'_PPO学习过程与固定任务评价',f'''# Phase 2C-7V：固定任务学习能力评价

## 协议与实际预算

使用历史已存在的开发任务Dev_mix：50mm Z，Roll +0.1°、Pitch −0.1°，6秒上升参考，5秒加载[0.48,−0.64]Nm、7秒卸载，完整15秒。同任务、同物理初态、同参考、同外载、同安全规则；三个seed各评价{list(POINTS)}，另运行同权限经典50Hz对照。共{len(v)}次真实动力学评价、{int(v.physical_steps.sum()):,}物理积分。全部完成且无保护失败。无新PPO决策训练、无梯度更新。201个历史checkpoint均建立SHA清单，曲线只选每seed9个冻结点，不宣称评价全部64次更新。

先冻结7V protocol.json，再进行评价。该曲线属于事后分析既有开发集；不重新选择正式final、不把开发曲线冒充独立盲测。F_roll额外四次录像只是已公开最终任务的行为展示，不用于调参。initial三个模型另与原pilot summary的initial_hash参数哈希逐一核对；update/pilot/final文件SHA与原训练记录核对，7V前后全部文件再做SHA比较。

历史Stage C环境RNG恢复行为见原stage_c_rng_audit.json：SB3载入时待应用的vec seed覆盖了恢复的环境采样RNG，所以C阶段任务序列按seed重新开始；网络、优化器及其它已记录状态连续。本轮没有修补旧训练或改写此记录，课程分布分析以实际发生的任务序列为准。

## 从initial到final的固定任务变化

{table(pd.DataFrame(comparison))}

正百分比代表误差降低。seed42在这个开发任务加载和卸载均改善；seed123的Reward明显改善但卸载IAE变差；seed2026的Reward改善但两个IAE均较initial变差。由此不能把Reward上升直接等同突变抗扰性能提高。

{table(v[['seed','decisions','evaluation_return','load_iae_deg_s','unload_iae_deg_s','so3_peak_deg','z_rmse_mm','force_rate_N_s','success']])}

另有完整角速度、XY峰值、观测裁剪、约束残差、间隙及原生驱动力指标保存在checkpoint_evaluation.csv，每行含模型SHA与原始物理日志目录。曲线九个面板分别显示指标；连接线只引导视觉，点之间未执行的checkpoint没有虚构指标。

## 阶段解释

Stage A：seed123加载改善而卸载变差；seed42变化小，seed2026也没有一致双窗口收益，不能称三个种子都已学成有效抗扰器。Stage B：seed42明显改善；seed123卸载代价上升；seed2026的6144点Reward和加载IAE退化。Stage C：三种子final的开发Return都较initial好，但IAE存在分化和非单调变化。

seed42从16384到24576时开发Return和加载IAE反弹，final又恢复；seed2026在12288/16384附近的Return和IAE也不一致。可称局部性能退化/波动，单一开发任务不足以确定灾难性遗忘机制。不能将课程分布变化和随机轨迹难度归因为算法必然改善或必然遗忘。

训练Stage C前16与后16个完整回合的描述性统计如下（任务混合不同，不是配对效果估计）：

{table(pd.DataFrame(earlylate))}

## 哪个checkpoint较好

{table(pd.DataFrame(best))}

按Return三seed的已采样最佳均为final；按加载/卸载IAE最佳点不同。此为分析结论，不修改历史final_selection.json，也不对最终测试集重新选择策略。

## 最终冻结测试结论不变

历史四任务×三seed中，加载和卸载IAE同时改善仅2/12项：seed42/F_pitch、seed2026/F_30；没有seed满足原跨任务共同门禁。seed42/F_roll相对冻结combined baseline加载改善34.27%，卸载恶化43.33%。该百分比比较对象是combined baseline，不是本轮视频中的classical50；比较对象不能混用。

本轮的新证据是“相同开发任务上训练过程可测量但指标分化”，不是“PPO已稳定优于强传统控制”。所有样本量有限，无统计显著性声明。
''')
    groups=pd.DataFrame(read(RUN/'observation_groups.json'));sens=pd.read_csv(RUN/'sensitivity_groups.csv')
    write(69,'_Actor策略控制行为解释',f'''# Phase 2C-7V：Actor控制行为解释

## 真实控制链

Dynamic Reference → C4因果观测器+fourDlead+FF-VA → 加入二维PPO力矩请求 → 名义Jacobian转六支链力 → 单一共享预算器 → 原生执行器 → MuJoCo闭链动力学。

Actor是58输入、2输出的连续高斯策略；训练采样随机动作，评价用确定性均值，再Box裁剪至[−1,1]。请求Mx/My分别等于0.05×裁剪动作，单位Nm。它不修改平台目标、支链行程或外载。1kHz物理控制，每20ms才产生一次新Actor动作；共享预算逐1ms处理。传统C4补偿器历史代码名C3，报告名称与代码名称需要对应。

## 58维输入的准确顺序（零起始、右端不包含）

{table(groups[['name','start','end_exclusive','dimension','unit','scale','source']])}

观测每维最终裁剪±10。参考姿态是相对home旋转向量；SO3误差是R_actual R_ref^T的旋转向量，输入单位rad，图上Roll/Pitch为deg。观测器输出来自因果控制器估计，不是实际未知外力；控制命令、历史请求/接受量及参考均已知。这里不是58个独立传感器。MuJoCo中的位姿和速度是假定可测状态，未来实机还需传感/状态估计验证。本轮没有把真实drive增益、质量、滞后参数、外载值及事件时刻提供给Actor。

## 四任务×三种子的动作审计

{table(a.drop(columns=['source']))}

requested_accepted_rms_ratio是两个波形RMS的比，不是逐时刻标量接受率。restoring/damping符号比例只做描述，不是闭环稳定性证明；混合动态参考时不能要求每时刻残差都逆误差或逆速度。

最重要的补充发现：日志accepted等于J_nominal^T×(真实命令−同状态无RL反事实命令)的Roll/Pitch分量。无RL反事实同样继承**真实前一时刻**的共享预算状态，因此本身可能受限速。它不是独立“关闭PPO”的另一条物理轨迹，也不必等于请求。

{table(go.drop(columns=['meaning']))}

用已保存F_C3、F_C0、fourDlead请求、FF请求与RL映射力，逐1ms独立重建含RL及无RL两次投影。总命令与无RL反事实最大复算误差分别{go.command_max_error_N.max():.4g}N、{go.counter_base_max_error_N.max():.4g}N。seed42/F_roll接受/请求RMS约0.654，但真实总请求投影比例仅0.0214%，无RL反事实限速比例57.69%。不能把RMS差直接解读为“35%控制权被预算裁掉”。对本轮全部12项反事实审计，没有幅值超5N案例，主要为20N/s变化率约束。

日志的实际几何Jacobian与控制器名义Jacobian不同；用前者复算接受力矩最大差{a.actual_vs_nominal_mapping_max_Nm.max():.5g}Nm，属于此处几何近似/状态差异，不能冒充两个Jacobian严格一致。接口仍保留名义映射，不改变控制算法。

## 加载、持续、卸载和恢复

{table(b[(b.task_id=='F_roll')][['seed','event','window','requested_along_load_Nm','accepted_along_load_Nm','first_recorded_request_change_ms']])}

上述along_load按该任务真实外载方向投影，仅用于离线解释，不输入Actor。F_roll事件位于决策栅格中间，首次记录请求变化为10ms；它可能也受正在变化的参考影响，不能称10ms外力估计延迟。其他任务/事件的实际机会及首变化见policy_behavior.csv。20/40/60/100ms是累计窗口均值，不是一个独立瞬时采样点。物理图有±100ms放大和每20ms决策标记。

seed42在加载前已存在明显偏置，卸载后前100ms和恢复窗仍保持与此前相同方向的补偿分量。加载收益与卸载代价的非对称，与该偏置、动态参考以及同状态反事实限速一致；目前没有在相同物理状态做去偏置/替换动作的受控闭环干预，所以不能断言其为唯一因果机制。也没有证据把所有退化都归因于传统控制器与PPO竞争。

## 局部输入敏感性

每seed使用最后一个真实rollout的64个等间隔观测；对各输入施加±0.01归一化单位，冻结Actor中央差分。输出按裁剪动作×0.05Nm计算。64×58×2次单seed输入扰动，没有运行新训练。

{table(sens.sort_values('mean_l2_sensitivity_Nm_per_normalized_unit',ascending=False).groupby(['seed','axis']).head(2))}

结果表明历史请求/接受量、因果估计等会改变输出，但组维数及归一化尺度影响L2幅值，不能当成无偏特征重要性排序。输入维之间有运动学关系，单维扰动通常离开真实状态流形。这里只能证明局部数值敏感性，不能证明Actor已经辨识或估计未知外力矩，也不能据此给出闭环因果贡献。

全部36张物理时序图位于`results/runs/phase2c7v_20261108_v1/behavior_plots`。经典反馈图线来自独立、同任务真实轨迹；增强视频中的same-state PD第三曲线是离线公式诊断且未执行，二者不可混淆。
''')
    video=read(RUN/'training_video_validation.json');comp=read(RUN/'comparison_video_validation.json');gui=read(RUN/'analysis_gui_validation.json')
    write(70,'可视化及视频说明',f'''# Phase 2C-7V：可视化及视频使用说明

## 直接复制运行：只读训练分析

```powershell
Set-Location '__PROJECT_ROOT__'
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_training.py
```

选择seed、All stages/A/B/C、指标。Raw data显示真实点/数据；Display trend只控制显示平滑，不覆盖CSV。Reward五项可切换绝对代价或占比；Actor/Critic、Value loss、EV、KL、clip、标准差、价值目标/优势和固定开发评价都可选择。鼠标缩放/拖动曲线，右键可自动恢复范围。checkpoint选择显示该实际文件训练步数及SHA，只读、不加载训练器。67个文件/seed均可查看，只有9个已评价点有新开发曲线。

## 真实动力学冻结策略Viewer

```powershell
Set-Location '__PROJECT_ROOT__'
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --seed 42
```

窗口选择F_roll/F_pitch/F_mix/F_30、冻结combined baseline/Classic50/PPO及seed后点Reset。Pause/resume暂停/继续；Single decision暂停并推进一次50Hz决策，即20个真实1ms积分；拖动机器人视图调整相机，滚轮缩放。每个回合按冻结任务时长结束，窗口保留结果，可Reset运行新真实回合。关闭窗口归档，不会改变历史目录。八条实时曲线包含Z、Roll、Pitch、SO3、外载、请求/接受力矩、六支链相对行程和驱动力。只允许三个已验证final和四个已验证任务；早期checkpoint不任意开放到最终任务。

```powershell
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --classical
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_policy.py --task F_pitch --seed 123
```

Viewer是新的冻结确定性评价，不是历史随机训练现场。Qt主线程持有live MjData并做data.ctrl/mj_step，绘制读取同一状态，GUI不改qpos或控制限制。

## 增强视频

- `assets/videos/stewart_phase2c7_training_analysis.mp4`：{video['seconds']:g}秒、{video['frames']}帧、1920×1080/25fps。原seed42六段真实随机训练RGB，不重新predict、不重造动作。左上机器人/当前值，右上姿态/外载，左下完整Return历史，右下请求/边际接受/支链力/物理代价。Current checkpoint显示当时采样策略版本×512，当前决策可能位于其后。完整回合跨更新边界时版本随真实日志变化。
- `assets/videos/stewart_phase2c7_policy_comparison.mp4`：{comp['seconds']:g}秒、{comp['frames']}帧、同分辨率。F_roll相同任务classic50、seed42、123、2026各14秒真实确定性执行。该任务预先选作已知加载/卸载权衡示例，不是事后挑最佳任务。

训练视频明确POST-PROCESSED TRAINING VISUALIZATION；已完成Return为蓝色，未来数据灰色，截断片段x。任何完整回合要等结束后才算历史点，不把当前片段提前当完整回报。训练RGB原档只有seed42，另外两个seed只有完整数字日志，未伪造它们的训练RGB。本轮三seed对比是新冻结评价录像。

每个源帧核对frames.jsonl时间、integration_states qpos SHA和对应物理行；输出视频全帧解码通过。验证文件为training_video_validation.json、comparison_video_validation.json。姿态扰动角很小，机器人保持真实几何比例，不能为了肉眼显著而放大qpos；看右侧角度/误差曲线判断细微抗扰变化。

```powershell
Start-Process '__PROJECT_ROOT__\\assets\\videos\\stewart_phase2c7_training_analysis.mp4'
Start-Process '__PROJECT_ROOT__\\assets\\videos\\stewart_phase2c7_policy_comparison.mp4'
```

## TensorBoard

原标签`attitude/reward`、`attitude/episode_return`及五项分量仍可看。派生目录逐决策完整记录，完整回合标签为`training/complete_episode_return`，片段独立`training/incomplete_fragment_return`，优化器标签以`optimizer/`开头，固定任务以`evaluation/`开头。三类曲线不要混读。

```powershell
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' -m tensorboard.main --logdir_spec 'original:results/runs/phase2c7_20261107_v1/training,derived:results/phase2c7v_tensorboard' --host 127.0.0.1 --port 6033
```

浏览器打开http://127.0.0.1:6033。端口6032原服务不修改。Qt单独即可看曲线，不要求同时开三个窗口。数据已冻结，浏览器刷新不会自动增加训练步数。该启动命令已经实际运行，HTTP 200；服务加载完成后确认原6个run及派生3个seed均可见，验证在tensorboard_http_validation.json。首次打开要等待后台读取事件文件，不能把首次加载不全误认为缺少记录。

## 已实际执行的验收入口

```powershell
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_training.py --validate
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --seed 42 --validate-steps 1000
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --classical --validate-steps 1000
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/evaluate_phase2c7v_checkpoints.py collect
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' scripts/plot_phase2c7v.py
& 'D:\\miniconda\\envs\\stewart_rl\\python.exe' -m pytest tests/test_phase2c7v.py -q
```

分析窗口实际检查{gui['combinations']}个seed×stage×指标组合。两个真实物理GUI测试均与无窗口记录前1000步逐位一致，包含暂停与20步单次推进。截图保存在7V运行目录。初版Qt控件命名覆盖metric虚函数及阶段列缺失已修复，失败日志保留。没有通过修改环境解决GUI问题。
''')
    testrows=[]
    for file in ('new_tests_02.xml','historical_tests.xml','video_final_tests.xml'):
        p=RUN/file
        if p.exists():
            tree=ET.parse(p).getroot();suites=list(tree) if tree.tag=='testsuites' else [tree]
            testrows.append(dict(suite=file,tests=sum(int(x.get('tests',0)) for x in suites),failures=sum(int(x.get('failures',0)) for x in suites),errors=sum(int(x.get('errors',0)) for x in suites),seconds=sum(float(x.get('time',0)) for x in suites)))
    hist=read(RUN/'history_after.json')
    answers=[
      '三seed完整训练回报波动明显，阶段均值不构成单调学习证据；固定Dev_mix的final Return均较initial改善。详见67阶段表与68配对表。',
      '不能直接比较A/B/C：参考运动、扰动强度和采样任务分布改变。全部完整回合虽都是15秒，任务难度仍不同；9个片段另外显示。',
      '姿态项始终占主导，C阶段XYZ项占比上升；实际绝对值和占比都由五项原Reward复算，未改权重。',
      '确实更新：192次Actor/Critic参数变化均非零，共15360次Adam优化步；本轮7V新增更新为0。',
      '多个阶段EV有负值，Stage B的强扰动价值目标与预测差距突出；负EV/大loss并非单独的发散证明。',
      f'最高Value Loss为{peak["train/value_loss"]:.3f}，seed{int(peak.seed)}第{int(peak["update"])}次更新，{peakq.task_composition}。具体价值目标和预测范围见67。',
      '不总对应。Dev_mix上seed42两个IAE改善；123卸载恶化；2026两个IAE恶化，同时三个final Return均提高。说明全轨迹加权奖励与事件窗口指标并不等价。',
      '九个采样checkpoint中，三个seed按开发Return均final最佳；加载/卸载最优点不同，见68。不改正式模型选择。',
      '存在部分连续checkpoint性能反弹；单任务、有限采样不足以认定遗忘机制或估计其统计频率。',
      '存在。F_roll三个seed加载前100ms请求均非零，其中seed42沿外载方向约−0.03571Nm；它也可能是动态参考补偿，不能全称外力抵消。',
      '50Hz动作只有每20ms更新机会；F_roll首次记录改变在事件后10ms。该数值包含栅格相位及参考变化，不是证明网络在10ms内估计出扰动。',
      '1ms接受/请求RMS比在不同seed/task变化，详见69。seed42/F_roll约0.654；接受量是同状态反事实边际作用，不能直接当成65.4%权限利用率。',
      '有符号相关和局部输入敏感性，但不是简单常数PD。历史请求/接受量也敏感；不能从相关性宣称物理因果或最优控制。',
      'seed42有卸载后残差偏置持续，且反事实限速影响边际量；与加载/卸载不对称相符。没有动作替换闭环干预，尚不能唯一定位原因。',
      '三个种子经历不同随机动作/任务并得到不同参数与输出偏置。Stage C历史环境RNG恢复还存在已披露的重置行为；这些是可信差异来源，未做因果分解或显著性检验。',
      '存在50Hz更新及20N/s共享变化率限制。但当前大部分实际请求未触发预算投影，不能仅凭接受RMS比要求扩大权限。',
      '同权限经典反馈是显式即时姿态/角速度规则；PPO依赖更多参考与历史控制状态，出现偏置及非对称响应。真实F_roll四方法视频与完整四任务图均保留差异，不只展示最好seed。',
      '不能证明PPO估计了未知外力矩。Actor读取因果观测器估计并对状态反馈，与Actor自身建立正确扰动估计不是同一结论。',
      '模型控制器上的二维连续残差强化学习，PPO以高斯探索训练、确定性均值评价，经Jacobian及共享预算作用到六个力执行器；不是端到端六缸策略或在线模型辨识证明。',
      '下一轮应先审计Critic尺度/目标拟合与奖励相对加载卸载指标的一致性；保持物理权限不变做有限对照。随后才考虑有界历史观测消融。控制权限先解释边际定义与限速，不盲目放宽。当前不支持无门禁扩大训练。']
    write(71,'最终验收报告',f'''# Phase 2C-7V最终验收报告

## 结论

已完成历史奖励复算、三种子只读checkpoint评价、动作与共享预算审计、Qt交互分析、真实动力学Viewer、两段增强视频及派生TensorBoard。没有新PPO训练、梯度更新、权重修改或原控制器参数调整。**开发Reward提升不等于加载/卸载抗扰能力一致提升；原Phase 2C-7共同成功门禁仍未通过。**

历史SHA核对{hist['checked']}个文件，passed={hist['passed']}，差异{len(hist['differences'])}。所有新实验与派生数据在`{RUN}`，原项目其它阶段结果未覆盖。

## 本轮真实运行清单

- 重算98,304决策 / 1,966,080历史物理行；不是再次训练这些步数。
- 27次冻结checkpoint Dev_mix评价+1次经典对照：420,000个新评价物理积分。
- F_roll经典及3seed录像：56,000个新评价物理积分。
- 两次1000步Qt真实物理对照：2,000积分；GUI只读分析144组合不积分。
- 合计新增**评价**物理积分478,000；新增训练决策0、梯度更新0。历史回归测试另有测试专用积分，不计入研究评价样本量。
- 历史126完整回合+9片段、192优化事件、201checkpoint SHA；12组最终策略物理行为、36张全程/加载/卸载图；每seed64真实观测×58维局部敏感性。
- 90秒原现场训练分析视频与56秒新冻结评价对比视频；均1920×1080，25fps，全帧解码与状态/时间戳核对。

## 自动与GUI验收

{table(pd.DataFrame(testrows))}

历史Phase0“后续RL依赖不应安装”断言在当前已有torch/SB3环境中预期不成立；保留原测试及失败，不声称全套100%通过。新增测试初版对转换后的world外载逐位相等要求失败，最大差约4.9e−15；仅将该浮点计算比较改为rtol=0、atol=1e−12，未放宽动力学安全限制。初版失败XML/日志保留。最终新测试覆盖模型和日志SHA、完整回合、阶段、Reward、GAE、bootstrap、相同评价条件、Actor输入、治理器映射、真实GUI与视频证据。

GUI因控件名metric覆盖Qt虚函数产生的崩溃、阶段列遗漏和审计脚本字段/语法问题均已修复，失败日志保留。两次GUI真实执行前1000步与无窗口逐位一致，关闭后归档完整。

## 二十个研究问题

'''+'\n\n'.join(f'{i}. {text}' for i,text in enumerate(answers,1))+f'''

## 文件导航

67：Reward/日志/GAE与Critic审计；68：固定开发任务学习曲线；69：58维输入、动作时序与共享预算；70：PowerShell命令与窗口/视频教程。

关键CSV：`results/phase2c7v_episode_returns.csv`、`phase2c7v_reward_components.csv`、`phase2c7v_training_summary.csv`、`phase2c7v_checkpoint_evaluation.csv`、`phase2c7v_policy_behavior.csv`、`phase2c7v_action_audit.csv`。完整原始新评价物理数据和模型SHA在7V运行目录。视频、图片与TensorBoard路径见70。

## 明确的限制与未执行项

- 没有评价所有64更新checkpoint，只评价事先冻结的9点/seed；中间点不插值成真实成绩。
- 仅一个已有开发任务构建学习曲线，不能代表全面泛化；F_*已查看的任务不重新称盲测。
- 原训练RGB仅seed42的6回合；其他seed没有真实训练RGB不能恢复现场像素，只有完整数字日志。
- 每rollout最后GAE所需独立bootstrap状态记录不齐，保留真实NPZ值和环境/rollout奖励差，不编造独立复算。
- 没有做闭环控制消融去证明偏置/限速的唯一因果作用；局部输入扰动可能离开可达状态流形。
- 同状态边际接受残差不等同独立无PPO轨迹差，更不等同未知外力估计。
- 没有新增故障、实机试验、连续安全证明或统计显著性结论。
- 本阶段结束，未启动新的训练或下一研究阶段。
''')
    write_json(RUN/'report_manifest.json',dict(reports=[str(p) for p in (ROOT/'docs').glob('*Phase2C7V*.md')],evaluation_episodes=32,evaluation_physical_steps=478000,new_training_decisions=0,new_gradient_updates=0,tests=testrows))
if __name__=='__main__':main()
