"""Finalize Chinese reports from frozen-run evidence, never invented outcomes."""
import json
import numpy as np
import pandas as pd
from phase2c7_protocol import ROOT,RUN,SEEDS,read,write_json
from src.model import sha


def table(frame):
    head='| '+' | '.join(map(str,frame.columns))+' |\n'
    head+='| '+' | '.join(['---']*len(frame.columns))+' |\n'
    return head+'\n'.join('| '+' | '.join(f'{x:.6g}' if isinstance(x,float) else str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def save(name,text):
    (ROOT/'docs'/name).write_text(text,encoding='utf-8')


def main():
    decision=read(ROOT/'results/phase2c7_final_decision.json')
    train=pd.read_csv(ROOT/'results/phase2c7_training_summary.csv')
    evaluation=pd.read_csv(ROOT/'results/phase2c7_controller_comparison.csv')
    stage=pd.read_csv(RUN/'stage_composition.csv')
    effects=pd.read_csv(RUN/'relative_effects_percent.csv')
    diagnostics=[]
    for seed in SEEDS:
        u=pd.DataFrame(read(RUN/f'training/seed{seed}/continue/updates.json'))
        diagnostics.append(dict(seed=seed,updates=len(u),adam_steps=int(u.adam_steps.sum()),
            max_value_loss=u['train/value_loss'].max(),minimum_explained_variance=u['train/explained_variance'].min(),
            maximum_target_std=u.value_target_std.max(),minimum_target=u.value_target_min.min(),maximum_KL=u['train/approx_kl'].max()))
    diag=pd.DataFrame(diagnostics);diag.to_csv(RUN/'critic_diagnostics.csv',index=False)
    final=train[train.phase=='continue']
    aggregate=evaluation.groupby('method').agg(cases=('success','size'),completed=('success','sum'),
        load_iae_deg_s=('load_iae_deg_s','mean'),unload_iae_deg_s=('unload_iae_deg_s','mean'),
        so3_peak_deg=('so3_peak_deg','mean'),z_rmse_mm=('z_rmse_mm','mean'),xy_peak_mm=('xy_peak_mm','mean'),
        actual_force_slew_N_s=('force_rate_N_s','mean')).reset_index()
    save('64_Phase2C7多任务PPO训练记录.md',f'''# Phase 2C-7 多任务PPO训练记录

三种子真实完成新增{decision['new_training_decisions']}次PPO决策、{decision['new_training_physical_steps']}次MuJoCo积分。这里不计预检、测试和独立评价的额外物理成本。每次正常动作20个1 ms积分步。Pilot共同门禁通过后，三个种子均获得相同的累计32768决策预算。

{table(final[['seed','decisions','physical_steps','updates','adam_steps','complete_episodes','failures','wall_s']])}

上表continue行的决策/物理步/更新是累计数；完整回合和墙钟为该次Stage C进程自身值。全部阶段的真实组成如下，预算截断的不完整回合保留，不能当成完整成功回合。

{table(stage[['seed','phase','stage','decisions','physical_steps','reward_mean','reward_min','failures','task_ids']])}

## 价值估计诊断

{table(diag)}

早期1.2 N·m Home任务出现约−300量级的价值目标和约1万量级Value Loss。原始物理记录显示该任务单决策cost约最高11，持续外载下误差累积，使GAE/价值目标比轻载任务大。大Loss本身不等于数值发散；同时检查有限性、目标分布、解释方差、完整回合和安全记录。本轮没有因最终测试结果修改奖励、网络、归一化或追加预算。

共记录135个训练回合片段，其中126个完整结束、9个被课程边界或预算截断；没有把后者计为完整成功或失败。所有训练回合片段的奖励cap率及观测裁剪率均为0，详见`phase2c7_reward_diagnostics.csv`。实际运行版本见`runtime_versions.json`，本次使用MuJoCo 3.17.0、SB3 2.9.2、CPU版PyTorch，不能套用早期阶段的版本声明。

每512决策保存一次完整rollout、Actor/Critic变化范数、实际Adam步数、模型SHA、Value Loss/KL/EV/优势与价值目标分布。`training_samples.csv`保存原始高斯动作、裁剪动作、实际接受残差、Actor均值/标准差及奖励五项。每个回合保留1 ms物理CSV与完整积分状态；选定seed42回合现场录制RGB。

原始高斯动作是float32，CSV使用NumPy最短可往返字符串。核验时恢复float32后与全部rollout逐值完全相等；不能把十进制CSV默认解析为float64后的约1e−8表示差异当成动作不一致。初次采用1e−15 float64容差的失败测试日志保留，修正核验数据类型后通过，未改写任何原始动作数据。

模型目录为`models/phase2c7/seed42`、`seed123`、`seed2026`。包含initial、pilot、逐次update及final。Pilot与Stage C之间明确结束预算截断回合并开始新完整回合。网络和优化器继续，PyTorch/NumPy/Python全局随机状态恢复；SB3加载后首次VecEnv reset会按各自训练种子重新播种环境任务采样。实际Stage C任务序列已与该种子的PCG64序列核对（`stage_c_rng_audit.json`）。虽然代码赋回了保存的环境RNG状态，该首次reset会覆盖它，因此不能宣称环境随机流无缝续接，更不能宣称恢复了被截断回合的连续物理现场。三个种子均使用相同规则，没有挑选采样顺序。

## 查看与训练命令

以下训练命令已在本次实验实际运行。输出目录有存在检查，重复运行不会覆盖旧模型。再次训练应先创建新的独立实验配置和输出目录，不能删除本轮目录后冒充复现。

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/train_phase2c7_ppo.py --seed 42 --phase pilot
# 123、2026也执行相同Pilot；完整开发评价和共同门禁通过后：
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/train_phase2c7_ppo.py --seed 42 --phase continue
```

TensorBoard日志已实际通过`--inspect`验证。下面是启动网页的命令；网页服务需在终端保持运行，Ctrl+C关闭。

```powershell
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m tensorboard.main --logdir results/runs/phase2c7_20261107_v1/training --port 6032
```

网页地址为http://127.0.0.1:6032。旧6031页面若仍指向历史目录，不会自动显示新阶段曲线。

以下复算/视频命令也已经实际执行。它们读取本次原始数据并更新本阶段派生产物，不重新训练，也不改写旧阶段实验。

```powershell
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/collect_phase2c7.py collect
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/render_phase2c7_videos.py dynamic_ppo_training
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/render_phase2c7_videos.py classical_vs_ppo
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/render_phase2c7_videos.py dynamic_disturbance_demo
```
''')
    save('65_Phase2C7经典控制与PPO独立评价.md',f'''# Phase 2C-7 经典控制与PPO独立评价

最终状态：**{decision['status']}**。

冻结四个独立运动时序/载荷任务，每种方法均使用匹配的参考、物理初态、5 ms执行器滞后、外载时序、保护及共享预算。比较冻结组合控制器、同权限50 Hz PD（Kp=25、Kd=1）、三个新PPO及一个旧52维课程PPO迁移参考。旧模型使用原特征提取，不假装转换为新58维网络。

模型集合在`final_selection.json`冻结，三个种子均选累计32768终点checkpoint。测试前基于传统控制和有界随机动作进行了安全预检，故不称完全盲测；未据此调奖励、挑种子或挑checkpoint。

下表为四个完整任务的等权描述统计，不能替代逐任务CSV，也不代表统计显著性。SO(3) peak列为各任务峰值的平均；Z列为各任务RMSE的平均。

{table(aggregate)}

## 三个新种子的逐任务相对变化

百分比统一为`100*(经典值−PPO值)/经典值`，正数表示指标降低；误差及力变化率均越低越好。

{table(effects)}

共同成功判据：至少两个种子分别在至少两个独立任务中同时改善加载与卸载1 s姿态IAE；全部到达，且各任务Z-RMSE、XY峰值、实际力变化率不超过经典对照的110%。同一任务的两个IAE必须同时改善。不能只用Reward、最好种子或一个漂亮视频宣布成功。

本次最明确的权衡例子是seed42的F_roll：加载IAE降低约34.27%，卸载IAE却增加约43.33%。seed123的四任务平均加载和卸载IAE均略低，但没有一个任务同时改善两者，说明仅汇总均值会掩盖方向间的收益交换。三个种子都未满足冻结的跨任务成功条件。

{table(pd.DataFrame(decision['qualifying']))}

完整评价包含恢复状态及右删失、卸载反向超调、Yaw误差、六力平方积分、附加预算占用及请求/接受残差。字段定义、绝对值和模型SHA见`phase2c7_controller_comparison.csv`、`phase2c7_attitude_metrics.csv`与`phase2c7_action_audit.csv`。参考为每个1 ms控制区间施加的参考，实际为该区间积分后的状态；所有对照使用相同时间对齐。

本轮没有改变2D动作权限，没有加入Delta Fz或新增传统控制器。只能对这些冻结轨迹、外载和执行器动态给出结论；不能推广为全空间、任意负载或实机安全保证。
''')
    history=read(RUN/'history_after.json')
    videos=[read(RUN/f'video_{kind}.json') for kind in ('dynamic_ppo_training','classical_vs_ppo','dynamic_disturbance_demo')]
    q=pd.DataFrame(decision['qualifying'])
    improves=lambda key:int((effects[key]>0).sum())
    save('66_Phase2C7最终验收报告.md',f'''# Phase 2C-7 最终验收报告

**{decision['status']}**。三个种子均完成有界真实训练；未自动启动后续研究。预检、模型、原始物理日志及失败记录均保存于`results/runs/phase2c7_20261107_v1`。

## 实际交付

- 46/46次真实联合预检通过；新58维Gymnasium环境、组合参考和单次共享预算协调器。
- 新训练{decision['new_training_decisions']}决策、{decision['new_training_physical_steps']}物理步、{sum(x['updates'] for x in diagnostics)}次Actor/Critic更新。
- {len(evaluation)}次独立冻结评价；完成{int(evaluation.success.sum())}次。
- 15000步零残差逐步等价；基线和最终PPO窗口分别运行1000步，两者均与无窗口积分状态逐样本一致。
- 原始全量回归379通过、1项历史依赖断言失败。该Phase 0测试要求没有PyTorch/SB3，与后续已安装并用于PPO的环境冲突；保留原测试及失败输出，没有将其改称通过。新增环境15项全部通过。另有训练/状态5项、视频3项证据测试全部通过，见`phase2c7_training_evidence_tests_02.log`和`phase2c7_evidence_tests.log`。首轮动作比较的float32/float64表示误差测试失败也保留，说明见64号报告。
- {history['matched']}个历史文件SHA256保持不变，历史Phase 2C-6仍为PARTIAL SUCCESS。

## 必答问题

1. 50 mm＋Roll/Pitch组合运动：已真实执行，最大预检单轴参考为0.25°，双轴采用旋转向量组合，未放大显示几何。
2. 外力矩：真实后六路接口施加，预检逐步核对世界wrench。
3. 最大验证幅值：本次既定动态组合中合力矩1.2 N·m；2.4与4.8未纳入本轮。
4. PPO网络：三个种子均实际完成累计32768决策和64次更新。
5. 动作有效性：非零随机动作引起实际六支链力变化；逐任务请求/接受幅度及投影率完整记录。
6. 最大姿态偏差：12个新PPO种子×任务配对中{improves('so3_peak_deg')}个低于同权限经典反馈，其余如实保留。
7. 最大角速度：12个配对中{improves('omega_peak_rad_s')}个降低。
8. 加载和卸载IAE：12个配对中{int(effects.both_iae_improved.sum())}个同时改善。
9. Z稳定性：12/12个配对的Z-RMSE没有恶化，改善约0.21%～1.37%，幅度较小；全部通过110%代价门槛。
10. XY代价：12/12个配对的最大单轴XY偏移均降低，约0.77%～19.26%；没有以明显XY恶化换取姿态收益。
11. 是否超过经典：按冻结共同判据，合格种子数{int(q.qualifies.sum())}/3；最终结论见报告首行。
12. 三种子一致性：不一致。双IAE同时改善次数分别为seed42 1/4、seed123 0/4、seed2026 1/4，不能宣称稳定收益或统计显著。
13. 旧PPO迁移：OldCurriculum42使用原52维观测完成4/4任务；相对经典对照，其平均加载IAE更高、卸载IAE略低，未得到两者同时更好的平均表现。
14. 不同速度：训练rise为5/6 s；最终包含5.5/6.5 s，均可安全完成，但未证明不同速度下均能改善姿态抗扰。
15. 不同扰动方向：Roll、Pitch和复合方向的收益不同，复合任务没有种子同时改善两个IAE窗口。
16. 新外载时序：冻结的新时序均完成；性能收益仍不稳定。未用最终时序结果选择模型。
17. Critic尺度：确有高损失与负解释方差区间；已记录价值目标和GAE，不能只凭大Loss判发散。
18. 历史状态/三维残差：本轮未实现。应先按误差及实际权限分析提出独立消融，不能直接增加动作范围。
19. 主要瓶颈：当前证据直接显示的是扰动方向及加载/卸载之间的收益权衡，传统基线误差已较小。样本覆盖、对隐藏执行器动态缺少记忆等是待验证解释，不能据本轮直接断言为根因。
20. 下一阶段：暂不建议直接扩为六维策略或追加大预算。若继续研究，优先对现有卸载反向超调做诊断，再预登记小规模历史观测二维策略对照；当前没有必须加入Delta Fz的证据。本轮暂停等待审核。

## Windows启动命令

```powershell
cd __PROJECT_ROOT__
# 冻结组合控制器，真实动力学：
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c7_policy.py --task C2
# 新PPO的独立确定性评价，非训练现场：
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/view_phase2c7_policy.py --task F_mix --model models/phase2c7/seed42/final.zip
```

窗口下拉框选择已验证任务和控制器，点Reset开始新的物理回合；Pause暂停/继续；关闭窗口保存并退出。左键拖动旋转视角，右键拖动平移，滚轮缩放。每个任务完成后保留窗口和曲线，Reset可重新运行。窗口默认无额外墙钟自动退出。右侧八组曲线使用真实单位，微小姿态误差通过曲线刻度观察。

训练视频是现场随机采样RGB；对比视频是单独确定性评价；传统演示单独标识。不可混用。

{table(pd.DataFrame([dict(video=PathLike(v['path']),frames=v['frames'],seconds=v['seconds'],sha256=v['sha']) for v in videos]))}

所有视频逐帧核对真实积分状态哈希、物理时间、图像变化并完整解码；机器人几何比例未修改。实际调用的命令和测试日志均保存在run及results目录。TensorBoard启动命令见64号报告。

## 未执行与限制

未训练六维策略、未增加Delta Fz、未做2.4 N·m动态强载专项、未扩到±0.35°、未增加训练预算、未做实机验证。碰撞覆盖仍为原审计代理集合；离散几何检查和有限仿真成功不构成连续或实机安全证明。一次随机预检不证明所有随机动作序列安全。最终测试虽独立于训练，安全预检已查看其基线响应，不能称完全盲测。
''')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('configs/phase2c7_*'),*ROOT.glob('models/phase2c7/*/final.zip'),
               ROOT/'src/rl/stewart_dynamic_attitude_env.py',ROOT/'src/workspace/dynamic_attitude_study.py',ROOT/'scripts/train_phase2c7_ppo.py']}
    write_json(RUN/'release_manifest.json',manifest)


def PathLike(value):
    return __import__('pathlib').Path(value).name


if __name__=='__main__':main()
