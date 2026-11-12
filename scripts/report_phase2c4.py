"""Final Chinese reports and artifact manifest from actual frozen-run evidence."""
import _bootstrap,json,platform,sys
from pathlib import Path
import numpy as np,pandas as pd
import torch,mujoco,stable_baselines3,gymnasium
from src.model import ROOT,sha,write_json
from src.rl.stewart_large_motion_env import RUN
from src.rl.large_motion_observations import FIELDS

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def table(d):
    def fmt(v):return '—' if pd.isna(v) else f'{v:.9g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def put(name,body):(ROOT/'docs'/name).write_text(body,encoding='utf-8')

def main():
    env=read(ROOT/'results/phase2c4_env_validation.json');protocol=read(RUN/'protocol_frozen.json');cfg=protocol['ppo'];reward=protocol['environment']['reward'];summaries=pd.read_csv(ROOT/'results/phase2c4_training_summary.csv');comparison=pd.read_csv(ROOT/'results/phase2c4_controller_comparison.csv');improvement=pd.read_csv(ROOT/'results/phase2c4_improvement_vs_FFVA.csv');action=pd.read_csv(ROOT/'results/phase2c4_action_audit.csv');decision=read(ROOT/'results/phase2c4_final_decision.json');safety=read(ROOT/'results/phase2c4_safety_audit.json');testlog=(ROOT/'results/phase2c4_pytest.log').read_text(encoding='utf-8',errors='replace');testline=next((s.strip() for s in reversed(testlog.splitlines()) if ' passed' in s),testlog[-300:]);old=read(RUN/'history_freeze.json');changed=[k for k,h in old.items() if not (ROOT/k).exists() or sha(ROOT/k)!=h];assert not changed
    versions=dict(python=sys.version,platform=platform.platform(),mujoco=mujoco.__version__,torch=torch.__version__,sb3=stable_baselines3.__version__,gymnasium=gymnasium.__version__,numpy=np.__version__,pandas=pd.__version__);write_json(RUN/'environment_versions.json',versions);write_json(RUN/'history_preservation.json',dict(passed=not changed,files=len(old),changed=changed));proof=read(RUN/'validation/zero.json');reward_audit=read(RUN/'reward_training_independent.json');assert reward_audit['passed']
    scope='保护仍为原大工作区、每链相对C0总附加±5N/20N/s、原生±400N、weld及Jacobian和部分几何代理检查。887个网格凸包配对、103个连接组件排除和30个无原生硬止挡被动铰链的局限保留；无接触不等于全几何安全或实机认证。'
    commands="""```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\\miniconda\\envs\\stewart_rl\\python.exe'
# 新建独立日志，真实动力学运行最后checkpoint；不是训练现场回放
& $py scripts/view_phase2c4_policy.py --method PPO --model models/phase2c4/seed42/final.zip
# 原FF-VA对照
& $py scripts/view_phase2c4_policy.py --method FF-VA
# 只看TensorBoard，不启动训练
& $py -m tensorboard.main --logdir results/runs/phase2c4_20261105_v1/training --host 127.0.0.1 --port 6034
```
本轮已实际启动6034并验证HTTP200；若已运行，不重复占端口。暂停只暂停物理时钟；Reset显式新建MjData和新episode，不在运动中写qpos；Stop关闭、归档。完整15秒结束后停留显示，墙钟600秒上限。受限CLI只提供固定已预检任务，不允许界面扩载。

有界GUI开/关验证实际命令：
```powershell
& $py scripts/view_phase2c4_policy.py --method FF-VA --validate-steps 2000 --compare-headless
```
新PPO入口的有界加载检查命令见运行清单。默认完整GUI交互未作为额外训练样本。窗口为单线程Qt＋MuJoCo Renderer，物理与显示所有权一致；不是原生GLFW Viewer。历史2C3B窗口保留。

以下PPO加载命令分别实际运行至2000和8000物理步（8000步已覆盖真实50mm上升及顶部）：
```powershell
& $py scripts/view_phase2c4_policy.py --method PPO --model models/phase2c4/seed42/final.zip --validate-steps 8000
```
这是GUI预算截断的部分评价，不算完整15秒成功。完整结果使用正式24条全轨迹评价。
"""
    put('47_Phase2C4大位移强化学习环境设计.md',f'''# Phase2C-4：50mm残差PPO环境设计

日期2026-11-05。工程`{ROOT}`，独立实验`{RUN}`。用户本轮明确授权带着已知XY耦合代价研究有限Z残差，未削弱FF-VA、未扩成六维、未修改旧权重。

## 真实动力学与动作

复用StewartResidualEnv的Gym/CSV/关闭契约、LargeOwner、完整链式参考、原FF-VA包装、SharedBudgetGovernor、ActuatorLag、外载与安全流程。物理1ms、原C4/FF控制1kHz；动作20Hz，每决策50个真实mj_step。L50-T6完整Home→上升6秒→顶部保持→下降6秒→Home，总15秒，完整300决策。不是6秒episode。

动作1维Box[-1,1]，delta_Fz=0.1*a N，原完整J解`J.T*Fleg=[0,0,deltaFz,0,0,0]`。利用已冻结FF包装的probe端口在唯一共享governor前合成，不额外增加预算；C4最终命令记忆包含FF和PPO。`ff_probe_effect_N`仅是同一状态下无RL反事实命令差，不等于独立闭环或真实力传感。Fcommand、Fdrive及原生force分别保存，Actor只看已知命令，不看后两者。

固定工况：tau5ms、世界Z−5.5N、加载5.2s、卸载11s，原Home及姿态。原六执行器与后六外扰通道严格隔离，所有运动通过ctrl和mj_step。

## 零动作与Reset

完整{proof['paired_physical_steps']}步逐位比较qpos/qvel/act/mocap/ctrl/其他必要积分状态、实际力、drive、C4 observer、共享governor、参考、载荷及失败状态；完整积分档案逐位相同。第二episode前50步也逐位复现第一episode，Reset重新构建物理和控制状态、局部随机数、观测历史、上一动作与外载时钟。

第一轮实现存在一次回合末观测指针错误：原runner在完成后清空active，新观测错误访问它。保留失败CSV/控制台，改为独立已知参考指针后完整重验；没有修改物理控制器，也没有在错误环境上开展PPO。见`validation/failed_terminal_observation_attempt1`。

## 53维因果字段

{table(pd.DataFrame(FIELDS,columns=['字段','维度']))}

单帧53维；3帧159维只保留接口、未混入首轮训练。实际平台/支链反馈按理想可测状态处理，与实机传感器不等价。没有tau、外载真值/时序开关、隐藏drive、实际执行器力或未来物理状态。已知当前参考解析导数合法；observer是因果历史摘要，因此“单帧Actor”不代表系统没有任何历史信息。

尺度：位移Z/目标0.05m，slide0.06m，线速度0.03m/s，角速度0.05rad/s，参考加速度0.03m/s²，Z误差1e−5m、XY误差5e−5m，角误差0.000349rad，已知命令减Home后/5N，observer力/20N、力矩/0.8Nm，参考FFz/20N，接受总附加/5N，上次接受Z残差/0.1N。固定缩放、±10裁剪，无VecNormalize，评价不更新统计。随机预检观测裁剪率为0，未沿用旧1mm行程尺度。

## 奖励

冻结候选配置：`{json.dumps(reward,ensure_ascii=False)}`。每1ms计算Z、XY、SO(3)、请求动作和总附加命令变化率代价，物理积分乘0.001/0.05。cap100限制单决策最大物理代价；各分量按相同比例裁剪，原始实际力变化另外评价。不得把Box动作裁剪率与PPO比率clip_fraction混为一谈。

对固定正/负/随机/交替动作真实完整轨迹比较奖励，并用相同其他状态的误差干预验证Z改善增益、横向恶化惩罚。失败惩罚超过剩余最大折扣代价加原保护200ms上界，避免提前失败获利。独立向量方程复算物理分量与每决策积分，{len(reward_audit['trials'])}条训练episode档案通过；物理reward与SB3因TimeLimit加入bootstrap后的effective_rewards分别存在rollout NPZ，不能混为一谈。

正常完整15秒truncated=True并独立complete/success；安全失败terminated，执行原真实保护Hold并计入物理步；训练预算结束中途episode标incomplete，不能算失败或完成。正常尾部TimeLimit的价值bootstrap沿用原SB3语义。

{scope}
''')
    put('48_Phase2C4_PPO训练与超参数.md',f'''# Phase2C-4：PPO真实训练与超参数

## 冻结预算及生效参数

{table(pd.DataFrame([dict(parameter=k,value=json.dumps(v,ensure_ascii=False)) for k,v in cfg.items()]))}

小型Tanh两层32单元Actor/Critic，Actor最终均值输出层置零，高斯log_std初始−1，随机探索仍保留，初始确定性均值接近零残差。使用版本`{json.dumps(versions,ensure_ascii=False)}`，CPU、Torch单线程；没有超参数搜索。

## 真正运行的三个种子

{table(summaries[['seed','decisions','physical_steps','updates','optimizer_steps','complete_episodes','safety_terminations','wall_s','model_sha256']])}

合计{int(summaries.decisions.sum())}次PPO决策、{int(summaries.physical_steps.sum())}次真实MuJoCo积分。42号种子256步冒烟是4096的前缀，接着3840步，不重复计费；另外两个各4096，均16次rollout优化。每回合300决策，256更新边界不等于回合边界，同一episode会跨策略版本。每个种子保存initial、最后模型及每256步checkpoint；按最后4096模型评价，未筛选最佳种子或最佳测试checkpoint。

冒烟保存/重载参数与确定性动作一致；在进行重载验证时保存并恢复Python/NumPy/Torch RNG，避免诊断消耗采样随机流。Actor和Critic每次参数差范数均记录并验证非零。每个优化期间实际物理步数不变。

{table(action)}

## 看训练与复盘

TensorBoard：`http://127.0.0.1:6034/#timeseries`，日志目录`{RUN/'training'}`。采样窗口CSV含raw Gaussian、Box后动作、请求Fz、接受命令、预算投影、Z/XY/SO(3)、奖励分量。`ppo_update_events.csv`含Policy/Value loss、KL、PPO clip、explained variance、GAE及价值目标分布、参数变化和checkpoint哈希；`rollout_*.npz`保留训练buffer，真实物理日志逐1ms保存。

图：`phase2c4_learning_curves.png`、`phase2c4_action_vs_applied.png`、`phase2c4_update_timeline.png`、`phase2c4_reward_components.png`，都在assets/plots。完整回合回报与末尾未完成前缀分开；不以提前结束后较低RMSE证明优势。

真实训练录像取42号种子episode0/6/12，早中后期预先指定，不按表现挑选。原始RGB由训练进程实际积分时采集，其他episode默认不渲染；视频是这些RGB的后期加字，绝非最终策略确定性重新生成。逐帧qpos哈希与实际积分状态比对。ACTUAL STOCHASTIC PPO TRAINING与独立评价明确区分。加载真值只用于离线视频/指标，不输入Actor或奖励。

可见的50mm运动主要由冻结FF-VA基线跟踪参考完成，PPO只施加微小附加力；不能把整段大位移运动算成PPO独自学会。微小残差效果靠曲线和严格配对指标判断，不放大机器人实际位移。物理逐步动作有效率见`phase2c4_physical_action_effectiveness.csv`，接受的是已知命令差，不是直接测得的独立RL物理力。

训练命令实际运行过`train_phase2c4_ppo.py --seed 42/123/2026`；原路径存在时会拒绝覆盖，不要直接重跑这些命令写历史目录。用户复现实验须显式设置新的独立run及models目录、复制冻结配置；本轮没有额外授权更长训练。

程序已提供`--output-root`和`--model-root`用于将同一冻结实验写入新目录；参数入口实际执行过`--help`，以下完整重训示例没有再次执行，不计为本轮结果，也不会自动启动：
```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\\miniconda\\envs\\stewart_rl\\python.exe'
$newStudy = Join-Path '__PROJECT_ROOT__\\results\\runs' ('phase2c4_reproduce_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py scripts/train_phase2c4_ppo.py --seed 42 --output-root "$newStudy/training" --model-root "$newStudy/models"
foreach ($trialSeed in @(123,2026)) {{
    & $py scripts/train_phase2c4_ppo.py --seed $trialSeed --output-root "$newStudy/training" --model-root "$newStudy/models"
}}
```
同机CPU短采样的种子复现已测试，未额外重复4096决策全训练，也不声称跨硬件完全逐位复现。权重可以直接用Viewer加载，无需重训。

{scope}
''')
    cols=['task_id','method','success','z_rmse_mm','position_rmse_mm','load_IAE_mm_s','unload_IAE_mm_s','xy_peak_mm','so3_rmse_deg','max_force_N','max_force_rate_N_s','force_energy_N2_s','additional_force_peak_N']
    put('49_Phase2C4传统控制与PPO独立评价.md',f'''# Phase2C-4：传统控制与PPO独立评价

## 冻结的评价协议

最终checkpoint按最后4096预先选择，三个种子都保留。训练内固定任务和两个新负载组合分开。新组合E1/E2在训练前冻结，并由FF-VA完整动力学预检仅确认安全，没有借此修改奖励/控制器或选择模型。它们不是全新机械模型，而是未用于训练的新时序/幅值组合；原历史T4/T6/T8/T10没有重新称作盲测。训练和模型选择冻结后才对全部方法开启正式评价。

新组合：`{json.dumps(protocol['evaluation']['final_tasks'],ensure_ascii=False)}`。同Home、T6轨迹、物理模型、原约束、每步积分。PPO用确定性高斯均值经Box裁剪；initial_stochastic明确为未训练42号Actor随机采样对照。FF-VA+zero独立实际重跑，不是凭名称复制结果。

## 全部24次评价

{table(comparison[cols])}

## 相对最强FF-VA的绝对/相对变化

正数表示改善，负数表示恶化；mm、mm·s、deg及N/s见列名。不是只与C4比较。下表同时列训练内与独立组合，CSV保留全部3D、非目标方向、实际力及能量代价。

relative_gain列以比例表示，0.01=1%；absolute_gain保留物理单位。123号种子的独立任务全程Z绝对改善约0.00024mm，即0.24微米量级；不能只用百分比夸大工程意义。

{table(improvement[['task_id','method','seen_training_task','absolute_gain_load_IAE_mm_s','relative_gain_load_IAE_mm_s','absolute_gain_unload_IAE_mm_s','relative_gain_unload_IAE_mm_s','absolute_gain_z_rmse_mm','relative_gain_z_rmse_mm','absolute_gain_xy_peak_mm','absolute_gain_so3_rmse_deg','pass_continue']])}

## 有限多工况准入

冻结门槛：至少2/3种子在两个独立组合同时满足加载/卸载IAE各改善≥1%、全程Z不恶化、XY峰值≤FFVA*1.05+0.005mm、SO(3)≤FFVA*1.05+0.0001deg、力变化率≤FFVA*1.1，且无安全失败、参数真实更新。该门槛是探索性工程筛选，不是统计显著性检验。

实际每种子门槛：`{json.dumps(decision['individual_seed_gate'],ensure_ascii=False)}`；是否允许多工况：{decision['multi_condition_gate_passed']}。多工况实际执行={decision['multi_executed']}。未通过时不追加随机化训练、历史帧消融或预算，保留负结果。若未来继续，先只改变已预检的加载时刻，再逐项预检幅值/tau，不能把tau10无载和tau5负载结果拼成tau10+负载安全证据。

## 结论边界

**{decision['decision']}**。三个训练种子和两个独立负载组合非常有限，确定性重复不是独立随机场景，不声明统计显著。微米或更小的绝对误差变化需结合原本约0.01mm的强基线解释，不把百分比等同工程价值。即使PPO优于原C4/fourDlead也不能据此宣称超过强FF-VA；Z改善若伴随XY/姿态恶化，只能报告控制取舍。

作用时序见`phase2c4_transient_timing.csv`：本组FF-VA总Z误差峰值在载荷事件后约72～73ms，20Hz Actor在事件后首次新反馈约50ms，仅有一次新的决策机会，而C4在此期间以1kHz运行。±0.1N请求约为5.5N外载幅值的1.8%，不是性能改善上限，但说明权限很小。此证据支持研究时间尺度/信用分配和observer交互，不支持盲目追加样本或未经验证放大动作。窗口动作符号/幅值描述见`phase2c4_action_window_analysis.csv`；符号相关不是因果识别。

E1任务动作描述：

{table(pd.read_csv(ROOT/'results/phase2c4_action_window_analysis.csv').query('task_id == "E1_newtiming"')[['method','event','mean_requested_Fz_N','mean_accepted_command_Fz_N','opposite_error_sign_fraction']])}

42号在两个窗口平均请求均为负，加载获益但卸载退化；123号在这些窗口呈异号、约95%的样本请求方向与当前Z误差相反。这是闭环轨迹上的描述，不能单独证明识别了真实外载。控制正确方向应结合当前误差、原observer和FF，而不是机械地给负外载配一个正残差。

{scope}
''')
    unseen=improvement[~improvement.seen_training_task];gaincols=['relative_gain_load_IAE_mm_s','relative_gain_unload_IAE_mm_s','relative_gain_z_rmse_mm'];seedmeans=unseen.groupby('method')[gaincols].mean()*100;seedmeans.columns=['加载IAE改善百分比','卸载IAE改善百分比','全程Z_RMSE改善百分比'];qa=[
    ('50mm环境真实有效？','Gym检查、完整15000步零动作配对、真实随机动作和GUI对照均通过；真实ctrl和mj_step，非qpos动画。'),
    ('Actor/Critic优化？',f'三个种子实际{int(summaries.updates.sum())}次rollout更新；全部参数变化非零，checkpoint可重载。'),
    ('零动作严格复现？','完整积分、observer/governor/lag/12ctrl逐位一致，Reset前50步再次一致。'),
    ('奖励与目标一致？','独立复算、单项敏感性及失败折扣上界通过；但平方误差全程目标与1秒IAE并不完全相同，不能预设学会抗扰。'),
    ('识别加载/卸载？','123号在所测加载/卸载窗口均改善、请求呈异号；这支持有效反馈响应，仍不足以证明辨识了未知负载。42号存在相反窗口取舍，Actor没有外载标志。'),
    ('动作方向/幅值？','请求±0.1N；E1的123号约95%样本与当前Z误差反向，42号窗口平均请求同为负、卸载退化。实际接受和raw/Box另行审计，不简单按外载正负判动作。'),
    ('经常被预算抑制？',f'见动作表，预算投影率范围{action.budget_projection_fraction.min():.6g}～{action.budget_projection_fraction.max():.6g}，与高斯Box裁剪率不同。'),
    ('加载IAE改善？','42和123在两个独立组合均改善；2026一个改善、一个恶化。绝对变化全列49号报告。'),
    ('卸载IAE改善？','123和2026在两个独立组合均改善，42在两个组合均恶化；加载改善不能替代卸载结论。'),
    ('全程Z改善？','123和2026在两个独立组合均改善，42均恶化；平均百分比见表，绝对收益很小。'),
    ('XY是否恶化？','42的独立任务XY峰值轻微增大，123和2026轻微减小；原FF-VA横向耦合仍存在，未解决。'),
    ('姿态是否恶化？','42的SO(3)轻微增加，其余两种子轻微降低，绝对变化很小；RPY日志保留，未触原姿态保护。'),
    ('控制力变化？','确定性PPO的峰值变化率相对FF-VA仅有极小变化，均未超过冻结的10%不利门槛；实际力/能量单列，附加20N/s不等于全力20N/s。'),
    ('安全失败？',f'训练安全终止{int(summaries.safety_terminations.sum())}次，评价成功{int(comparison.success.sum())}/24；测试故障注入不冒充真实失败。'),
    ('相对C4？','匹配任务的PPO全程Z误差均更小，但主要收益来自FF-VA，不能全部算成PPO贡献。'),
    ('相对fourDlead？','匹配任务全程Z误差均更小，同样主要来自FF-VA，不能替代最强基线对照。'),
    ('相对FF-VA？',decision['decision']+'；123通过共同门槛，2026加载收益不一致，42有卸载/全程退化，未形成跨种子稳定优势。'),
    ('三个种子一致？',f'共同门槛{decision["individual_seed_gate"]}，保留个体差异，不筛种子。'),
    ('已见和独立差异？','训练内train_fixed与E1/E2分开，加载时序/幅值有限泛化，非大范围鲁棒证明。'),
    ('单帧是否足够？','首轮仅验证可采样优化；包含因果observer状态，未证明一般任务Markov充分性。'),
    ('是否加历史？','159维接口保留但未自动运行；需在同物理预算的独立消融中检验，不能预设收益。'),
    ('是否多工况？',f'冻结门槛通过={decision["multi_condition_gate_passed"]}，实际执行={decision["multi_executed"]}；不达门槛就停止。'),
    ('20/50Hz？','本轮仅20Hz。可作为后续独立假设，同物理暴露、折扣/GAE时间尺度匹配，当前未验证。'),
    ('扩Fx/Fy？','未获本轮验证，不直接扩大Actor；先诊断名义耦合、控制权限与预算，再独立预检。'),
    ('扩大正式预算？','本轮不自动扩大；按最终决策先处理设计问题或停止扩展，不能用追加训练隐藏负结果。')]
    qtext='\n'.join(f'### {i}. {q}\n\n{a}\n' for i,(q,a) in enumerate(qa,1))
    put('50_Phase2C4验收报告.md',f'''# Phase2C-4 验收报告

2026-11-05，工程`{ROOT}`，实验`{RUN}`。

## 真实完成及研究判断

**{decision['decision']}**。本轮完成环境、完整零动作等价、随机预检、真实三种子PPO、24次独立闭环评价、真实训练/评价RGB视频及受限Viewer。没有预设PPO应胜出，原FF-VA和XY代价保留。

{table(summaries[['seed','decisions','physical_steps','updates','optimizer_steps','complete_episodes','safety_terminations']])}

实际训练{int(summaries.decisions.sum())}决策、{int(summaries.physical_steps.sum())}个1ms积分，256烟雾已包含42号预算，没有免费重复前缀。预检、评价、GUI和自动测试是额外独立成本，不混入训练样本。

独立任务各方法平均相对改善（正为改善，基线FF-VA）：

{table(seedmeans.reset_index())}

绝对收益及全部24行在49号报告和CSV中。通过环境/优化验收不等于控制性能成功。多工况门槛通过={decision['multi_condition_gate_passed']}，实际执行={decision['multi_executed']}；没有绕过门槛启动随机化训练。

## 工程验收与追溯

完整pytest真实输出：`{testline}`。原历史xfail单列，未计普通通过。完整日志`results/phase2c4_pytest.log`。另有训练前12项检查通过；故障注入只用于单元测试终止语义，不计真实机器人安全失败。

{len(old)}个冻结历史文件SHA未变，结果`history_preservation.json`。全部实际模型哈希见training_summary及每个ppo_update_events；无覆盖历史模型。原模型/前馈参数/预算/工作区未放松。原始CSV可独立复算；`reward_training_independent.json`验证全部物理奖励及每决策积分。评价Z与预算用独立向量公式复算，安全审计通过={safety['passed']}。

首轮终止观测指针错误及其失败物理日志归档，修复后重新完整验证再训练。没有用直接写live qpos隐藏问题。Reset创建新的独立物理对象，GUI暂停/重置/退出均不额外生成策略训练动作。

过程偏差如实记录：启动门禁先完成在线公式敏感性、物理分量积分与回合奖励核对；更严格的独立向量公式复算在冒烟首轮期间补完，随后又复算全部训练档案，均通过，而非全部在首次采样前完成。未据此重训或追加预算。收尾补齐Gym `info.is_success`元数据，沿用原最后0.5秒<0.75mm/<0.1deg判据，不改变训练观测、奖励、动作、物理或done。详见`implementation_notes.json`。

## 可直接观看与复盘

{commands}

实际训练视频`assets/videos/stewart_phase2c4_ppo_training.mp4`，独立评价视频`assets/videos/stewart_phase2c4_ppo_evaluation.mp4`，均按现场RGB后期合成、2倍播放，逐帧qpos SHA对齐完整积分档案并验证解码/真实50mm位移。训练episode跨256更新边界按每帧真实policy_version展示，未假设整个episode策略固定。训练末尾未完成prefix不做完整成功视频。

只做离线分析的命令实际执行：`analyze_phase2c4_training.py`、`audit_phase2c4_reward.py --training`、`render_phase2c4_video.py training/evaluation`。不自动再次learn。所有新代码、模型、配置、报告和原始轨迹列入`results/phase2c4_artifact_manifest.json`。

## 25项问题

{qtext}

## 未完成与边界

未开展十万步训练、SAC/TD3/BC、全6DoF残差、Fx/Fy探索、实机辨识/部署或安全认证。未把旧3mm策略直接放到50mm。159维历史/20vs50Hz/更广泛随机化如未执行就保留为未来假设，不能写成结果。新的独立任务仅两个组合，不能泛化为真实液压系统或所有负载扰动。

{scope}

本轮完成后暂停，不自动追加算法或训练预算。
''')
    paths=list(RUN.rglob('*'))+list((ROOT/'models/phase2c4').rglob('*'))
    for folder,pattern in [('scripts','*phase2c4*.py'),('configs','phase2c4*'),('tests','*phase2c4*.py'),('results','phase2c4*.*'),('assets/plots','phase2c4*.png'),('assets/videos','stewart_phase2c4*.mp4'),('docs','4[7-9]_Phase2C4*.md'),('docs','50_Phase2C4*.md')]:paths+=list((ROOT/folder).glob(pattern))
    paths+=[ROOT/'src/rl'/p for p in ['stewart_large_motion_env.py','large_motion_observations.py','large_motion_rewards.py','large_motion_action_adapter.py']];files={}
    for f in sorted(set(paths)):
        if f.is_file() and '__pycache__' not in f.parts and f.name not in ['phase2c4_artifact_manifest.json','phase2c4_manifest_validation.json','phase2c4_tensorboard_console.log']:files[str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    write_json(ROOT/'results/phase2c4_artifact_manifest.json',dict(stage='phase2c4',run=str(RUN),files=files,history_unchanged=True,final_decision=decision['decision']));print('REPORTS47-50 generated, manifest',len(files),'files')
if __name__=='__main__':main()
