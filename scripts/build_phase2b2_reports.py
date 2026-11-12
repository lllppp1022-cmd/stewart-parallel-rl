"""Render Chinese reports from independently recomputed CSVs; does not simulate."""
import _bootstrap
import json,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha

def table(df,cols):
    def fmt(v):
        if pd.isna(v):return 'N/A'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df[cols].itertuples(index=False,name=None))+'\n'

def main():
    run=ROOT/'results/runs/phase2b2_20261029_v1';all_=pd.read_csv(ROOT/'results/phase2b2_all_trials.csv');f=all_[all_.stage=='frozen_evaluation'];r=all_[all_.stage=='robustness'];probe=pd.read_csv(ROOT/'results/phase2b2_probe_response.csv');v=json.loads((ROOT/'results/phase2b2_validation.json').read_text(encoding='utf-8'));known=json.loads((ROOT/'results/phase2b2_known_input_check.json').read_text(encoding='utf-8'))
    tests=ET.parse(ROOT/'results/phase2b2_pytest_acceptance.xml').getroot();s=tests.find('testsuite') if tests.tag!='testsuite' else tests
    assert int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0
    count=s.attrib['tests'];passed=int(count)-int(s.attrib.get('skipped',0));decision='PHASE 2C CONDITIONALLY BLOCKED'
    head='# Phase 2B-2：受限残差动作可控性与鲁棒性\n\n日期：2026-10-29。项目：`__PROJECT_ROOT__`。主实验：`results/runs/phase2b2_20261029_v1`。本阶段没有训练网络，没有封装Gymnasium。\n\n'
    design=head+'''## 控制路径与冻结边界

新增 `residual_action.py` / `residual_governor.py` / `phase2b2_experiment.py` 包装原C0/C3；未复制或改写其控制律。历史2902文件逐一SHA256一致，详见冻结清单及validation。1kHz物理与原控制周期不变，C3原Jacobian100Hz刷新。C3参数：观测带宽30rad/s、补偿增益1；C1-L泄放1/s。原weld、姿态、行程、条件数50、部分碰撞代理5mm保护不变；部分代理不能构成完整碰撞保证。

动作只写前六个力执行器；后六路外载由实验profile独立写入。输入非法、维度错误、非有限、Jacobian病态时保守拒绝/保护，不求伪逆制造结果。实际状态始终通过原mj_step积分；仅显示镜像和独立运动学检查允许设置scratch状态。

## 三维与六维动作

三维为世界坐标、upper_center参考点的 `[δFz(N),δMx(Nm),δMy(Nm)]`，嵌入 `[0,0,δFz,δMx,δMy,0]`，解 `J_l.T δF=W`。请求与实现分开记录；不以理论逆映射代替真实作用验证。六维直接按Leg1..6 MJCF原序输入支链力N。

20Hz每50个物理步决策；50Hz每20步决策。任务动作保持任务广义力请求，1kHz通过当前可用J映射；支链动作直接保持六力。两种保持语义有差别。正式六维反馈由同一个三维反馈在决策时映射并逐腿裁到±0.1N；这不是任意六维最优策略。自由六维物理耦合由12个单腿正负探针独立验证。

## 共享预算

原输出记F_base（C0或完整C3），原名义输出F_C0。只存在一个预算：`F_add_total=F_cmd-F_C0`，包含C3自身补偿及新增残差。

每个物理步、每支链将候选附加力投影到以下三个区间交集：

1. [-5,+5]N。
2. 上一步**实际合成附加力**±0.02N（dt=.001s，即20N/s）。
3. [-400-F_C0,+400-F_C0]N原生命令余量。

区间为空拒绝。实际原生力±400N不变。约束的是附加力，不是F_C0全部力变化率；总力变化率必须另看控制代价。零请求且原输出可行时原样返回base.copy，保留冻结控制器的浮点数值，不加多余积分。数值容差1e-10N与原C3检查量级一致，未放宽实验阈值。

全程零动作已严格复现旧轨迹。曾施加非零后再归零，仍可能存在共享限速尾迹，不能声称瞬时消除残差或重置观测器。F_residual_applied定义为最终cmd减冻结base，不等于总附加补偿；在投影时可以与请求不同。

## C3已知输入闭环

原C3一次command只执行一次观测器更新。若最终合成命令改变，包装器通过原observer.remember_applied_command记录当前测量坐标/速度及J.T@最终cmd，供下一步使用；不再次更新观测器，不改C3自身previous_compensation状态。全零保持原记忆写入路径，因此逐位等价。

此处已知的是下发命令，不是真实未知gain下的实际力。实际gain和完整真实状态仅用于安全和离线评价。不能以传入真实gain“修复”模型失配。实际相对作用额外记录为W_physical_*：由预积分状态的独立约束切空间映射与实际执行器力，减去同状态下基线命令经actual gain/native clip的反事实力。它只用于日志，不是向策略提供的观测。

## 候选动作与信息

非特权动作只接收当前/历史p、R、v、当前参考与C3可观测等效扰动。damping：Z误差250N/m、速度50Ns/m；倾转误差/角速度均2（相应Nm单位）。lead：对相邻决策的等效扰动差分乘-0.02s。任务反馈裁剪±[0.5N,0.025Nm,0.025Nm]。无扰动固定探针按预先冻结时钟.8～1.8s施加，与外载时间无关；ramp为预定义低频波形。lead/damping均是经典因果诊断，不是神经网络或新RL算法。

特权oracle读取当前实验载荷真值，1kHz给理想抵消请求，仍受共享预算。标记PRIVILEGED ORACLE DIAGNOSTIC，只有该stage允许创建，不能进入正式排序，且有信息/频率优势，不代表可学上界或严格理论最优。

## 日志语义

每回合保留config/plant/action_profile/initial_state/home_baseline、physics.csv.gz、458维完整integration_states.npz、events/safety_events、summary、原metrics及独立metrics_recomputed。记录12ctrl、实际force、qpos/qvel、时间、参考、p/R、传感器capture/read、请求/施加/共享补偿及广义力。CSV row为积分后状态，input_step/测量时刻对应此前积分输入；不混用前后时刻。预算字段只在budget_applicable=True解释，保护保持阶段不将占位值解释为合法残差。C1/C1-L/C2为原始控制器对照，不伪造其F_C0预算字段；其原积分限制不变。

恢复标准为位置≤.25mm且SO3≤.02deg连续.3s，需先真实越界；未越界=N/A，未恢复和提前截断分别标注。6s任务完成标准仍是原最后.5s位置<.75mm、姿态<.1deg且无保护失败。
'''
    design+='\n## 零残差与已知输入实测\n\n'+table(pd.read_csv(ROOT/'results/phase2b2_zero_residual_equivalence.csv'),['base','full_integration_state_bitwise_equal','all_physical_signals_equal','maximum_state_difference','same_terminal_reason'])
    design+=f'\n额外核对零动作保护事件相同。无外载时已知+0.5N任务Z输入，1.0～1.7s观测器Fz范围{known["observer_Fz_range_N"]}N，均值{known["observer_Fz_mean_N"]:.9f}N，没有误估成-0.5N外扰；修正实际命令记忆{known["corrected_memory_steps"]}次。\n'
    (ROOT/'docs/19_Phase2B2残差动作与约束设计.md').write_text(design,encoding='utf-8')
    study=head+'''## 先冻结再比较

新工况test_force8_time：Home、世界Fz=-8N，1650～3549输入步；test_periodic_roll：已有允许的小幅周期参考但新外载Mx=-.4Nm，1350～3949步。只用C3做安全预检，未据其误差调参。两者冻结后，参数仅由旧force10与roll05上的有限开发实验选择。开发共4种C3候选（lead/damping×20/50Hz），以两窗口位置IAE/.25+姿态IAE/.02的开发平均选lead20；未使用正式测试调参。控制代价另列，未伪称筛选了Pareto最优控制器。

每个正式新工况比较原C0/C1/C1-L/C2/C3、C0任务/支链damping、C3任务/支链lead，共18回合，全部完成。仅两个新场景、确定性运行，不声称统计显著、随机独立场景成功率或泛化证明。

## 正式峰值与控制代价

单位：位置mm，姿态deg，实际总力变化率N/s，平方积分N²s。N/A不替换成0。

'''+table(f,['task','variant','load_peak_position_mm','load_peak_orientation_deg','unload_reverse_z_mm','unload_reverse_roll_deg','max_force_rate_N_s','force_square_integral_N2_s'])
    study+='\n## C3窗口误差与共享限速\n\n'+table(f[f.variant.isin(['C3','C3_task','C3_leg'])],['task','variant','load_position_iae_mm_s','unload_position_iae_mm_s','shared_add_peak_N','shared_add_rate_peak_N_s','projected_fraction'])
    study+='''
8N场景lead任务残差将位置峰值0.101420降到0.100108mm（约1.29%），卸载Z反向0.100903降到0.099592mm（约1.30%）；加载/卸载IAE各降低约5%。总力峰值变化率约50.98325N/s基本相同，但力平方积分微增，不能宣称控制努力整体降低。周期场景峰值位置0.283303→0.284055mm、姿态0.015905→0.015941deg略恶化；共享附加变化率从12.145566升至20N/s。卸载Roll反向稍降但整体IAE增加。因此无跨场景全面收益。

20/50Hz没有普遍最优。开发force10上C0 damping20峰值力变化率38.787025、50Hz为51.029600N/s；误差非常接近。C3 lead20开发归一化得分.282938、50Hz为.286076；名义1kHzC3未被降频。开发方向改善不能等同有限频率Actor一定可学。

## 探针实际响应与非目标耦合

探针真实施加窗口为.8～1.8s，下面重新按该窗口计算，不能误用外载统一2～4s窗口。delta为与同一C0零输入在2s前的实际位置差，正负符号与实际响应对应。

'''+table(probe,['variant','action','peak_position_mm','peak_angle_deg','delta_x_peak_mm','delta_y_peak_mm','delta_z_peak_mm'])
    study+='''
Leg1+.5N时非目标Y偏移约.482mm，动作方向和编号确实有效，但自由支链输入明显耦合X/Y/Yaw。单腿±.1N已逐通道验证；只对Leg1额外测试+.25/+.5N，不推断其他腿或任意六腿同时组合的安全性。任务空间Fz也有动态Y耦合，力矩有横移，理论瞬时W的零分量不意味着闭环轨迹该轴始终为零。当前Z/Roll/Pitch任务优先考虑三维结构，但它不能主动修复非目标Yaw失配。

## 特权诊断（不得与正式策略排名）

'''+table(all_[all_.stage=='oracle_diagnostic'],['task','variant','load_peak_position_mm','load_peak_orientation_deg','unload_reverse_z_mm','max_force_rate_N_s'])
    study+='''
force10 C3原峰值.147895mm，oracle为.144897mm；卸载.147365→.144363mm。oracle强制总补偿趋于同一个理想量，所以C0/C3 oracle轨迹相同；这不是两个自主控制器泛化能力相同。roll05可达到.007587mm位置峰值。有限实验仅说明有受限物理改善空间，未知载荷的因果推断延迟、已占用限速预算仍限制实际收益。结果不证明RL必优，不将特权真值用于候选选择。
'''
    (ROOT/'docs/20_Phase2B2动作可控性与强基线研究.md').write_text(study,encoding='utf-8')
    robust=head+'''## 独立实际模型与信息隔离

每次创建新实际MjModel，仅在第一次积分前改副本的upper质量/惯量、局部COM或前六actuator gain。质量和惯量同比例改变，不改几何和历史MJCF。名义C3/C0模型及局部观测模型在90个配置内完全相同。实际参数变化写plant.json，未向策略接口暴露。改变实际模型的mj_setConst使用新scratch，不写live状态。参数物理合理但只代表小幅合成试验。

单因素26回合：质量/惯量+1/+2%；COM x+.1/+.25mm；六腿统一gain.99/.98；交替gain=1±.005/±.01。传感器部分对C0/C3配对：p标准差1/2μm、旋转1/2×10⁻⁵rad、速度1/2×10⁻⁵相应SI；平台测量延迟1/2/5ms。噪声seed42，本地生成器，同工况C0/C3的12维原始噪声序列逐位相同。各因素失败即不升更大等级，没有组合扰动，也没有扩大预算追求成功。

## 部分观测通道扰动

只扰动传给控制器的平台p/R/v。全链关节q/dq、计算Jacobian的名义几何反馈仍理想。真实状态用于独立安全与离线评价，不能传给新增决策器替代其测量。噪声没有硬件传感器依据；只能称合成候选与部分通道敏感性。C0主要关节PD因而对这些噪声/延迟几乎不变，这不能证明真实全传感器C0更鲁棒。延迟队列持过去样本，启动保持首样本；capture≤read且延迟为固定物理步，无额外mj_step或未来信息。

## 实际结果

下表未到加载期的失败峰值N/A，不能用截短RMSE对完整6s排序。

'''+table(r,['variant','success','failure_reason','simulation_time_s','load_peak_position_mm','peak_orientation_deg','observer_pre_bias_Fz_N','actual_command_force_peak_difference_N'])
    robust+='''
小幅质量/惯量改变C3加载前等效Fz偏置，+1%约-1.962N、+2%约-3.924N，符合额外重力方向；加载峰值未明显变坏。COM的两个小偏移未使测试失败，但不能外推3mm旧失败工况。统一gain.98时偏置约-10.222N，控制命令与真实力不同；加载峰值从名义.147895升至.152343mm。这是执行器失配和重力共同形成的等效扰动，不能把全部估计当真实外载。

交替gain±.5%完成但非目标姿态显著变大；±1%在.384s、尚未2s加载时Yaw=-.200308935deg，触发原±.2deg姿态限制。保存200步受限保持到.584s，最终Yaw约-.395282473deg，说明保持不能瞬时消除动量或静态失配。没有降低阈值，没有补造成成功；本次load/unload恢复均被右删失。该例不属于接触成功，也不能用“未加载所以无问题”排除。

5ms平台延迟使C3加载位置峰值.147895→.160570mm（约8.57%），仍完成；小噪声改变有限。并未发现这些候选合成噪声导致数值发散。只凭这个有限pilot不能给实际传感器工作上限。

## 研究价值与下一步条件

统一轻度增益失配/质量偏差是较合理的未来受限辨识或学习候选，配合有限延迟，但目前C3已很好处理，新增残差并未在这些失配工况中正式验证改善。交替gain±1%的失败主要在Yaw，当前三维Z/Roll/Pitch通道未证明可控，不能直接用作随机探索任务。

进入下一阶段前应先冻结一个安全、具有足够剩余误差的失配任务，检查局部可控性、动作可观察性和控制代价，并与受限经典补偿公平对照。不能通过扩大残差、放宽工作区或向Actor提供actual gain制造收益。未执行更大失配、组合失配、完整传感器噪声、硬件试验；本阶段不追加它们。
'''
    (ROOT/'docs/21_Phase2B2模型失配及观测鲁棒性.md').write_text(robust,encoding='utf-8')
    final=head+f'''## 验收结论

**{decision}**。动作接口有效、零残差逐位等价、预算/原生力及观察器闭环通过。有限因果残差仅有微小且不跨场景一致的改善；尚无模型失配下超过强C3的可靠额外收益，也未形成可安全随机探索且收益明确的任务。本结论是研究门禁，不是把工程可运行等同PPO有效。本轮完成后暂停。

## 实际执行清单

'''+table(all_.groupby('stage',as_index=False).agg(trials=('success','size'),completed_success=('success','sum')),['stage','trials','completed_success'])+f'''
90回合中89完整完成、1真实保护失败，共{v['research_physical_steps']}次1ms物理积分（534.584仿真秒）。这个汇总含探针、预检、开发、正式、oracle、鲁棒性，不作为统一任务成功率。64动作研究+26鲁棒性，恰好达到预先冻结90回合上限；没有无限搜索或额外更大失配实验。

另有观察器非侵入性2048步off/on各一次；首次受沙箱WinError5管道限制，已完成2048步off后退出，目录和错误记录保留；有界授权重试真实Qt+原生MuJoCo及RGB通过。因此观察器验证实际额外6144积分，不把它混作新科学回合或正式成功样本。自动测试内短动力学单独归测试成本。

## 独立复算与证据

`phase2b2_validation.json`逐一检查90回合：完整458维状态有限、CSV qpos/qvel/time与NPZ逐位一致、全频实际共享附加幅值/变化率、±400N原生命令与实际力、参数组隔离、延迟因果性、窗口误差；独立scratch前状态重算切空间广义力映射最大误差{v['kinematic_map_independent_scratch_max_error']:.3g}。完整6s零动作和保护事件与原C0/C3一致。历史{v['history_files_unchanged']}个文件SHA256未变。

全部原159测试和新增测试真实运行：**{passed}/{count}通过**。首轮183通过，最终新增6条验收证据测试后再跑。XML与日志在results，原两条pandas float_precision弃用警告保留；未修改旧测试来通过。独立复核脚本首次因读取actions_frozen中selected层级错误KeyError退出（40条后），修复验证脚本后v2全部90通过；不是动力学失败，旧日志保留。

录像18秒、1920×1080、25fps、450帧全部解码且450帧不同；来源是3条正式回合现场actual_rgb，而不是重积分/运动学播放。每帧qpos哈希、物理步/时间/载荷与CSV相符，曲线只显示过去。另检查记录帧支链实际位移非零。真实运动很小，保持物理尺寸，没有视觉夸大。视频比较C3/C3_task/C3_leg，无PPO标签或虚构RL训练。

## 二十项问题逐项回答

1. **零残差复现？** 是，C0/C3各6000步完整integration、12ctrl、力、位姿、执行器状态逐位一致，保护事件和终止原因相同。
2. **三维映射？** 世界upper_center处J_l.T解算，通过单位/方向单测、实际映射及真实力复核；受预算影响不保证W_requested=W_realized。
3. **六维真实响应？** 六腿各±.1N逐通道测试有不同有符号位移，真实mj_step积分，非qpos动画。
4. **共享预算？** 是，C3原补偿和新增请求合计每腿±5N、20N/s，不叠加成±10N。
5. **有益方向？** 因果lead的目标Z修正使8N加载/卸载峰值微降；倾转开发也有改善，不能据此断言全局方向都好。
6. **恶化耦合？** 单腿力明显引入XY/Yaw；Leg1+.5N约.482mm额外Y偏移，任务力矩也有横向耦合。
7. **20/50Hz？** 不普遍单调；开发lead20略优，C0 damping50可能增加总力变化率，1kHz基础控制不变。
8. **卸载超调？** 8N Z约降1.30%；周期Roll仅微降且整体IAE恶化，不宣称全面解决。
9. **不增加力变化率改善？** 8N总力变化率基本持平有微小峰值改善，但平方积分略增；周期共享补偿变化率增加，不能宣称无代价Pareto优势。
10. **特权空间？** oracle在受限预算下有更小部分误差，有限诊断有物理空间，但非理论最优证明。
11. **是否需要真值？** 正式lead/damping仅用当前/历史观测，无载荷时刻；oracle使用真值单列。尚未证明可因果学到oracle收益。
12. **质量/质心敏感？** 小幅1/2%与.1/.25mm均完成，质量产生重力估计偏置；不能外推大失配。
13. **gain敏感？** 是，统一.98轻退化，交替±1%在外载前Yaw保护失败。
14. **噪声/延迟敏感？** 部分平台小噪声影响很小，5ms延迟峰值升约8.57%；全链仍理想，非硬件鲁棒性证明。
15. **未来任务？** 优先安全小幅统一增益/质量失配与有限延迟，需先确认残余收益；不直接对Yaw失败任务随机探索。
16. **三/六维？** 三维更贴当前任务且少耦合。匹配反馈六维几乎相同，不是任意六维策略强弱结论。
17. **动作范围依据？** 已实测任务±.5N/±.025Nm、各腿±.1N及Leg1正.25/.5N；不是任意同时组合或±5N随机安全证明。
18. **新违规？** 1例交替gain±1%姿态保护，完整保存；其余89回合完成。本轮没有降低阈值。
19. **值得学习的问题？** 有失配和延迟敏感性，但未证明新增残差在这些条件下的可靠优势；强C3使名义可改进空间很小。
20. **进入Gym/PPO？** 有条件阻塞，须先确认可控、可观测、收益足够且安全的失配目标及代价门禁。本轮不创建Gym或训练。

## 实际测试过的PowerShell入口

下面在项目根运行。回放是历史真实动力学记录，窗口会标识ARCHIVE REPLAY，不运行新实验。播放按钮控制归档时间，原控制面板交互不能改变历史记录。原生动态窗口实际验收由observer脚本打开，固定2048步并自动正常关闭；这是有界验证，不替代原无限交互窗口。

```powershell
Set-Location __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/watch_phase2b2.py --trial results/runs/phase2b2_20261029_v1/frozen_evaluation/test_force8_time/C3_task --validate-seconds 2
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/validate_phase2b2_observer.py
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/compare_phase2b2_residuals.py --run-root results/runs/phase2b2_20261029_v1
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts/validate_phase2b2.py --run-root results/runs/phase2b2_20261029_v1
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m pytest tests -p phase2a_pytest_temp_plugin
```

持续观看时去掉`--validate-seconds 2`，保留原回放窗口直到用户关闭；去掉参数的完整命令本轮未另做无限时长运行。主实验两脚本已经真实执行，不建议为看视频重跑90回合。若复现实验必须用全新run-root，原目录存在会拒绝覆盖。MP4直接在播放器打开，无需安装RL。

## 交付与未执行事项

设计/动作/鲁棒性详见docs19～21；核心表为phase2b2_c3_comparison.csv、action_sensitivity.csv、probe_response.csv、robustness.csv、oracle_diagnostics.csv，完整90行all_trials.csv。raw回合均含config、plant、action_profile、summary、原始状态和独立复算，另有decision_effectiveness逐决策投影/无效/施加延迟统计。

未执行：Gymnasium、PPO/SAC/BC；失配下残差正式干预；任意同时六腿动作箱安全验证；更大或组合不确定性；真实硬件与完整传感器实验；形式化/完整碰撞安全保证。本轮没有把这些列为已完成。无新的训练模型/checkpoint可交付，因为未训练网络。观察器首次权限错误和validator首轮错误均保留；最终有效验收已通过。

不自动开始Phase2C。建议用户先审阅周期场景退化、Yaw失配失败和控制代价，决定是否值得再定义更有解释力的小型学习任务。
'''
    (ROOT/'docs/22_Phase2B2验收报告.md').write_text(final,encoding='utf-8')
    write_json(ROOT/'results/phase2b2_decision.json',dict(decision=decision,rl_training=False,tests_passed=passed,research_trials=90,physical_steps=v['research_physical_steps'],history_files=2902,reasons=['Small non-uniform nominal benefits','No demonstrated residual improvement under uncertainty','Non-target yaw failure not established controllable by 3D action']))
    paths=list((ROOT/'docs').glob('*Phase2B2*.md'))+list((ROOT/'assets/plots').glob('phase2b2*.png'))+list((ROOT/'results').glob('phase2b2*.csv'))+[ROOT/'assets/videos/stewart_phase2b2_residual_action_validation.mp4']
    write_json(ROOT/'results/phase2b2_delivery_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],decision=decision))
    print('Chinese reports generated from actual CSV, tests',passed,'decision',decision)

if __name__=='__main__':main()
