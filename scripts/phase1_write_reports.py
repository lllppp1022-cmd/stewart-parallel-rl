import _bootstrap
import json,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha

def read(name):return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def markdown(df,cols):
    text='| '+' | '.join(cols)+' |\n|'+ '|'.join(['---']*len(cols))+'|\n'
    for row in df[cols].itertuples(index=False,name=None):text+='| '+' | '.join(f'{v:.8g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |\n'
    return text
def main():
    table=pd.read_csv(ROOT/'results/phase1_controller_comparison.csv');pooled=pd.read_csv(ROOT/'results/phase1_nominal_pooled_metrics.csv');force=pd.read_csv(ROOT/'results/phase1_force_mapping_validation.csv');fine=pd.read_csv(ROOT/'results/phase1_timestep_comparison.csv')
    collision=read('phase1_collision_audit.json');preserve=read('phase1_phase0_preservation.json');manifest=read('phase1_suite_manifest.json');video=read('phase1_video_validation.json');delivery=read('phase1_video_delivery_manifest.json');gui=read('phase1_gui_validation.json');config=json.loads((ROOT/'configs/phase1_protocol.json').read_text(encoding='utf-8'));tests=ET.parse(ROOT/'results/phase1_pytest.xml').getroot().find('testsuite')
    assert len(table)==42 and not preserve['changed'] and tests.get('failures')=='0' and tests.get('skipped')=='0'
    nominal_main=table[(table.condition=='nominal')&(table.timestep_s==.001)];perturbed=table[table.condition!='nominal'];fine_rows=table[table.timestep_s==.0005]
    completed=int(table.success.sum());failed=int((~table.success).sum());n_nominal=int(nominal_main.success.sum());n_perturbed=int(perturbed.success.sum());n_fine=int(fine_rows.success.sum())
    root='''# Stewart Phase 1：六支链力控制与平台动力学跟踪

本阶段完成三种传统控制器、42次有限动力学比较、力映射和数值精度验证。没有安装或训练PPO/SAC/BC，Phase0报告、CSV、视频、旧源码和第三方模型保持原样。

[验收报告](docs/07_Phase1验收报告.md) · [控制器](docs/04_Phase1控制器设计.md) · [动力学/虚功验证](docs/05_Phase1动力学与Jacobian力映射验证.md) · [碰撞局限](docs/06_Phase1安全工作空间与碰撞审计.md)

[54秒真实动力学视频](assets/videos/phase1_dynamic_control.mp4)

![平台跟踪](assets/plots/phase1_pose_tracking.png)

## 直接运行（PowerShell）

```powershell
conda activate stewart_rl
Set-Location __PROJECT_ROOT__
python scripts/demo_phase1_control.py --controller C2 --trajectory z_roll --viewer --seconds 30
python scripts/demo_phase1_control.py --controller C1 --trajectory periodic --seconds 6
python scripts/compare_phase1_controllers.py
python -m pytest -q
```

解释器可直接替换为 `& 'D:\\miniconda\\envs\\stewart_rl\\python.exe'`。轨迹支持home/z/roll/pitch/z_roll/roll_pitch/periodic，控制器C0/C1/C2。默认真实1ms物理步、1kHz控制更新。`--timestep 0.0005`仅细化物理步，控制仍1kHz。超过6秒的Viewer预算用于最终目标静态保持，不是新增轨迹。每次写入独立results/runs目录。

实际已运行C2/z_roll原生Viewer（90秒明确预算，手动提前关闭），正常退出并归档；它显示同一个正在积分的live MjData。没有逐物理步viewer.lock竞争。窗口可能慢于墙钟实时：本机Python完整日志/查询循环不构成硬实时1kHz保证。

`compare_phase1_controllers.py`默认只显示已保存结果；`--run-frozen-suite`是显式42次批量试验入口，检测到已归档suite会拒绝覆盖，不能误当成观看命令。其他新入口与可复现边界见验收报告。

## 结果范围与限制

名义21/21、0.5ms复核6/6、扰动7/15完成。8次失败没有排除。PID在当前名义短时任务上比C0/C2误差更大；三者均未解决所有增益失配条件。

原机器人接触仍禁用。新增14个质量为0的腿/平台/底座包络代理，在独立查询模型中检查86个配对；没有覆盖Cardan全部零件，也没有把代理接触力加进正式plant。**不能宣称整机碰撞安全，不能开放大范围RL随机探索。**

42次正式原始全物理步日志为每run的physics.csv.gz；根目录phase1_trajectory_tracking.csv是50Hz查看投影，不替代完整数据。独立复算使用完整原始日志。视频为9个既有案例的真实重复执行，状态与正式记录逐位一致，不作为新增独立科学样本。

本阶段结束后暂停，未开始Phase2/Gymnasium封装或任何学习训练。
'''
    root=root.replace('名义21/21、0.5ms复核6/6、扰动7/15完成。8次失败没有排除。',f'名义{n_nominal}/{len(nominal_main)}、0.5ms复核{n_fine}/{len(fine_rows)}、扰动{n_perturbed}/{len(perturbed)}完成。{failed}次失败没有排除。')
    (ROOT/'PHASE1_README.md').write_text(root,encoding='utf-8')
    doc4='''# Phase1控制器设计

## 模型、变量和冻结边界

物理模型来源与Phase0相同，原XML未改。1ms物理步，1kHz控制。原±400N控制/力限幅、slide[-0.1,0.35]m及weld solref/solimp均保留。平台p/R、世界线速度v/角速度omega来自upper_center（不是偏移后的质心），Jacobian与wrench也以该点为参考。

q_slide/dq_slide是6个移动副相对行程/速度；actuatorpos与相对行程一致，不能当作安装点弦长或约1m的支链总长度。u是名义力控制输入，F_actual为考虑plant增益及原生限幅后的实际执行器力。后6个外扰通道与这6个控制输入分开。

src/audit_control.py完整冻结，5000步原Z参考C0复现得到力指令、qpos差异均为0（逐位相同）。Phase1比较将三者置于同一新参考上：每20ms完整链式IK/静平衡节点，三次Hermite给出连续行程与速度，1ms逐点检查行程，离网格直接IK核对最大误差见phase1_preflight/*_interpolation.json（约1e-11m）。这不是声称每1ms都在线求解一次IK，也不是未经检查的过冲样条。

## C0：冻结平衡力+支链PD

F_raw = F_nominal_static + 5000(q_ref-q_slide) + 200(dq_ref-dq_slide)。P为N/m，D为N·s/m。保留Phase0增益、符号和接口；新轨迹的静力参考在名义模型预计算，没有读取扰动物理参数。没有加入新的加速度前馈。

## C1：支链PID

在相同P/D与F_nominal_static上，I_state += error×0.001s；Ki=3000N/(m·s)，积分状态上限±0.005m·s，对应最多±15N积分补偿。输出仍±400N。使用实测slide速度反馈，避免对位置差分直接微分。

条件抗饱和：如果候选输出已超力限且积分会继续推向饱和，则冻结该分量积分；反向消饱和可继续。保护保持时清零积分。无效数值/越行程目标被拒绝。抗饱和和原生限幅有独立测试，但正式试验没有实际触及积分或力上限，不能把“有测试”说成这些极端工况已广泛验证。

PID不是先验更优。其名义周期任务有积分相位/累积误差，Ki在正式矩阵前冻结，没有根据结果追加预算或调参来制造优势。

## C2：Jacobian任务空间控制

采用世界twist=[vx,vy,vz,wx,wy,wz]，世界wrench=[Fx,Fy,Fz,Tx,Ty,Tz]，参考点upper_center。

W_act = J_l^T F_actual；控制时W_cmd = J_l^T F_nominal_static + K_task e_pose + D_task(twist_ref-twist_actual)。e_pose旋转部分为Log(R_ref R_actual^T)，不是欧拉角相减。求F_raw = solve(J_l^T,W_cmd)，随后遵守原±400N限幅。超限即不能精确实现所需wrench，不能把裁剪说成精确广义力实现。

为匹配C0而非人为削弱基线，K_task=J_home^T(5000I)J_home、D_task=J_home^T(200I)J_home，保留完整交叉项。不同块单位分别对应N/m、N/rad、N·m/m、N·m/rad，不能把整个矩阵叫作统一N/m。

J_l从冻结名义几何与实际可观测关节构型计算，100Hz刷新；P/D力更新1kHz。三个方法都运行同一条件数监控。条件数按0.35m尺度归一化，阈值50；接近奇异或非法解则拒绝并保护。六输入六维局部非冗余映射，没有使用盾构零空间力分配公式。

## 共享安全与模型隔离

每物理步检查有限状态、原生警告、slide范围、weld平移/角残差（都维持1e-4m/1e-4rad）、受限位姿；50Hz独立proxy查询只覆盖已定义的配对，5mm为这些代理的保守触发门槛。实际XY偏移2mm、Z偏移5mm、Roll/Pitch0.5°、Yaw0.2°是受限试验保护界限，不是完整可认证工作空间。

失败后停止参考推进，转为以违规时实际slide构型为目标的受限PD保持0.2秒，积分清零，并保留继续运动。日志desired reference仍显示原任务参考，以计算未完成误差；实际保持目标另从事件qpos与control_time精确重建在phase1_applied_reference_audit.csv，不把两者混用。保持是响应式仿真措施，不保证能瞬时停住或回到安全集。

所有补偿读取冻结nominal模型；独立plant副本只接受质量/惯量、质心和增益扰动。Controller.command不接收plant模型。真正外载仅由运行器写入后6通道，不能由C0/C1/C2偷偷托举平台。
'''
    (ROOT/'docs/04_Phase1控制器设计.md').write_text(doc4,encoding='utf-8')
    doc5=f'''# Phase1动力学与Jacobian力映射验证

## 虚功与实际广义力

J_l为6×6，平移列是dl/dp，转动列是dl/dtheta；输入世界角速度，不是欧拉角速度。由P=F^T qdot_slide=(J_l^T F)^T twist得到映射，F以N计，wrench前三维N、后三维N·m。

在4个真实闭合构型（Home、Z/Roll、Roll/Pitch、周期轨迹）各3组seed42有界力/twist上进行12次核对。用完整链IK有限差分，再用MuJoCo mj_differentiatePos构造42×6约束切空间T；独立核对T^T qfrc_actuator与J_l^T F。

- 虚功最大误差：{force.virtual_work_abs_error_W.max():.9g}W。
- MuJoCo广义功率最大误差：{force.mujoco_power_abs_error_W.max():.9g}W。
- wrench最大分量差：{force.wrench_max_abs_error.max():.9g}（力/力矩各自单位）。
- 行程Jacobian最大差：{force.stroke_jacobian_max_error.max():.9g}。

映射是执行器贡献，不把重力、惯性、弹簧和约束反力都归给6个F。静态平衡通过完整名义模型求出；动态控制没有假装使用精确逆动力学加速度补偿。

## 轨迹与名义动力学

7种参考：Home、Z+4mm、Roll/Pitch各+0.4°、Z+2.5mm/Roll+0.25°、Roll/Pitch各+0.25°、带平滑包络的2秒周期Roll/Pitch（最大0.2°/0.15°）。位置初态与Phase0相同。各6秒，非周期动作1秒保持、2秒五次缩放、后续保持；周期动作在1～5秒，最后1秒归零保持。

这是六维状态/力控制，但当前主动激励只有Z/Roll/Pitch；X/Y/Yaw目标恒定，不声称已验证全六轴独立工作空间。全部组合均在网格节点与离节点做独立IK、行程和条件数检查，而非从单轴端点推断。

下表为7个名义任务、每任务完整物理步的pooled RMSE，未混入扰动或0.5ms重复复核；“平均每任务RMSE”是另一统计量，另存在CSV，不能混淆。

'''+markdown(pooled,['controller','completed','trials','pooled_position_rmse_mm','pooled_orientation_rmse_deg'])
    doc5+='\n## 0.5ms复核（控制周期保持1ms）\n\n'
    doc5+=markdown(fine,['controller','trajectory','position_rmse_1ms_mm','position_rmse_halfms_mm','orientation_rmse_1ms_deg','orientation_rmse_halfms_deg','matched_time_max_position_difference_mm','matched_time_max_slide_difference_mm'])
    doc5+='''
原配置solref=[0.0001,1]及solimp不变，refsafe开启。1ms物理步下有效时间常数至少2ms，0.5ms下至少1ms；约束有效刚度随此规则变化。因此不能把这个比较宣称为固定连续约束参数的纯积分收敛证明。plant.json逐次记录原始solref/solimp、refsafe与物理步长。

控制器始终1kHz；0.5ms时力保持两个物理步，PID积分仍按1ms更新，Jacobian刷新仍100Hz。误差以位置向量和SO(3)角度分别计算；平移/方向weld残差分别记录，mixed_equality_residual不统一解释为米。

所有指标从physics.csv.gz的XYZ和四元数独立复算，与运行器统计的差见phase1_metric_recomputation.csv。失败回合不会从表中删除；保护早停必须结合进度与终止原因解释RMSE。

## 外扰、计算耗时和复现限制

5种扰动：upper质量/惯量均×1.10（20→22kg）；upper质心局部X+3mm；前6力增益均×0.9；增益分别[0.9,1.05,0.95,1,0.92,1.02]；2～4秒施加upper_center局部[0,0,-10N,0.5N·m,0,0]。这不是执行器时间常数或迟滞模型，不声称研究了所有动态响应不一致。

外载site坐标系已用mj_jacSite和qfrc_actuator核对，局部→世界变换误差为0，见phase1_external_frame_validation.json。后6通道只在明确外载组启用。

controller_compute_s含参考求值与控制/条件监控，不含记录、渲染和独立碰撞查询；另存每步物理耗时和总墙钟。Python循环通常慢于物理时钟，尾部计算时间也会超过1ms，不宣称硬实时控制性能。本轮没有独立隔离CPU基准，耗时是实际执行现场的观测。
'''
    (ROOT/'docs/05_Phase1动力学与Jacobian力映射验证.md').write_text(doc5,encoding='utf-8')
    doc6=f'''# Phase1安全工作空间与碰撞审计

## 验收结论：部分查询可用，整机接触安全未完成

原plant机器人网格contype=0/conaffinity=0原样保留。因此原生ncon=0仍不能证明无碰撞。没有把所有STL接触打开，也没有通过新接触力改变三控制器比较。

新增models/phase1_proxy_query.xml是独立查询副本。12个腿capsule覆盖legbot/legup各自全部编译网格顶点，半径取局部XY径向最大值，轴段覆盖完整Z范围；底座/平台增加2个包络实心圆柱。全部质量0，编译后body_mass最大差{collision['native_mass_max_change']}。原模型与代理模型的差异、完整原45个geom及代理参数列在phase1_collision_audit.json。

## 配对与过滤

- 不同支链的lower/lower、lower/upper、upper/upper均参与，未关闭所有机器人自碰撞查询。
- 同一腿lower/upper为真实伸缩嵌套，排除；base/lower为U铰安装，upper平台/upper腿为上端安装，排除；固定base/地面排除。
- 其余腿/平台/地面关系保留，总{collision['proxy_pairs']}对。
- 排除是body级的，不能推广到大角度时整条腿与板之间的所有接触都合法；完整Cardan、安装凹槽、关节真实限位还没覆盖，包络实心化也可能过保守。

Home原生proxy接触数={collision['home']['native_contacts']}，未出现伪接触；最小proxy间隙{collision['home']['minimum_clearance_m']*1000:.6f}mm。7条参考共{collision['reference_samples']}个离散样本，最小间隙{collision['minimum_reference_clearance_m']*1000:.6f}mm。实际执行中50Hz按真实qpos在独立MjData查询，原live数据不会被覆盖，最小实际值另见比较表。

这些距离是**包络几何**表面距离，不是真实CAD全部零件的精确距离。distmax=0.2m，返回0.2时只能解释为距离下界≥0.2。原生几何负距离/接触语义另用已知尺寸capsule/sphere真实模型测试，包含穿透正例；不是硬编码机器人必须通过。

## 受限工作区、保护失败和后续要求

全部参考幅度小于Phase0已测试单轴范围，组合另做真实链IK/条件数检查。实际位姿保护仅是局部试验界限；不一致增益在Home保持初期就触及0.5°姿态界限。此时即使proxy间隙仍正、weld稳定，也不能称任务安全成功。受限保持会有后续运动，见phase1_failure_hold.png及实际日志。

当前5mm代理保护只作用于86个已定义配对，并且是50Hz离散检查，不具备连续时间碰撞安全证明。缺失Cardan覆盖、原生物理接触未在正式plant中启用、模型关节实际机械限位未标定，均属未完成项。不能据此扩大RL探索范围或部署实机。

后续最小工作：补Cardan/安装点代理并用真实几何碰撞反例验证，审计大幅姿态下的过滤合法性，单独验证接触增强plant对控制和约束稳定性的影响，再扩大工作空间。未经审核不自动开展。
'''
    (ROOT/'docs/06_Phase1安全工作空间与碰撞审计.md').write_text(doc6,encoding='utf-8')
    report=f'''# Phase1验收报告

客户端日期：2026-10-27。工程__PROJECT_ROOT__；解释器D:/miniconda/envs/stewart_rl/python.exe，Python3.11.17、MuJoCo3.17.0。没有安装/训练RL框架，没有开始Phase2。

## 真实执行范围

冻结协议configs/phase1_protocol.json，42次正式试验：21次名义、15次单因素扰动、6次0.5ms复核。实际物理积分{manifest['actual_physics_steps']}步（含受限保持）；失败提前停止导致少于288000步主预算，不补跑种子/不排除失败。名义与扰动是固定确定性工况，不把0.5ms复核或录像重复当作独立随机场景。seed42用于虚功随机测试；动力学没有伪造随机种子重复统计。

原控制器完整冻结并做5000步双模型逐位等价复现。新代码、模型参数、原始压缩CSV、NPZ、事件与运行源快照分别归档在suite和preflight目录。正式执行的早期源快照位于executed_sources；随后仅增加可选GUI入口/预算参数、可视化相机和报告，不改变已冻结控制律、plant或正式结果。

## 完成情况

- 名义：21/21；0.5ms复核：6/6；扰动：7/15。总计34/42，但不是统计成功率置信度。
- 原Phase0的{preserve['files_checked']}个文件逐项哈希核对，变化数0；涵盖原报告、CSV/NPZ、视频、源码与第三方文件（排除.git和缓存）。未修改Reacher/Panda/盾构工程。
- 自动测试：{tests.get('tests')} passed、0 failed、0 skipped，其中原25项全部重新执行，新增32项。初次预检有UTF-8被GBK默认解码失败及两个等待suite完成的skip；修复文件编码读取后最终验收全部通过，初次记录保留。
- 预检中mj_differentiatePos不接受非连续输出切片，已用连续临时数组修正，失败日志保留；未改指标门槛。
- 正式全部weld平移<{table.max_weld_translation_m.max():.9g}m、角度最大{table.max_weld_angle_rad.max():.9g}rad，未放宽1e-4门槛；原生数值警告数0。
- 力峰值{table.max_force_N.max():.6f}N，低于±400N；力限幅与积分限幅介入均0。失败不能归咎于饱和。

## 三控制器名义完整轨迹RMSE

下表是7个名义任务等时长全样本pooled RMSE：

'''+markdown(pooled,['controller','completed','trials','pooled_position_rmse_mm','pooled_orientation_rmse_deg'])
    disturbed=table[(table.condition!='nominal')&(table.timestep_s==.001)].copy()
    report+='\n## 扰动逐组结果（含全部失败）\n\n'+markdown(disturbed,['condition','controller','success','position_rmse_mm','orientation_rmse_deg','progress','failure_reason'])
    report+='''
质量+10%时C1完成，C0/C2在完整6秒后未满足0.75mm终点门槛。质心偏移与小幅外载三者均完成。增益均降10%时三者都未完成，其中PID触及实际位置界限而提前保护。增益不一致时三者在约0.126s触及姿态界限，保持0.2s后结束。表中progress只是已积分时间/6秒预算（含保护保持），约5.43%，不是几何路径完成率；这些失败发生在Home保持阶段，计划运动尚未开始。其RMSE不能与完成6秒任务直接排名。

所有失败、事件和后续保持轨迹见phase1_failure_analysis.json以及各run的events.json/physics.csv.gz。保持期间机器人仍可能继续移动；这是仿真响应式保护，不能宣称安全集不变。不能把原生ncon=0和满足已覆盖代理间隙混为整机无碰撞证明。

## 模型失配与公平性

全部controller只拿名义模型/预计算参考，实际plant质量、惯量、COM和gain独立记录在plant.json，补偿不读取这些真值。mass110仅平台20→22kg和主惯量同比增加；COM仅局部X+3mm；gain090为六通道×0.9；asymmetric_gain为[0.9,1.05,0.95,1,0.92,1.02]；external_load明确2～4秒site局部Fz=-10N、Tx=0.5N·m。没有通过更改关节范围、weld阈值或障碍物使之通过。

PID与C2都有局限，没有声称它们必然超过C0。C2匹配了C0在Home的任务空间刚度/阻尼，结果接近C0，不能夸大微小差异。PID在名义短任务上更差，但对增加质量有一定补偿作用；积分参数未经广泛调优，不能代表所有传统积分方法的极限。

## 数值、日志和视频验收

Z/Roll与周期轨迹对三方法各做0.5ms复核，控制周期保持1ms，均通过同一约束门槛。原solref/solimp配置不改，但refsafe下有效时间常数随物理步变为2ms/1ms，见05号报告与phase1_timestep_comparison.csv；不能误称固定约束模型参数的纯数值收敛。

根目录phase1_trajectory_tracking.csv是50Hz查看投影；每run physics.csv.gz保存全1000/2000Hz物理状态、期望/实际位置和姿态、行程、速度、力、积分、饱和、外扰、约束和计算耗时。phase1_metric_recomputation.csv从XYZ/四元数独立复算全部42次；保护时实际保持指令目标另由事件与控制时刻重建在phase1_applied_reference_audit.csv，不篡改原始记录。

原生Viewer：实际C2/z_roll/nominal、90秒有限预算，手动关闭后正常Python退出并保存demo_exit.json。显示的就是运行器的同一live MjData；没有额外step、qpos播放或逐物理步viewer.lock。用户关闭属于取消演示，summary按viewer_closed标记，不算科学试验失败或成功率样本。没有声称每个控制器/每个参数组合都逐项人工点击GUI验收。

视频phase1_dynamic_control.mp4共54秒、540帧、960×720。初版字幕遮住动平台，故保留initial_capture及其SHA，重新真实执行9个已冻结案例，只调整相机；全部qpos/qvel/ctrl/time与正式数据逐位相同。额外54000个展示物理步不进入42次科学结果。全部540帧与对应实际状态哈希匹配，完整解码通过；未从保存qpos重建渲染冒充现场。

## 碰撞验收边界

新增质量0的12腿capsule与2板cylinder，独立模型86对，Home无伪接触，217个参考构型及实际执行离散查询间隙为正。原完整模型接触仍关闭，Cardan没有完整代理，body级安装排除仅在小范围有依据；因此碰撞部分结论是PARTIAL QUERY ONLY，整机碰撞安全与大范围工作空间未完成。详见06号报告，不把控制功能通过等同碰撞认证。

## 可复制PowerShell命令及实际范围

```powershell
conda activate stewart_rl
Set-Location __PROJECT_ROOT__
python scripts/demo_phase1_control.py --controller C2 --trajectory z_roll --viewer --seconds 30
python scripts/demo_phase1_control.py --controller C1 --trajectory periodic --seconds 6
python scripts/compare_phase1_controllers.py
python -m pytest -q
```

上述30秒Viewer用法与实际90秒预算的同一入口；实际通过了提前关窗。无窗口比较42次、预检、独立几何查询、视频解码与测试已真实执行。批量入口是compare_phase1_controllers.py --run-frozen-suite，默认只读，已有归档时拒绝覆盖。未重新运行Phase0长批量或任何RL训练。Python全日志/查询循环的墙钟可能慢于仿真时间，未做硬实时认证。

## 用户十个问题的回答

1. C0保留名义平衡力+支链PD；C1加入有界积分/条件抗饱和/速度反馈；C2以真实链Jacobian在任务空间计算wrench，再求六力并限幅。三者都真实运行。
2. 六力经各slide驱动、闭链几何映射形成upper_center世界wrench=J_l^T F_actual；虚功和MuJoCo广义力交叉验证通过，不使用冗余零空间公式。
3. 名义小范围组合和周期任务全部完成；增益失配场景不能稳定完成全部任务。只激励Z/Roll/Pitch，不泛化为任意6轴工作空间。
4. 位置/姿态RMSE见上面的pooled表和逐工况表，三维XYZ与SO(3)完整计算，失败不筛除。
5. 正式任务无力或积分饱和；单元测试验证了原生限幅和抗饱和分支，不能把该分支测试当作全部极端物理工况验证。
6. 1ms动态轨迹weld精度满足原1e-4平移/旋转门槛；有些任务仍因位姿精度或受限工作区失败。
7. PID改善质量增加工况，但名义误差更高、增益失配仍失败；C2与C0接近，未证明普遍优越。
8. 已覆盖代理的真实几何查询和负距离测试有效；原正式plant未启用整机物理接触，Cardan覆盖缺失，不能宣称完整碰撞检测合格。
9. 存在可重复剩余误差，但不能据此证明BC/PPO必要。优先诊断动态前馈、增益辨识/鲁棒控制和积分短时行为，再讨论学习方案；本轮没有让PPO比较对象人为变弱。
10. 控制/状态/参考接口可支持后续受限Gymnasium原型，但扰动鲁棒性和整机碰撞安全尚未通过，不具备立即开放大范围随机探索的工程条件。建议先审核本轮失败和补齐安全覆盖，再决定Phase2范围。

## 未完成事项与暂停

整机Cardan接触、原生碰撞增强plant的动态控制验收、全工作空间/真实机械关节限制、执行器动态延迟/迟滞、广泛PID整定、加速度前馈和实机均未完成。没有因收尾而自动增加这些算法。本阶段完成后暂停，不封装Gymnasium、不安装或启动RL框架。
'''
    report=report.replace('名义：21/21；0.5ms复核：6/6；扰动：7/15。总计34/42',f'名义：{n_nominal}/{len(nominal_main)}；0.5ms复核：{n_fine}/{len(fine_rows)}；扰动：{n_perturbed}/{len(perturbed)}。总计{completed}/{len(table)}')
    (ROOT/'docs/07_Phase1验收报告.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'results/phase1_acceptance.json',dict(client_date='2026-10-27',status='CONTROL_BASELINES_EXECUTED; COLLISION_QUALIFICATION_PARTIAL',formal_trials=len(table),completed=completed,failed=failed,nominal_completed=n_nominal,nominal_trials=len(nominal_main),disturbed_completed=n_perturbed,disturbed_trials=len(perturbed),fine_completed=n_fine,fine_trials=len(fine_rows),actual_formal_physics_steps=int(table.physical_steps.sum()),display_only_repeat_steps=delivery['display_only_physics_steps'],tests_total=int(tests.get('tests')),historical_tests=25,new_tests=int(tests.get('tests'))-25,phase0_changed=len(preserve['changed']),video_live_state_verified=video['state_matches']==video['frames'],full_collision_safe=False,new_rl_training_steps=0,next_phase='Paused awaiting review; no automatic Phase2'))
    print('Phase1 reports generated from real frozen artifacts.')
if __name__=='__main__':main()
