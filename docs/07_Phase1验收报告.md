# Phase1验收报告

客户端日期：2026-10-27。工程__PROJECT_ROOT__；解释器D:/miniconda/envs/stewart_rl/python.exe，Python3.11.17、MuJoCo3.17.0。没有安装/训练RL框架，没有开始Phase2。

## 真实执行范围

冻结协议configs/phase1_protocol.json，42次正式试验：21次名义、15次单因素扰动、6次0.5ms复核。实际物理积分269495步（含受限保持）；失败提前停止导致少于288000步主预算，不补跑种子/不排除失败。名义与扰动是固定确定性工况，不把0.5ms复核或录像重复当作独立随机场景。seed42用于虚功随机测试；动力学没有伪造随机种子重复统计。

原控制器完整冻结并做5000步双模型逐位等价复现。新代码、模型参数、原始压缩CSV、NPZ、事件与运行源快照分别归档在suite和preflight目录。正式执行的早期源快照位于executed_sources；随后仅增加可选GUI入口/预算参数、可视化相机和报告，不改变已冻结控制律、plant或正式结果。

## 完成情况

- 名义：21/21；0.5ms复核：6/6；扰动：7/15。总计34/42，但不是统计成功率置信度。
- 原Phase0的167个文件逐项哈希核对，变化数0；涵盖原报告、CSV/NPZ、视频、源码与第三方文件（排除.git和缓存）。未修改Reacher/Panda/盾构工程。
- 自动测试：57 passed、0 failed、0 skipped，其中原25项全部重新执行，新增32项。初次预检有UTF-8被GBK默认解码失败及两个等待suite完成的skip；修复文件编码读取后最终验收全部通过，初次记录保留。
- 预检中mj_differentiatePos不接受非连续输出切片，已用连续临时数组修正，失败日志保留；未改指标门槛。
- 正式全部weld平移<1.86000606e-07m、角度最大5.71648187e-05rad，未放宽1e-4门槛；原生数值警告数0。
- 力峰值94.054931N，低于±400N；力限幅与积分限幅介入均0。失败不能归咎于饱和。

## 三控制器名义完整轨迹RMSE

下表是7个名义任务等时长全样本pooled RMSE：

| controller | completed | trials | pooled_position_rmse_mm | pooled_orientation_rmse_deg |
|---|---|---|---|---|
| C0 | 7 | 7 | 0.070064967 | 0.0035956991 |
| C1 | 7 | 7 | 0.17667238 | 0.0043828386 |
| C2 | 7 | 7 | 0.070056885 | 0.0035953074 |

## 扰动逐组结果（含全部失败）

| condition | controller | success | position_rmse_mm | orientation_rmse_deg | progress | failure_reason |
|---|---|---|---|---|---|---|
| mass110 | C0 | False | 0.77951442 | 0.0049085646 | 1 | completion_accuracy |
| mass110 | C1 | True | 0.46615789 | 0.006344111 | 1 | nan |
| mass110 | C2 | False | 0.76348621 | 0.004159344 | 1 | completion_accuracy |
| com_x3mm | C0 | True | 0.38569321 | 0.025296614 | 1 | nan |
| com_x3mm | C1 | True | 0.33320885 | 0.0098618539 | 1 | nan |
| com_x3mm | C2 | True | 0.38402291 | 0.02527545 | 1 | nan |
| gain090 | C0 | False | 2.2821328 | 0.016518962 | 1 | completion_accuracy |
| gain090 | C1 | False | 1.4381553 | 0.017308884 | 0.75283333 | restricted_position_envelope |
| gain090 | C2 | False | 2.1751179 | 0.012298203 | 1 | completion_accuracy |
| asymmetric_gain | C0 | False | 1.1270581 | 0.50889985 | 0.054333333 | restricted_orientation_envelope |
| asymmetric_gain | C1 | False | 1.1233 | 0.50820833 | 0.054333333 | restricted_orientation_envelope |
| asymmetric_gain | C2 | False | 1.1264707 | 0.50864582 | 0.054333333 | restricted_orientation_envelope |
| external_load | C0 | True | 0.48220138 | 0.015296944 | 1 | nan |
| external_load | C1 | True | 0.44080078 | 0.011536418 | 1 | nan |
| external_load | C2 | True | 0.45309235 | 0.014758361 | 1 | nan |

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
