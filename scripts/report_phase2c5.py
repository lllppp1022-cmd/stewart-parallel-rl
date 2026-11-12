"""Chinese research reports backed by the phase's saved runs."""
import _bootstrap
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pandas as pd
import stable_baselines3
import torch

from src.model import ROOT, sha, write_json
from src.workspace.attitude_study import RUN


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(d):
    def cell(v):
        return "—" if pd.isna(v) else f"{v:.8g}" if isinstance(v,(float,np.floating)) else str(v).replace("|","/")
    return "\n".join(["| "+" | ".join(d.columns)+" |",
                      "| "+" | ".join(["---"]*len(d.columns))+" |"]+
                     ["| "+" | ".join(map(cell,row))+" |" for row in d.itertuples(index=False,name=None)])


def save(number, title, body):
    (ROOT/"docs"/f"{number}_Phase2C5{title}.md").write_text(f"# Phase 2C-5 {title}\n\n"+body.strip()+"\n",encoding="utf-8")


def main():
    res=ROOT/"results"
    g=pd.read_csv(res/"phase2c5_attitude_workspace.csv")
    tracking=pd.read_csv(res/"phase2c5_attitude_tracking.csv")
    preflight=pd.read_csv(res/"phase2c5_torque_preflight.csv")
    boundary=pd.read_csv(res/"phase2c5_torque_boundary.csv")
    training=pd.read_csv(res/"phase2c5_training_summary.csv")
    evaluation=pd.read_csv(res/"phase2c5_classical_vs_ppo.csv")
    delta=pd.read_csv(res/"phase2c5_delta_vs_classical.csv")
    gate=read(RUN/"training_gate.json")
    decision=read(res/"phase2c5_final_decision.json")
    history=read(RUN/"history_audit.json")
    safety=read(res/"phase2c5_safety_audit.json")
    torque=read(RUN/"torque_gate.json")
    protocol=read(ROOT/"configs/phase2c5_eval.yaml")
    smoke=read(RUN/"smoke.json")
    reward_components=pd.read_csv(res/"phase2c5_training_reward_components.csv")
    regularization_share=reward_components[reward_components.component.isin(["reward_action","reward_slew"])].groupby("seed").share.sum()
    old_line=(res/"phase2c5_historical_pytest.log").read_text(encoding="utf-8",errors="replace").splitlines()[-1]
    new_line=(res/"phase2c5_pytest.log").read_text(encoding="utf-8",errors="replace").splitlines()[-1]
    durable=history["entries"]-len(history["missing"])
    scope="几何检查涵盖 887 个独立凸包配对，103 个连接组件配对保留排除。30 个被动铰链缺少原生硬止挡。所有姿态均保留部分覆盖标记。"
    save(51,"姿态工作区与运动学审计",f"""
在 Home 与少量 +50 mm 高度，共 {len(g)} 个离线姿态点通过完整链式 IK、闭链残差、原生限位、Jacobian 和几何间隙查询。采样包含单轴正负角度、双轴同号和异号组合。逐点主动与被动关节坐标保存在 `results/phase2c5_attitude_workspace.csv`。

采样最小原生行程余量 {g.native_margin.min()*1000:.3f} mm，最小已查询间隙 {g.clearance_m.min()*1000:.3f} mm，最大归一化 Jacobian 条件数 {g.condition.max():.6f}。{scope}

历史执行器配置实际 Roll/Pitch 保护为 ±0.5°，Yaw 为 ±0.2°。新工作区配置将原保护与已验证运动分开列出。保留 XY 2 mm、Z −2 至 +52 mm、weld 1e−4 m/rad、Jacobian 条件数 50 和几何查询 5 mm 门槛。1°/2°样点与 +50 mm 姿态组合的结果属于离线几何证据。真实姿态运行范围见 52 号报告。

参考使用世界固定轴旋转向量乘 Home 旋转矩阵。每段五次时间缩放在端点的角速度和角加速度为零。沿路径 41 点计算完整 IK 与约束静力平衡，支链目标使用三次插值。角速度和角加速度由旋转向量的标量时间导数解析得到。控制器和评价使用 SO(3)，RPY 用于显示与原保护。
""")
    save(52,"主动RollPitch运动验证",f"""
R1、R2 分别执行 Home、正向 0.35°、Home、负向 0.35°、Home。R3 沿世界旋转向量 (±0.25°, ±0.25°, 0) 经过相同顺序。三条轨迹 Z 固定在 Home，均为六秒。C0、C4 与 fourDlead 共 {len(tracking)} 次真实动力学执行，完成 {int(tracking.success.sum())} 次。参考经完整 IK、支链 PD 和参考静力项生成六支链命令，再由 MuJoCo 积分。

{table(tracking[["task","method","roll_min_deg","roll_max_deg","pitch_min_deg","pitch_max_deg","so3_rmse_deg","xyz_peak_mm","force_peak_N","success"]])}

无扰姿态跟踪均方根误差与突加载荷分开筛选。C0 在跟踪上的均值略低。正式抗扰基线为 {torque["method"]}，依据四方向 0.5 N·m 匹配试验的加载与卸载窗口总 IAE。匹配分数为 `{torque["matched_level2_scores"]}`，越小越好。

非零姿态保持 D2 和姿态运动 D3 分别验证了小幅复合力矩。实拍录像取 C4 的三条主动运动，附加实时误差曲线与支链数据，保持实际渲染角度。

{table(preflight[preflight.task!="Home"][["task","method","mx_Nm","my_Nm","so3_peak_deg","success"]])}
""")
    save(53,"强突加力矩与可恢复范围",f"""
后六路外扰从动平台 upper_center 施加世界坐标 Mx/My。控制器在加载前、加载中和卸载后持续使用前六路执行器。由 1 ms 记录复算世界 wrench 与期望阶跃的最大偏差。4.8 N·m 四个方向触发原位置保护，保留全部轨迹、事件和保护保持记录。

Level 1 为 0.15 N·m，Level 2 为 0.5 N·m，Level 3 为已验证最大共同档 2.4 N·m。强载扫描采用 Home、5 ms 合成滞后、2.0 秒加载、3.5 秒卸载。四方向均成功的 2.4 N·m 与触发保护的 4.8 N·m 构成离散试验区间。中间点及其他时序的边界尚未测定。

{table(boundary[["direction","amplitude_Nm","so3_peak_deg","xyz_peak_mm","add_peak_N","add_rate_N_s","failure","recoverable"]])}

方向 0 至 3 依次为正 Roll、负 Roll、正 Pitch、负 Pitch。失败时原保护将参考切换到保持点，因此表中 XYZ 跟踪误差按当时参考计算，保护判定仍按相对 Home 的原工作区。

由完整 Jacobian 和整套 ±5 N 附加幅值预算计算的静态纯轴上界，Roll 约 {g.pure_axis_3_static_add_Nm.min():.3f} 至 {g.pure_axis_3_static_add_Nm.max():.3f} N·m，Pitch 约 {g.pure_axis_4_static_add_Nm.min():.3f} 至 {g.pure_axis_4_static_add_Nm.max():.3f} N·m。这个数值假设预算全部可用。真实试验中基线同时使用预算，且执行器驱动力变化受限。扫描达到每支链附加 20 N/s，位置保护先于姿态保护停止 4.8 N·m 试验。

恢复判据为 0.1° SO(3)，窗口末尾至少持续 0.5 秒低于阈值。若全程低于容差，恢复时间只反映采样边界，无法用来判断策略谁恢复得更快。保护截断标为右删失。训练采用预检的 Level 1/2 有限组合。2.4 N·m 在正式评价中单列为已预检的强载回归。
""")
    save(54,"姿态Gymnasium与PPO训练",f"""
二维动作通过 ±0.05 N·m/轴映射到世界 Mx/My，六力解满足 J 转置乘六力等于目标 wrench。真实命令使用原共享 governor 一次，反事实 governor 只用于同状态接受命令审计。C4 记忆的是实际最终下发命令。独立物理正负动作试验中，两轴平均姿态差为 `{gate["measured_signed_action_difference_deg"]}` °。

52 维因果单帧包括 SO(3)误差、角速度与参考、XYZ误差和速度、支链行程/速度、已知命令、因果 observer、基础附加命令、上一请求与接受残差、参考姿态。角误差用 0.001 rad、角速度 0.02 rad/s、位置误差 0.2 mm 归一化。动作和观测均有固定缩放。Actor 的数据链为当前反馈、当前参考及上述控制器状态。

物理与传统控制 1 kHz，PPO 50 Hz，每决策保持 20 个物理步。完整六秒为 300 决策。正常结束使用截断，安全保护使用终止并计入保持物理步。6000 个零动作积分状态与冻结基线逐位等价，RGB 开关同样通过。Reset 两次均复现头 20 步。Qt 的 2000 步实际档案独立对照通过。Gymnasium 接口检查、八个训练组合随机动作和独立最终组合安全预检通过。

训练启动前完成 RGB 开关等价，Qt 的实机窗口档案也在启动前录下。Qt 档案的单独逐位比对在冒烟开始后完成，未重复训练。原 fourDlead 保持既有内部 20 Hz lead 子项，底层依旧每毫秒更新命令。

奖励在 1 ms 积分 SO(3)、角速度、XYZ、接受力矩及其变化率。尺度和权重见 `stewart_attitude_env.py`。逐决策物理代价有上界，保护惩罚超过余下折扣代价上界。固定正负与随机预检、原始数据复算见 `phase2c5_reward_audit.csv`。训练组合在方向、符号、强度、时序和持续时间上变化，采用冻结的有限配对分布。

{table(training[["seed","decisions","physical_steps","updates","adam_steps","complete_episodes","failures","wall_s","model_sha"]])}

42 号前 512 决策冒烟含 {smoke["physical_steps"]} 个物理步，保存重载后动作与参数一致。该前缀计入 8192。三个种子按最后一次更新选模型。PPO 配置保存于 `configs/phase2c5_ppo.yaml`，50 Hz 的 gamma 和 lambda 分别为 0.99^0.4 与 0.95^0.4。实际 PyTorch {torch.__version__}，SB3 {stable_baselines3.__version__}，MuJoCo {mujoco.__version__}，Python {sys.version.split()[0]}。

每次更新保留参数变化、Adam 步数、loss、KL、rollout 和模型 SHA。全量采样和物理记录独立核对 {safety["training_audit_episodes"]} 条训练轨迹或前缀。TensorBoard 指向 `results/runs/{RUN.name}/training`，训练视频采用 42 号预先指定的早、中、晚期实际随机采样。

训练期间实际各代价分量及占比。回报随已采样载荷任务和动作幅值共同变化，因此独立评价单列姿态指标。

{table(reward_components)}

动作幅值与动作变化率合计占各训练种子已积分代价的 {100*regularization_share.min():.2f}% 至 {100*regularization_share.max():.2f}%。此占比描述实际随机采样的优化信号，不能用作冻结评价中姿态性能的替代。
""")
    average=delta[~delta.task.str.contains("regression")].groupby("method")[[
        "gain_load_iae_deg_s","gain_unload_iae_deg_s","gain_so3_peak_deg","gain_omega_peak_deg_s","gain_xyz_peak_mm"]].mean()*100
    save(55,"传统姿态控制与PPO独立评价",f"""
{len(evaluation)} 次匹配闭环评价。基础控制为 {torque["method"]}。同权限经典对照在 50 Hz 施加 SO(3)和角速度 PD 力矩，Kp=25 N·m/rad，Kd=1 N·m·s/rad，±0.05 N·m/轴。未训练随机 Actor 使用同一初始模型的随机采样。三个末次 PPO 模型使用确定性动作。

冻结任务为 `{protocol["tasks"]}`。前三项为未参与学习的方向、幅值和时序组合，传统安全预检在训练前完成。强载回归与独立组合分开解释。

{table(evaluation[["task_id","method","success","so3_peak_deg","omega_peak_deg_s","load_iae_deg_s","unload_iae_deg_s","xyz_peak_mm","yaw_peak_deg","force_peak_N","force_rate_N_s","force_energy_N2_s"]])}

三个独立组合上相对同权限经典反馈的平均改善百分比。正值为该项数值降低。

{table(average.reset_index())}

绝对差、每工况百分比、每轴姿态、恢复和删失、预算利用率见 `phase2c5_delta_vs_classical.csv` 与完整评价 CSV。主判断要求至少两个种子在每个独立组合的加载和卸载 IAE 都优于同权限经典反馈，且无保护及超过 10% 的 XYZ 或力变化率增量。各种子结果 `{decision["seed_gates"]}`。

**{decision["decision"]}**。评价结论限定于该扰动分布和有限训练预算。比较图和实际控制时序分别保留，外载真值仅作为离线评价和视频标注。
""")
    save(56,"验收报告",f"""
2026-11-06。独立结果目录 `{RUN}`。

**{decision["decision"]}**。本轮完成主动 Roll/Pitch 运动、强载范围扫描、真实二维 PPO 及冻结对照。三个种子共 {training.decisions.sum()} 决策、{training.physical_steps.sum()} 物理步、{training.updates.sum()} 次 Actor/Critic 更新。完整评价成功 {int(evaluation.success.sum())}/{len(evaluation)}，训练保护终止 {training.failures.sum()} 次。强载扫描的四次位置保护单独保留。

| 验收问题 | 实际结果 |
|---|---|
| 1 可控 Roll/Pitch | Home 高度下验证单轴 ±0.35° 参考和双轴 ±0.25° 旋转向量参考，实际峰值见 52 号 |
| 2 主动倾斜与回正 | 九条有界轨迹完成 |
| 3 支链力驱动 | 参考 IK、支链力命令与 1 ms 积分构成执行链 |
| 4 几何覆盖 | {scope} |
| 5 世界力矩方向与幅值 | 正负 Mx/My 及复合力矩在外扰通道记录中核对 |
| 6 最大已测试恢复力矩 | 2.4 N·m，四方向及本轮时序 |
| 7 三档 | 0.15、0.5、2.4 N·m |
| 8 失败范围 | 4.8 N·m 四方向触发位置保护，保留右删失与原保持过程 |
| 9 二维动作权限 | 实际正负请求产生可测方向差，两轴约 0.0023° |
| 10 零动作 | 完整 6000 步、两次 Reset 及观察记录对照通过 |
| 11 Actor 因果边界 | 当前反馈与控制器历史摘要，外载真值、隐藏驱动和 tau 留在实验器 |
| 12 决策频率 | 50 Hz，常规每决策 20 物理步 |
| 13 网络更新 | 三种子实际更新和模型 SHA 可追溯 |
| 14 姿态峰值 | 按下表及逐任务峰值差判断 |
| 15 最大角速度 | 按下表及逐任务角速度差判断 |
| 16 恢复时间 | 原 0.1° 容差下多条轨迹未离开容差，时间分辨力有限 |
| 17 未见方向 | E_mix 复合方向单列全部种子 |
| 18 未见幅值 | E_roll、E_pitch 与 E_mix 单列 |
| 19 XYZ | 下表与逐任务峰值用于检查代价 |
| 20 执行器 | 力峰值、变化率、平方积分和附加预算分别核算 |
| 21 强经典对照 | {decision["seed_gates"]} |
| 22 种子一致性 | 保留三个种子的完整分项结果 |
| 23 当前瓶颈 | 强基线的剩余姿态误差较小，负载和卸载瞬态、残差限速及位置耦合需同时优化 |
| 24 历史与尺度 | 首轮固定单帧及 ±0.05 N·m，历史消融、尺度扩大和延长训练均待另行验证 |
| 25 综合任务 | 晋级判据为 {decision["promote_50mm_combination"]}，Phase 2C-5C 状态为未执行 |

{table(average.reset_index())}

历史全套原样运行，结果为 `{old_line}`。三个失败指向同一个被旧清单收录、现已消失的 Word 临时锁文件。独立 SHA 审计确认 {durable} 个持久文件匹配。历史测试断言及原 xfail 保留。Phase 2C-5 新增测试结果为 `{new_line}`。

三个视频均由现场 RGB 与匹配积分状态制作，逐帧姿态状态哈希和物理时间戳通过校验。命令在 PowerShell 下运行。

```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\\miniconda\\envs\\stewart_rl\\python.exe'
& $py scripts/view_phase2c5_policy.py --task R3
& $py scripts/view_phase2c5_policy.py --task E_mix --model models/phase2c5/seed42/final.zip
& $py -m tensorboard.main --logdir 'results/runs/{RUN.name}/training' --host 127.0.0.1 --port 6035
```

窗口运行一次六秒任务，完成后保留状态和曲线。左键拖动旋转相机，右键拖动平移，滚轮缩放。暂停时物理时钟停止。Reset 新建积分对象和独立日志。实际验证运行见 `viewer` 目录及 `qt_equivalence.json`。

本轮研究停在 2C-5B。保护覆盖局限、极限边界、历史帧消融与综合运动状态均见上述报告。
""")
    files=list(RUN.rglob("*"))+list((ROOT/"models/phase2c5").rglob("*"))
    for folder,pattern in [("scripts","*phase2c5*.py"),("configs","phase2c5*"),("tests","*phase2c5*.py"),
                           ("results","phase2c5*.*"),("assets/plots","phase2c5*.png"),
                           ("assets/videos","stewart_phase2c5*.mp4"),("docs","5[1-6]_Phase2C5*.md")]:
        files.extend((ROOT/folder).glob(pattern))
    files.extend([ROOT/"src/workspace/attitude_study.py",ROOT/"src/rl/stewart_attitude_env.py"])
    manifest={}
    exclusions={"phase2c5_artifact_manifest.json","phase2c5_manifest_validation.json","phase2c5_tensorboard.log"}
    for f in sorted(set(files)):
        if f.is_file() and f.name not in exclusions and "__pycache__" not in f.parts:
            manifest[str(f.relative_to(ROOT))]=dict(sha256=sha(f),bytes=f.stat().st_size)
    write_json(res/"phase2c5_artifact_manifest.json",dict(run=str(RUN),files=manifest,decision=decision))
    print("Reports 51–56 and",len(manifest),"manifest entries")


if __name__=="__main__":
    main()
