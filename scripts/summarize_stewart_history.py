"""Build a source-linked retrospective from the frozen Stewart archives."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def csv_rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (RESULTS / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)


def table(rows):
    clean = lambda row: "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |"
    return "\n".join([clean(rows[0]), clean(["---"] * len(rows[0])), *map(clean, rows[1:])])


def save(name, content):
    with (ROOT / "docs" / name).open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")


milestones = [
    ("0", "闭链与驱动可用性", "原MJCF、实际关节链IK、步长对照", "phase0_timestep_sensitivity.csv", "2ms严门槛失败，采用1ms基线"),
    ("1", "平台轨迹与力控制", "传统反馈、Jacobian力映射、代理几何查询", "phase1_controller_comparison.csv", "工作区与碰撞查询仅覆盖验证样本"),
    ("1.5", "实时交互", "动力学线程、Viewer及数据归档", "phase1_5_run_summary.csv", "运动和录制同步"),
    ("1.5.1", "交互可读性与持续运行", "彩色支链、相机、状态镜像和事件日志", "phase1_5_1_visual_validation.json", "区分实际积分与显示"),
    ("2A", "突加载荷", "传统控制对照和恢复窗口", "phase2a_disturbance_summary.csv", "以物理指标筛选抗扰方法"),
    ("2B-1", "改进经典抗扰", "反馈和因果扰动估计", "phase2b1_controller_comparison.csv", "观测器与控制联合评估"),
    ("2B-2", "残差权限与鲁棒性", "共享限幅、动作探针和失配试验", "phase2b2_action_sensitivity.csv", "有效残差依赖状态和预算"),
    ("2B-3", "增益失配与Yaw", "四维任务空间经典补偿", "phase2b3_controller_comparison.csv", "筛选可控目标"),
    ("2C-0", "未知执行器动态及突变负载", "滞后、噪声和观测延迟；C4基线", "phase2c0_actuator_lag_validation.csv", "失配能否形成学习任务需分别验证"),
    ("2C-1", "首次真实残差PPO", "Fz单维修正、Gymnasium与价值审计", "phase2c1_training_summary.csv", "训练成功不等于超过经典对照"),
    ("2C-2", "有限动态任务与真实录像", "小幅Z参考与PPO对照", "phase2c2_evaluation_comparison.csv", "小位移需用支链与位置曲线辨认"),
    ("2C-3A", "50mm几何与动力学可行性", "工作区扫描、IK、低速动力学验证", "phase2c3a_workspace_scan.csv", "仅采用准入轨迹"),
    ("2C-3B", "50mm速度和加速度跟踪", "FF-VA和多速度误差对照", "phase2c3b_controller_comparison.csv", "使用增强传统基线"),
    ("2C-4", "50mm残差PPO", "Z跟踪与外载学习", "phase2c4_controller_comparison.csv", "按事件分段评价"),
    ("2C-5", "Roll/Pitch动态与力矩抗扰", "二维力矩策略、姿态保护、Home强扰动边界", "phase2c5_torque_boundary.csv", "4.8 N·m触发保护"),
    ("2C-6", "姿态优先奖励及课程", "V2与中强外力矩", "phase2c6_training_summary.csv", "区分历史模型复用与新增训练"),
    ("2C-7", "综合Z与姿态运动", "三阶段PPO和冻结评价", "phase2c7_training_summary.csv", "强经典反馈仍是主要对照"),
    ("2C-7V", "回报和实际动作可解释性", "检查点闭环、1ms行为分析、真实训练画面", "phase2c7v_checkpoint_evaluation.csv", "发现恢复种子覆盖及回报与IAE分歧"),
    ("2C-8", "匹配奖励消融与Critic诊断", "独立任务RNG、V2/V3三种子", "phase2c8_training_summary.csv", "未晋级，转向可测模型失配"),
]
timeline = [("阶段", "问题", "技术", "证据", "结果规模", "边界或承接")]
for phase, question, method, evidence, lesson in milestones:
    count = len(csv_rows(evidence)) if evidence.endswith(".csv") else "JSON"
    timeline.append((phase, question, method, f"results/{evidence}", count, lesson))
write_csv("stewart_project_milestones.csv", timeline)

training = [("阶段", "方法", "种子", "预算口径", "决策", "物理步", "完整回合", "PPO更新", "模型证据", "来源")]
for stage, suffix in [(1,"summary"),(2,"comparison"),(4,"summary"),(5,"summary"),(6,"summary"),(7,"summary"),(8,"summary")]:
    name = f"phase2c{stage}_training_{suffix}.csv"
    for row in csv_rows(name):
        method = row.get("method") or row.get("group") or "PPO"
        scope = "历史复用" if row.get("reused") == "True" else row.get("phase") or "原表口径"
        training.append((f"2C-{stage}", method, row.get("seed"), scope,
            row.get("new_decisions", row.get("decisions")), row.get("physical_steps"),
            row.get("complete_episodes") or "源表未列", row.get("updates"),
            row.get("model_sha") or row.get("sha256") or row.get("model_sha256") or row.get("final_parameter_hash") or "见原档",
            f"results/{name}"))
write_csv("stewart_algorithm_comparison.csv", training)

limits = [
    ("传统控制局部姿态精度高", "CONFIRMED", "2C-8独立评价表", "改变Plant后重新测量"),
    ("实际受限残差", "CONFIRMED", "2C-8 action_audit与原命令重算", "统一权限下做单因素动作探针"),
    ("回报与窗口IAE不同步", "CONFIRMED", "2C-8 development_progress", "冻结相同奖励，报告加载与卸载"),
    ("Critic拟合偏差", "CONFIRMED", "2C-8 critic_summary与GAE复算", "尺度处理的控制影响仍待对照"),
    ("事件前后残差偏置", "CONFIRMED", "2C-8 action_events", "在配对条件下评估偏置的代价"),
    ("训练覆盖制约泛化", "SUPPORTED HYPOTHESIS", "任务有限，种子结果不一致", "隔离参数、时刻和方向"),
    ("标称任务欠缺明显适应需求", "SUPPORTED HYPOTHESIS", "强基线误差低，早期已有失配专项", "配对实测独立Plant变化"),
    ("单帧信息不足", "UNVERIFIED", "没有单帧/历史匹配控制消融", "分组留出轨迹的历史信息检查"),
    ("真实硬件能力", "UNVERIFIED", "证据来自MuJoCo和部分几何代理", "硬件部署与验证单列"),
]
limitations = [("问题","证据强度","证据","后续检验"), *limits]
write_csv("stewart_known_limitations.csv", limitations)

boundary = [("任务", "轴", "幅值 N·m", "步数", "SO3峰 度", "保护原因", "可恢复")]
for row in csv_rows("phase2c5_torque_boundary.csv"):
    boundary.append(tuple(row[k] for k in ("task","direction","amplitude_Nm","steps","so3_peak_deg","failure","recoverable")))
methods = csv_rows("phase2c8_method_metrics.csv")
latest = [("方法","加载IAE deg·s","卸载IAE deg·s","姿态峰值 deg","XYZ RMSE mm")]
latest += [(r["method"], *[f"{float(r[k]):.8f}" for k in ("load_iae_deg_s","unload_iae_deg_s","so3_peak_deg","xyz_rmse_mm")]) for r in methods]
model = json.loads((RESULTS / "phase0_model_inspection.json").read_text(encoding="utf8"))
save("78_Stewart项目全阶段研究总结.md", """# Stewart 项目全阶段研究总结

以原始结果目录为索引。表中规模是对应证据表的行数，不跨实验合并成功率。

""" + table(timeline) + """

## 模型与运动证据

平台任务空间为三平移三转动。原模型 nq=43、nv=42、nu=12；六支链含移动副和两端转动机构，六个 weld 实现闭链。仅前六个执行器驱动支链，剩余为显式外扰通道。真实关节链考虑铰轴非共点偏置，IK求滑块坐标，Jacobian经差分和力映射验证。几何代理未覆盖所有原网格与万向节。

Phase 0 实测小幅升降与倾转。3A–3B扩展至50mm，5验证主动Roll/Pitch，7与8联合执行升降及姿态。可达性看IK和支链行程，动态可控性看真实力驱动，安全只引用实际检查过的任务与保护边界。

## 抗扰强度边界

早期0.15和0.5 N·m要随各阶段的姿态及外载波形阅读，不与后期强扰动直接汇总。0.5–1.2 N·m动态工况查阅 phase2c7_joint_preflight.csv；Phase 8的最终联合任务最大模长1.2 N·m。以下为 phase2c5_torque_boundary.csv 的定点原始结果。2.4 N·m只支持对应Home任务，4.8 N·m保护失败保留。

""" + table(boundary))
save("79_Stewart控制架构与算法演进.md", """# Stewart 控制架构与算法演进

```mermaid
flowchart LR
    Ref[冻结轨迹与链式IK] --> C0[名义模型 C0 支链力]
    State[真实位姿速度与支链反馈] --> C0
    State --> Obs[C4因果扰动估计与四维补偿]
    Ref --> FF[FF-VA速度加速度前馈]
    State --> Actor[50 Hz二维残差Actor或经典反馈]
    Obs --> J[任务空间力到六支链映射]
    FF --> J
    Actor --> J
    J --> Limit[共享相对C0幅度与变化率]
    C0 --> Limit
    Limit --> Plant[真实MuJoCo 1 ms积分]
    Plant --> State
```

C0用名义重力与反馈生成六支链基准。C4-Observer利用状态响应和已知已施加命令估计外载，fourDlead以带动态补偿的任务空间项补充基线，FF-VA使用参考速度与加速度。50Hz经典姿态反馈以姿态旋转向量和角速度误差产生Mx/My，Phase 8使用25和1的PD系数。Residual PPO在相同二维通道输出每轴±0.05 N·m。传统与学习补偿共同服从每支链相对C0±5 N及20 N/s，实际执行器原生限幅±400 N。实际驱动力变化率与相对C0限速口径不同。

实现证据见 src/visualization/owner.py、src/workspace/dynamic_attitude_study.py、src/rl/stewart_dynamic_attitude_env.py 和 scripts/evaluate_phase2c8.py。

README中的“没有PPO”和未安装学习包是Phase 0快照。2C-8运行归档及checkpoint证实后续安装并训练；旧Phase 0环境断言因此在2C-8回归失败。旧2C-7阶段切换时VecEnv seed覆盖恢复状态，2C-8用独立任务RNG并完成阶段边界逐位复现。
""")
save("80_Stewart强化学习实验结果与失败归因.md", """# Stewart 强化学习实验结果与失败归因

逐运行预算、种子、模型证据见 results/stewart_algorithm_comparison.csv。原表口径保留，累计阶段与复用模型不重复相加。早期Z残差决策周期50ms，姿态PPO改为20ms二维Mx/My，综合动态观测为58维。V1原有姿态代价与多目标组合，V2强化姿态跟踪并重标定各项，V3在V2基础上对归一化SO(3)误差叠加平滑绝对值与平方项各0.5。训练使用折扣全程成本，评价按加载与卸载窗口积分，允许实际出现取舍。

Phase 8最终四任务各方法均值如下。PPO均为三个种子，传统每任务单次。同权限经典仍为主要基准；局部PPO结果有改善，未建立稳定优越性。

""" + table(latest) + """

Phase 8共六组8192决策，96次优化。V3开发门禁失败，32768与正式ReturnNorm均未执行。原始Critic证据见 phase2c8_critic_summary.csv，六组平均预测减目标为正，部分EV为负，V3 loss较低伴随奖励定义和尺度变化。GAE复算通过。不同种子分别学习不同补偿，且存在事件前后偏置；随机动作、任务覆盖与价值误差对性能差异的因果贡献尚未分解。

""" + table(limitations))
save("81_Stewart研究现状与下一阶段技术路线.md", """# Stewart 研究现状与下一阶段技术路线

闭链动力学 → 传统控制基线 → 受限残差PPO → 姿态抗扰 → 课程学习 → 综合动态控制 → 未知动力学适应。

当前可靠对照是C4、fourDlead、FF-VA与同权限50 Hz姿态反馈。已有最佳局部PPO结果要按单任务、种子和加载/卸载指标陈述。现有结果支持真实闭环控制、50mm联合运动和有限工况局部学习收益。跨任务与种子的总体抗扰优势尚未建立。

下一阶段先用独立Plant失配验证传统性能退化与可恢复性，然后检验因果历史是否提供单帧以外的信息。试验按轨迹分组，禁止同一轨迹相邻窗口穿越训练与验证划分。通过后才能比较经典适应、单帧与历史PPO。冻结策略部署期间以新观测改变动作，参数更新只发生在训练阶段。
""")
print("Part A saved. Strong baseline: C4/fourDlead/FF-VA + classical50. PPO benefits are local; matched V3 did not promote. Next: isolate plant mismatch and measure history information.")
