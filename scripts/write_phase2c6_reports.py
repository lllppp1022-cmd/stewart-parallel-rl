"""Write Phase 2C-6 reports from measured, archived outcomes."""
from pathlib import Path

import pandas as pd

from collect_phase2c6 import completed_seeds
from phase2c6_protocol import ROOT, RUN, config, read

DOCS=ROOT/"docs"
RESULTS=ROOT/"results"
ARCHIVE="../results/runs/phase2c6_20261107_v1"


def report(name, lines):
    (DOCS/name).write_text("\n".join(lines).strip()+"\n",encoding="utf-8")


def table(headers, rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+[
        "| "+" | ".join(map(str,r))+" |" for r in rows]


def main():
    decision=read(RESULTS/"phase2c6_final_decision.json")
    dev=read(RUN/"development_gate.json")
    admission=read(RUN/"curriculum_admission.json")
    d=pd.read_csv(RESULTS/"phase2c6_classical_vs_ppo.csv")
    train=pd.read_csv(RESULTS/"phase2c6_training_summary.csv")
    preflight=pd.read_csv(RESULTS/"phase2c6_curriculum_preflight.csv")
    q=read(RUN/"qt_equivalence.json")
    critic=pd.read_csv(RESULTS/"phase2c6_critic_reward_scale.csv")
    peak=critic.loc[critic.value_loss.idxmax()]
    med=critic.groupby("group").target_std.median()
    clips={p.stem.removeprefix("video_"):read(p) for p in RUN.glob("video_*.json")}
    hist=read(RUN/"history_after.json")
    test=read(RESULTS/"phase2c6_test_summary.json")
    independent=d[d.task_id.str.startswith("F_")]
    strong=d[d.task_id=="E_strong_regression"]
    gate_passed=dev["passing_seeds"]
    available=completed_seeds()
    courses=completed_seeds(True)
    course_rows=train[train.group=="curriculum_v2"]
    new_train=train[~train.reused]
    rec={x:int((d.filter(regex=x).stack()=="not_triggered").sum()) for x in ("_band_01_status","_band_001_status")}
    cens={x:int((d.filter(regex=x).stack()=="right_censored").sum()) for x in rec}
    course_table=[]
    for path in sorted((RUN/"curriculum_training").glob("*/curriculum_progression.json")):
        for stage in read(path)["stages"]:
            course_table.append([path.parent.name,stage["level"],stage["actual_decisions"],stage["complete"],"通过" if stage["passed"] else "停止"])
    preflight_table=[]
    for level,g in preflight.groupby((preflight.mx_Nm**2+preflight.my_Nm**2).pow(.5).round(2)):
        preflight_table.append([f"{level:.1f}",f"{int(g.success.sum())}/{len(g)}",f"{g.so3_peak_deg.max():.6f}",
            f"{g.xyz_peak_mm.max():.6f}",f"{g.force_peak_N.max():.3f}",f"{g.condition_peak.max():.3f}",
            f"{g.weld_translation_m.max():.3g}",f"{g.weld_angle_rad.max():.3g}"])
    report("59_Phase2C6中强扰动课程学习.md",[
        "# Phase 2C-6C 中强扰动联合预检与课程学习", "",
        "每组在相同 Home 初始状态下运行 fourDlead 零残差与一条种子 6401 的独立均匀随机动作序列。使用原 ±0.05 N·m 每轴请求、1 ms 物理积分、50 Hz 决策、±5 N 共享附加力预算及 20 N/s 硬限速。保留加载与卸载的精确 1 ms 输入波形。完整物理日志与失败原因按候选留档。", "",
        *table(["外载范数 N·m","成功/实际执行","SO(3) 峰值 °","XYZ 峰值 mm","最大力 N","Jacobian 条件数","weld 平移 m","weld 角 rad"],preflight_table),"",
        f"合计 {admission['tested']} 个时序/方向组合、{len(preflight)} 个真实回合，接纳 {len(admission['admitted'])} 个组合。包含 1.2 N·m 的 Roll/Pitch 正负方向和两种复合方向。安全结论限于本轮明确记录的工况及动作序列。[预检表](../results/phase2c6_curriculum_preflight.csv) · [接纳记录]({ARCHIVE}/curriculum_admission.json)","",
        f"固定 V2 开发晋级种子为 {gate_passed}，课程门槛要求至少两个种子。", "",
        *(["课程未启动。0.8、1.0、1.2 N·m 的随机预检均属于安全实验，采用独立均匀动作序列。固定 V2 的真实随机训练视频单独保存。",""]
          if not course_table else ["在固定 V2 的 8192 决策后继续优化，额外最多 8192 决策。0.8/1.0/1.2 N·m 分别预算 2048/2048/4096，已验证基础任务继续混合采样。阶段检查保护、完成情况、姿态峰值及是否实际完成新强度任务。", "",
                                    *table(["种子","阶段 N·m","真实新增决策","完整回合","门禁"],course_table),""]),
        "2.4 N·m 用于强扰动回归。4.8 N·m 继续沿用历史位置保护触发的边界证据。几何检测覆盖部分代理对，预检支持这些仿真条件下的有限研究。"])
    table_rows=[]
    for method,g in independent.groupby("method",sort=False):
        table_rows.append([method,f"{int(g.success.sum())}/{len(g)}"]+[
            f"{g[k].mean():.6f}" for k in ("load_iae_deg_s","unload_iae_deg_s","so3_peak_deg",
              "omega_peak_deg_s","unload_reverse_overshoot_deg","xyz_peak_mm","force_rate_N_s","accepted_rms_Nm")])
    dev_rows=[]
    for seed,g in dev["checks"].items():
        dev_rows.append([seed,*[f"{100*(g['mean_iae_ratio'][k]-1):+.2f}%" for k in ("load","unload")],
                         g["improvement_counts"]["load"],g["improvement_counts"]["unload"],
                         f"{g['xyz_ratio']:.4f}",f"{g['force_rate_ratio']:.4f}","通过" if g["passed"] else "未通过"])
    strong_rows=[[r.method,"完成" if r.success else str(r.failure),f"{r.load_iae_deg_s:.6f}",
                   f"{r.unload_iae_deg_s:.6f}",f"{r.so3_peak_deg:.6f}",f"{r.xyz_peak_mm:.6f}"] for r in strong.itertuples()]
    deltas=[]
    for seed in available:
        names=[(f"V2_{seed}",f"V1_{seed}")]
        if seed in courses:
            names.append((f"C2_{seed}",f"V2_{seed}"))
        for a,b in names:
            lhs=independent[independent.method==a]
            rhs=independent[independent.method==b]
            keys=("load_iae_deg_s","unload_iae_deg_s","so3_peak_deg","omega_peak_deg_s",
                  "unload_reverse_overshoot_deg","xyz_peak_mm","force_rate_N_s")
            deltas.append([a,b,*[f"{100*(lhs[k].mean()/rhs[k].mean()-1):+.2f}%" if rhs[k].mean()!=0 else "见绝对值" for k in keys]])
    costs=["group","seed","decisions","physical_steps","updates","adam_steps","complete_episodes","failures","wall_s"]
    report("60_Phase2C6_PPO再训练与独立评价.md",[
        "# Phase 2C-6D PPO 再训练与独立评价", "",
        "开发条件 E_roll、E_pitch、E_mix 已在历史研究中开启。各组按训练种子配对；负的百分比表示 IAE 下降。", "",
        *table(["种子","加载 IAE 相对 V1","卸载 IAE 相对 V1","加载改善≥5%任务数","卸载改善≥5%任务数","XYZ 比值","最大力变化率比值","晋级"],dev_rows),"",
        "F_roll、F_pitch、F_mix 在训练前冻结并通过传统控制基线预检。所有策略最终 checkpoint 在正式独立闭环前冻结。最终试验按外载生效的物理输入步划定各 1000 样本 IAE。各控制器共用同一初始状态、模型、外载波形和保护。评价使用确定性策略并验证权重不变。下表为三个独立任务等权平均。", "",
        *table(["方法","成功","加载 IAE °·s","卸载 IAE °·s","姿态峰值 °","角速度峰值 °/s","反向超调 °","XYZ 峰值 mm","实测力变化率 N/s","接受 RMS N·m"],table_rows),"",
        "同权限反馈沿用原世界系 SO(3) 误差的负反馈符号，Kp=25、Kd=1，和 PPO 共用投影及执行器约束。反向超调取卸载后 Roll/Pitch 对加载方向投影的反向峰值。每任务的 Roll、Pitch、Yaw、关节力平方积分、预算占用、原生接触和保护结果见[独立闭环数据](../results/phase2c6_classical_vs_ppo.csv)。","",
        "V2 与同种子 V1 预算相同。课程相对固定 V2 增加了样本预算，差值反映课程与继续优化的合并效应。", "",
        *table(["方法","参照","加载 IAE","卸载 IAE","SO(3) 峰值","角速度峰值","反向超调","XYZ","实测力变化率"],deltas),"",
        "## 2.4 N·m 专项回归", "",
        *table(["方法","结果","加载 IAE °·s","卸载 IAE °·s","SO(3) 峰值 °","XYZ 峰值 mm"],strong_rows),"",
        f"0.1° 标准共有 {rec['_band_01_status']} 个加载/卸载窗口未触发和 {cens['_band_01_status']} 个右删失。0.01° 研究带相应为 {rec['_band_001_status']} 和 {cens['_band_001_status']}。若全窗未离开带则标记未触发。已离带要求结束前持续稳定至少 0.5 s，截断或不足时标记右删失。", "",
        "## 样本及优化预算", "",
        *table(costs,train[costs].round({"wall_s":2}).itertuples(index=False,name=None)),"",
        f"本轮新增 {int(new_train.decisions.sum())} 决策、{int(new_train.physical_steps.sum())} 物理步、{int(new_train.updates.sum())} 次 PPO 更新。若有课程组，其额外步数与固定 V2 的 8192 前缀分开列示。历史 V1 完整训练只读复用。", "",
        f"固定 V2 每批价值目标标准差的中位数为 {med['training']:.3f}，课程阶段为 {med.get('curriculum_training',float('nan')):.3f}。峰值 Value Loss 为 {peak.value_loss:.2f}（{peak.seed}，累计第 {int(peak.decisions)} 决策），该批更新前的价值均方误差为 {peak.pre_update_value_mse:.2f}，目标标准差为 {peak.target_std:.2f}，KL 为 {peak.approx_kl:.3g}。强弱工况混合扩大了回报尺度；目标与预测的偏差值得后续单独分析。物理和优化日志均有限。[价值目标逐批审计](../results/phase2c6_critic_reward_scale.csv)","",
        f"[模型选择及 SHA-256]({ARCHIVE}/final_selection.json) · [完整训练记录]({ARCHIVE}/training) · [TensorBoard 标量]({ARCHIVE}/tensorboard_scalar_tags.json)"])
    videos=[f"- [{key}](../assets/videos/{Path(x['path']).name})，{x['decoded_frames']} 帧、{x['seconds']:.1f} s，逐帧对齐 1 ms 归档状态哈希与时间。"
            for key,x in sorted(clips.items())]
    evidence=[
        f"独立策略对照的双 IAE 胜场以同权限经典反馈为参照。达到两个独立种子、每种子至少两个任务双 IAE 改善，同时满足安全/控制代价条件才可进入下一阶段。",
        f"V2 可用种子 {available}，课程完整种子 {courses}，课程执行过的种子 {course_rows.seed.tolist()}。",
        "有限的种子和时序不支持统计显著性判断。现有动作在轻中扰动具有可测的控制权，强扰动补偿余量较小。只有闭环独立收益支撑时，才考虑历史状态或更大运动任务。"
    ]
    report("61_Phase2C6验收与下一阶段决策.md",[
        "# Phase 2C-6 验收与下一阶段决策", "",f"**{decision['decision']}**","",
        *evidence,"",
        *table(["新方法","双 IAE 胜场/3","平均加载 IAE/经典","平均卸载 IAE/经典","无保护且完成","XYZ/变化率门槛"],[
            [m,f"{v['joint_IAE_wins']}/3",f"{v['ratio_vs_classical']['load_iae_deg_s']:.4f}",
             f"{v['ratio_vs_classical']['unload_iae_deg_s']:.4f}",v["all_success"],v["xyz_gate"] and v["slew_gate"]]
            for m,v in decision["independent"].items() if not m.startswith("V1")]),"",
        f"独立和强回归共 {len(d)} 次真实闭环，{int(d.success.sum())} 次达到原协议。硬保护依据原限位、共享预算、weld 与部分几何覆盖。Qt 与无 GUI 的匹配确定性评价逐位核对 {q['steps']} 个完整积分状态。","",
        f"历史审计核对 {hist['matched']} 项持久文件，另外验证 {len(hist['verified_test_repairs'])} 处授权的测试检查器修复。唯一缺失例外为历史清单中的临时 Word 锁。[修复 diff]({ARCHIVE}/lock_fix.diff) · [修复哈希]({ARCHIVE}/lock_fix_receipt.json) · [历史核对]({ARCHIVE}/history_after.json)","",
        f"回归和阶段验收分三批执行，共 {test['passed']} 项通过、{test['failed']} 项失败、{test['xfailed']} 项历史预期失败。[测试汇总](../results/phase2c6_test_summary.json)","",
        "## 实际视频","",*videos,"",
        "图上角度尺度单独标注。实际机器人画面来自训练或评价现场 RGB，几何保持真实比例。","",
        "若继续研究，优先做等交互预算的固定分布续训对照，分离课程分布与额外更新的影响。再单独审计不同扰动强度的价值目标尺度及卸载后的补偿方向，继续沿用当前动作权限和安全门槛。", "",
        "## Windows 查看命令","",
        "```powershell",
        "Set-Location __PROJECT_ROOT__",
        "$py = 'D:\\miniconda\\envs\\stewart_rl\\python.exe'",
        "& $py scripts/view_phase2c6_policy.py --task E_mix",
        "& $py -m tensorboard.main --logdir results/runs/phase2c6_20261107_v1/training --host 127.0.0.1 --port 6036",
        "```","",
        "本轮在 Phase 2C-6 停止。"])
    

if __name__=="__main__":
    main()
