"""Close the pretraining stage using measured physics and the history gate."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from audit_phase2c9_dynamics import ROOT, RUN
from phase2c8_protocol import read, write_json


def table(frame):
    names = list(frame.columns)
    rows = [names, ["---"] * len(names), *frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(f"{x:.7g}" if isinstance(x, float) else str(x) for x in row) + " |" for row in rows)


def save(index, title, body):
    (ROOT / f"docs/{index}_Phase2C9{title}.md").write_text(f"# Phase 2C-9 {title}\n\n{body}\n", encoding="utf8")


def audit(m):
    d = pd.read_csv(Path(m["folder"]) / "physics.csv.gz")
    task = m["task"]
    actual = Rotation.from_euler("xyz", d[["rpy_0","rpy_1","rpy_2"]], degrees=True)
    reference = Rotation.from_euler("xyz", d[["rpy_ref_0","rpy_ref_1","rpy_ref_2"]], degrees=True)
    error = (actual * reference.inv()).magnitude() * (180 / np.pi)
    force = d[[f"force_{i}" for i in range(6)]].to_numpy()
    additional = d[[f"F_command_{i}" for i in range(6)]].to_numpy() - d[[f"F_C0_{i}" for i in range(6)]].to_numpy()
    xyz = (d[[f"p_{i}" for i in range(3)]].to_numpy() - d[[f"p_ref_{i}" for i in range(3)]].to_numpy()) * 1000
    omega = d[[f"twist_actual_{i}" for i in range(3,6)]].to_numpy()
    values = dict(so3_peak_deg=error.max(), xyz_rmse_mm=np.sqrt(np.mean(np.sum(xyz**2, axis=1))),
                  omega_peak_rad_s=np.linalg.norm(omega, axis=1).max(), force_peak_N=np.abs(force).max(),
                  force_rate_N_s=np.abs(np.diff(force, axis=0)).max()*1000, add_peak_N=np.abs(additional).max(),
                  add_rate_N_s=np.abs(np.diff(np.vstack([np.zeros(6), additional]), axis=0)).max()*1000)
    for label, edge in (("load", "on"), ("unload", "off")):
        start = round(task[edge]*1000)
        values[label+"_iae_deg_s"] = error[d.input_step.between(start, start+999)].sum()*.001
    np.testing.assert_array_equal(d.input_step, np.arange(len(d)))
    assert m["steps"] == len(d) and d.failure.isna().all() and m["success"]
    for key, value in values.items():
        np.testing.assert_allclose(value, m[key], atol=1e-10, rtol=1e-9)
    assert values["add_peak_N"] <= 5+1e-8 and values["add_rate_N_s"] <= 20+1e-6 and values["force_peak_N"] <= 400
    return [dict(index=m["index"], source=m["folder"], metric=key,
                 recomputed=value, recorded=m[key]) for key, value in values.items()]


def main():
    preflight = [read(p) for p in sorted((RUN / "preflight_v2").glob("*/metrics.json"))]
    classical = [read(p) for p in sorted((RUN / "classical_diagnostic").glob("*/metrics.json"))]
    recording = read(RUN / "video_capture/09/metrics.json")
    assert len(preflight) == 14 and len(classical) == 2
    checks = pd.DataFrame([r for m in preflight + classical + [recording] for r in audit(m)])
    checks.to_csv(RUN / "physical_metric_recomputation.csv", index=False)
    frame = pd.DataFrame(preflight)
    compare = pd.DataFrame([dict(m, method=m.get("method", "frozen_zero_residual")) for m in preflight + classical])
    compare.to_csv(ROOT / "results/phase2c9_controller_comparison.csv", index=False)
    history = pd.read_csv(ROOT / "results/phase2c9_history_information.csv")
    gate = read(RUN / "history_gate.json")
    assert not gate["eligible"]
    video = read(RUN / "video_validation.json")
    freeze = read(RUN / "history_verification.json")
    tests = {}
    for label in ("new", "historical"):
        path = RUN / f"{label}_tests.xml"
        if path.exists():
            suites = ET.parse(path).getroot()
            cases = list(suites.iter("testcase"))
            failures = [x.attrib["name"] for x in cases if x.find("failure") is not None or x.find("error") is not None]
            tests[label] = dict(total=len(cases), failed=failures)
    assert tests["new"]["total"] == 2 and not tests["new"]["failed"]
    safety = dict(preflight_trials=len(frame), preflight_complete=int(frame.success.sum()),
                  preflight_physical_steps=int(frame.steps.sum()), classical_trials=len(classical),
                  recorded_repeated_trial_steps=recording["steps"], max_native_force_N=float(compare.force_peak_N.max()),
                  max_shared_add_N=float(compare.add_peak_N.max()), max_shared_slew_N_s=float(compare.add_rate_N_s.max()),
                  minimum_proxy_gap_m=float(compare.clearance_m.min()), max_weld_angle_rad=float(compare.weld_angle_rad.max()),
                  independently_recomputed_metrics=len(checks), max_recomputation_error=float(abs(checks.recomputed-checks.recorded).max()),
                  tests=tests, historical_archive=freeze)
    write_json(ROOT / "results/phase2c9_safety_audit.json", safety)
    write_json(ROOT / "results/phase2c9_final_decision.json", dict(status="PHASE 2C-9 BLOCKED",
        reason="No candidate met both attitude deterioration and incremental history transfer criteria",
        part_a_saved=True, history_gate=gate, ppo_decisions=0, ppo_models=[], formal_generalization_executed=False,
        preflight_only=True, classical_diagnostic_only=True, optional_budget_extension_executed=False, video=video))
    fields = ["reference_id","kind","value","success","steps","load_iae_deg_s","unload_iae_deg_s",
              "so3_peak_deg","omega_peak_rad_s","xyz_rmse_mm","force_peak_N"]
    plant = read(Path(preflight[0]["folder"]) / "phase2c9_plant.json")["actual"]
    save(82, "未知动力学建模与安全审计", f"""实际运行14次配对MuJoCo预检，共{int(frame.steps.sum())}步。Home与已审核的50 mm联合轨迹均使用 [0.6, −0.8] N·m，4.4 s加载、6.2 s卸载。积分1 ms、观测周期20 ms。冻结C4/fourDlead/FF-VA，零附加PPO动作。以下IAE单位deg·s、峰值单位deg、角速度rad/s。

{table(frame[fields])}

名义平台质量{plant['platform_mass_kg']} kg，主惯量{plant['platform_inertia_kg_m2']} kg·m²，局部质心{plant['platform_com_local_m']} m。独立Plant质量按1.03或1.10缩放，固定几何密度缩放同步调整主惯量。质心沿局部X平移0.5或1.5 mm，平移整个惯性分布，关于自身质心的主惯量保持。原5 ms合成一阶驱动力滞后对照2、6 ms；驱动力内部状态只归档。mj_setConst使用scratch MjData，live状态只在复位初始化后做mj_forward。

全部原保护保持。第一次启动因引用了数值相同但JSON整数/浮点表示不同的Home参考而找不到缓存，四个失败日志保留，使用原配置中的精确Home引用重新执行，历史缓存无需补写。

峰值实际力 {safety['max_native_force_N']:.4f} N，共享附加力 {safety['max_shared_add_N']:.4f} N，附加力最大变化率 {safety['max_shared_slew_N_s']:.8f} N/s。最小几何代理距离 {safety['minimum_proxy_gap_m']:.6f} m。这里只验证已运行的有界轨迹。
""")
    scores = history[["kind","value","held_out_reference","frames","balanced_accuracy"]]
    save(83, "历史反馈信息有效性", f"""使用真实预检观测作离线二分类。标签是名义Plant或单因素变化Plant，只给离线诊断器。每种条件每条轨迹抽取相同的决策时刻，每隔5决策采样，排除前10决策。按整个参考工况交叉留出，两个方向是Home训练→联合验证与联合训练→Home验证。岭回归归一化只拟合训练折，窗口长度1、5、10，当前帧完全共享。5帧的首尾间隔80 ms，10帧180 ms；窗口并不推迟最新观测。

{table(scores)}

研究准入规则由本次实现设定：{gate['criterion']}。

22 kg条件下，5帧在留出联合任务将准确率从85.57%提升至98.66%，10帧降至82.89%。留出Home的三者均100%，没有提升空间；Home姿态指标退化也不到5%。这是单折的可区分信息增益，不能据此断言历史在所有工况无效。每类仅两种任务和单一外载波形；没有跨参数值、时刻或方向的充分证据。未来需要增加能留出不同联合运动的组，并对单帧接近满分的工况单独设评价口径。
""")
    save(84, "自适应强化学习环境设计", """实际实现 StewartAdaptiveEnv 与 HistoryObservation。环境复用Phase 8任务RNG、奖励V2、58维因果测量和20步物理循环。每次reset建立独立Plant及名义控制器，Plant参数只在首次积分前施加；历史清空并复制初始帧填充。当前帧保留在堆叠末尾，候选维度290／580。任务参数与内部驱动力不会加入Actor观测。

名义零动作相对Phase 8，以及22 kg条件下单帧和历史包装，分别在跨加载和卸载的短轨迹上逐位核对观测、奖励、qpos、qvel和ctrl。两个针对性测试通过。正式随机化探索的完整安全预检与训练阶段恢复未执行。沿用每轴±0.05 N·m和原共享±5 N、20 N/s及原生±400 N。
""")
    save(85, "单帧与历史PPO训练", """本轮正式PPO决策数为0。历史准入没有候选通过，42、123、2026的两方法8192决策Pilot以及32768决策扩展均未执行，未生成模型权重、训练曲线或训练视频。

下一次若通过准入，应冻结同一任务采样序列、V2奖励、动作权限与预算。控制评价中以固定权重使用历史反馈，PPO参数更新只在训练端发生。
""")
    save(86, "未知动力学泛化评价", f"""完成了同一参数与外载下的闭环诊断。下表只比较22 kg工况的冻结传统控制与50 Hz经典附加补偿。经典方法使用因果姿态和角速度PD，加泄漏积分（1/s，Ki=10，积分限幅±0.025 N·m），饱和时不增加积分；总残差仍经原二维动作和共享预算。系数为本轮工程诊断设定，未做独立调参。

{table(compare[compare.value.eq(1.10)][["method",*fields]])}

独立泛化矩阵未开放。这里的两个工况来自准入任务，尚无未见参数、方向、时刻或组合上的策略泛化结论。逐条1 ms日志复算了SO(3)、IAE、角速度、位置与执行器指标。
""")
    save(87, "验收报告", f"""**PHASE 2C-9 BLOCKED，停在历史信息与任务准入。** Part A四份报告和三张CSV已保存。真实仿真已完成预检、两条经典诊断及一条重复录制。没有进行正式PPO训练和独立泛化。

历史归档核验 {freeze['count']} 个既有文件，变化 {len(freeze['changes'])} 个。测试结果详见 safety_audit.json 与实验目录JUnit。最终 {len(checks)} 项物理指标独立复算通过，最大绝对误差 {safety['max_recomputation_error']:.3g}。

真实基线视频共 {video['frames']} 帧，{video['seconds']} 秒。录制轨迹与对应预检的积分状态逐位相同，逐帧核对时间戳和qpos SHA。字幕标明Plant 22 kg与名义20 kg。未生成历史PPO训练或策略对比视频。

复现入口为 scripts/audit_phase2c9_dynamics.py、scripts/validate_phase2c9_history.py 与 scripts/render_phase2c9_videos.py。预检轨迹归档在 results/runs/phase2c9_20261109_v1/preflight_v2，正式结果不覆盖既有轨迹。准入的两个工况和单波形不足以验证历史PPO恢复更快。
""")


if __name__ == "__main__":
    main()
