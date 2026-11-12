"""Write audit/report/validation from actual scan, raw logs and test evidence."""
import _bootstrap,json,platform
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from src.model import ROOT,sha,write_json
from analyze_phase2c3a_workspace import RUN

def main():
    comparison=pd.read_csv(ROOT/'results/phase2c3a_controller_comparison.csv');scan=pd.read_csv(ROOT/'results/phase2c3a_workspace_scan.csv');audit=json.loads((RUN/'model_geometry_audit.json').read_text(encoding='utf-8'));limits=json.loads((ROOT/'results/phase2c3a_workspace_limits.json').read_text(encoding='utf-8'));record=json.loads((RUN/'recording_invariance.json').read_text(encoding='utf-8'));recalc=json.loads((RUN/'independent_recomputation.json').read_text(encoding='utf-8'));video=json.loads((RUN/'video_validation.json').read_text(encoding='utf-8'));cfg=json.loads((ROOT/'configs/phase2c3a_large_workspace.json').read_text(encoding='utf-8'));local=pd.read_csv(ROOT/'results/phase2c3a_local_model_audit.csv');mechanism=pd.read_csv(ROOT/'results/phase2c3a_tracking_mechanism.csv');testlog=(ROOT/'results/phase2c3a_full_tests.log').read_text(encoding='utf-8');test_summary=next(x for x in reversed(testlog.splitlines()) if 'passed' in x and 'xfailed' in x)
    frozen=json.loads((RUN/'history_freeze.json').read_text(encoding='utf-8'));changed=[k for k,v in frozen.items() if not (ROOT/k).exists() or sha(ROOT/k)!=v];write_json(RUN/'history_preservation.json',dict(checked=len(frozen),changed=changed,passed=not changed))
    safety=dict(trials=len(comparison),complete=int(comparison.complete.sum()),successful=int(comparison.success.sum()),warnings=int(comparison.warning_count.sum()),native_contact_observations=int(comparison.native_contact_observations.sum()),native_contacts_are_not_collision_proof=True,minimum_mesh_clearance_m=float(comparison.expanded_mesh_gap_m.min()),minimum_legacy_proxy_m=float(comparison.legacy_proxy_gap_m.min()),maximum_weld_translation_m=float(comparison.max_weld_translation_m.max()),maximum_weld_angle_rad=float(comparison.max_weld_angle_rad.max()),maximum_force_N=float(comparison.max_force_N.max()),native_saturation_steps=int(comparison.native_saturation_steps.sum()),failures=comparison[~comparison.success][['folder','failure_reason','progress']].to_dict('records'),missing_hinge_hard_stops=30,connected_hull_pairs_excluded=len(audit['excluded_pairs']),mechanical_hardware_certified=False)
    write_json(ROOT/'results/phase2c3a_safety_audit.json',safety)
    samples=[]
    for f in RUN.rglob('summary.json'):
        r=json.loads(f.read_text(encoding='utf-8'));samples.append(dict(folder=str(f.parent),physical_steps=r.get('physics_steps',0),complete=r.get('complete',False),failure=r.get('failure')))
    steps=sum(x['physical_steps'] for x in samples);write_json(RUN/'execution_inventory.json',dict(total_physical_steps=steps,episodes=samples,pytest_temp_integrations_excluded=True))
    dynamics50=comparison[(comparison.amplitude_m==.05)&(comparison.rise_s==10)&(comparison.tau_s==0)&(comparison.external_Fz_N==0)];slow=comparison[(comparison.rise_s==10)&(comparison.tau_s==0)&(comparison.external_Fz_N==0)];max_amp=float(comparison[comparison.success].amplitude_m.max()) if comparison.success.any() else None
    decision='LARGE-MOTION RL CONDITIONALLY BLOCKED' if max_amp else 'LARGE-MOTION RL BLOCKED'
    reasons=['Missing physical angle limits for30 hinges and unverified connected socket interiors; modeled queries are not full hardware clearance certification.','Observed nominal tracking error is largely consistent with missing reference damping/inertia feedforward. Establish an independently named stronger large-motion baseline before attributing it to unknown dynamics or asking PPO to compensate it.']
    write_json(ROOT/'results/phase2c3a_validation.json',dict(stage_decision=decision,reasons=reasons,completed_large_motion_simulation=bool(len(dynamics50)==3 and dynamics50.success.all()),maximum_tested_successful_reference_amplitude_m=max_amp,all_checks=dict(history_unchanged=not changed,independent_metrics=recalc['passed'],recording_invariant=record['passed'],video_valid=video['passed']),test_summary=test_summary,no_RL_training=True))
    nohard='30个被动hinge均未提供原生机械角度限位。其模型状态可计算，但真实硬限位余量为UNKNOWN，不能记作“无限安全余量”。连接关节、凹形套接与伸缩副内部按显式装配规则豁免，尚未建立真实内部干涉与硬止挡模型。'
    geo=f'''# Phase2C-3A大工作空间几何审计

日期2026-11-03。全部离线查询使用独立MjData，未写执行中的qpos或修改历史MJCF。原source/控制器/配置/权重哈希冻结于`history_freeze.json`。

模型nq={audit['nq']}、nv={audit['nv']}、nu={audit['nu']}（前6支链力，后6外载通道）、6个weld、6个slide、30个hinge、1个free，步长1ms。动平台质量20kg，质心与惯量直接导出至`model_geometry_audit.json`。六支链均按真实7个body/6个关节的连接链求IK；基座点、平台点、轴线与qpos地址逐支链存入同一JSON，没有用README理想弦长代替。

## 扫描结果

世界Z在Home基础上[-100,+100]mm，每2mm扫描，共101点。采样点均通过完整链式IK、现有原生关节范围、归一化Jacobian≤50及查询间隙≥5mm。**这只说明扫描区间内没有找到边界，不证明真实几何边界就是±100mm。** 正、负方向分别从Home连续求解；负向没有开展动态执行。

+30mm参考路径最高归一化条件数{limits['candidates']['L30']['maximum_condition']:.6f}，+50mm为{limits['candidates']['L50']['maximum_condition']:.6f}；范围内没有发生条件数阈值失效。各采样的六缸相对行程、限位余量及30个被动角度均保留于扫描CSV。原slide范围为[-0.1,+0.35]m，不能将它等同平台Z范围。

两条候选均获得**模型内受限动力学试验**的几何前提；不称为完整机械安全审计通过。被动关节、约束、距离、可达性和动态验证分别记录，不把IK成功当作动态成功。

## 扩展碰撞查询

原机器人网格的contype/conaffinity为0，floor仍保留原设置，故ncon=0不证明没有干涉。新版本直接查询所有原模型视觉网格凸包和地面，887个配对、103个带原因的装配豁免；没有开启原物理碰撞，也没有更改模型质量/惯量。非相邻支链的完整几何体均参与，与旧仅覆盖部分长杆的平台代理查询独立。

MuJoCo的网格查询基于凸几何，凹形套接会被凸包填充，因此连接组件的内部关系必须显式区别。距离超过distmax=.15m只记下界。依据：[MuJoCo计算文档](https://mujoco.readthedocs.io/en/stable/computation/)，[原生距离实现](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_support.c)。此机制能够保守检验非连接部件分离，不能认证被排除的关节内腔。

上/下平台间距另用2m查询范围补算，写入扫描CSV的`platform_gap_query_2m_m`列；原`platform_gap_m=.15`保留并标明只是下界，没有把截断值当真实间距。补算范围内平台间最小查询距离{scan.platform_gap_query_2m_m.min():.6f}m。

{nohard}

新参考路径还在0.5mm高度节点重新执行完整链IK、静力平衡和几何查询，节点中点独立IK与Hermite行程插值核验。另在Z=-2/0/30/50/52mm、XY±2mm、RP±0.5度、Yaw±0.2度的160个角点进行查询，均通过；这不是高维连续不变集的证明。

未检测到的边界：扫描内没有机械行程、Jacobian或已覆盖间隙首次违规；真正正/负边界仍未找到。未认证：连接组件内部硬止挡、实际被动副极限、未建模软管及实机制造/弹性误差。
'''
    (ROOT/'docs/38_Phase2C3A大工作空间几何审计.md').write_text(geo,encoding='utf-8')
    design=f'''# Phase2C-3A大位移动力学控制设计

新配置`configs/phase2c3a_large_workspace.json`，运行目录`{RUN}`。历史±5mm配置、旧RL环境、奖励与权重均不变。

## 参考及保护

L30/L50分别是Home→世界Z+30/+50mm→Home，幅值为单程高度。保持姿态。时间轨迹：1秒Home，T秒五次上升，1秒顶部保持，T秒五次回落，1秒Home；总时长3+2T。慢速T=10s、候选中速6s、候选较快4s。初末速度/加速度为0，空间路径来自完整链IK而非理想杆长。

L50三档峰值速度9.375/15.625/23.4375mm/s，峰值加速度2.88675/8.01875/18.0422mm/s²。慢速23秒每个参考毫秒检查行程；0.5mm几何节点及中点复核。名义静力+质量/阻尼参考项只用于力余量筛查，不作为新控制前馈下发，也不替代实际力保护。后续速度条件预留100N名义力余量，实际仍为±400N。

独立实际Z边界[-2,+52]mm来自扫描支持的区间及在线几何监控，给50mm参考保留2mm跟踪余量。XY±2mm、RP±0.5度、Yaw±0.2度、weld平移1e-4m/旋转1e-4rad、Jacobian≤50、原行程和原生力范围都保持。每个物理步检查有限数值、原生关节、闭链、姿态与工作区；原代理和新增网格查询周期20ms，属于响应式离散保护。

`LargeOwner`复用原力驱动PLANT_STEP，仅在独立函数命名空间中绑定新安全检查，不修改全局旧检查。失败仍将目标转为原受限保持，真实积分继续记录200ms，不把惯性继续运动抹掉。参考无效拒绝，提前终止不算成功。完整物理状态、ctrl、命令与真实执行器力独立保存。

## 对照及门禁

C0、C4-Observer（C3+原Yaw补偿）和fourDlead都直接调用原代码和参数，不重整定。慢速2幅值×3方法共6回合；全部通过才进行L50速度6回合与滞后/负载6回合。T4需要T6三方法均通过。无重试、不筛选失败。

负载矩阵：L50慢速，C4/fourDlead分别在5ms无外载、10ms无外载、5ms且世界Fz=-5.5N下运行。加载在上升70%时刻，卸载在回落50%时刻。合成一阶滞后沿用原实现，零滞后逐步等价验证保留。

Home新旧Owner零位移测试逐步比较qpos/qvel/ctrl；现场RGB开关各1800真实步比较全积分状态完全一致。录像只读取实际live状态，颜色改在渲染scene，不改变物理几何。独立回放可以显示原RGB，不能称为再次执行。

{nohard}

本轮没有任何PPO/BC/SAC训练，也没有将旧3mm PPO部署到50mm。旧255维特征中的slide缩放为1mm，50mm时若照搬会有严重分布偏移和裁剪；参考Z特征50/5=10已达边界。动作单位与Jacobian接口仍可形式兼容，策略能力不能因此迁移。
'''
    (ROOT/'docs/39_Phase2C3A大位移动力学控制设计.md').write_text(design,encoding='utf-8')
    table='| 方法 | 幅值mm | 单程秒 | 滞后ms | 外载N | 完成 | Z RMSE mm | XY峰值mm | 最大力N |\n|---|---:|---:|---:|---:|---|---:|---:|---:|\n'
    for r in comparison.itertuples():table+=f'| {r.method} | {r.amplitude_m*1000:.0f} | {r.rise_s:g} | {r.tau_s*1000:g} | {r.external_Fz_N:g} | {r.success} | {r.z_rmse_mm:.6f} | {r.xy_peak_mm:.6f} | {r.max_force_N:.3f} |\n'
    performance=f'''# Phase2C-3A经典控制器性能对比

全部数据来自真实1ms积分日志独立复算，表中保留全部成功与失败，未用短回合误差掩盖不完成。负载回合与无外载不可直接混成整体改进率。

{table}

## 误差机制

固定幅值下速度变化见`phase2c3a_speed_comparison.png`，每回合完整轨迹误差、SO(3)、相位对齐窗口、回Home误差、六缸相对行程、力及补偿预算见CSV。相位指标只是在±200ms窗口作最小均方对齐，不是纯时延辨识。

Home局部名义Z等效质量{local.home_Mzz_kg.iloc[0]:.6f}kg，50mm处{local.nominal_reduced_Mzz_kg.iloc[-1]:.6f}kg；50mm静态Z偏差{local.static_z_model_error_N.iloc[-1]:.6f}N。这些是独立名义模型计算，不是实验测得的外载。不能据位移较大就断言局部模型失效。

原C0使用静态前馈+关节PD，原C4观察器保留名义阻尼/惯性模型；未下发新的参考阻尼与惯性前馈。根据线性化误差方程，对准静态误差作离线预测，与慢速实际误差的相关性很高；详细值见`results/phase2c3a_tracking_mechanism.csv`。这支持已知阻尼与参考动态项是重要来源，但局部线性近似不能证明全部原因。高速、滞后、加载时需分别检查。

因此，残余误差不能全部标成“未知扰动”“PPO才能解决”。后续应单独命名并验证参考速度/加速度前馈等强传统基线，再比较学习型补偿；本轮没有实现或重整定这个新控制器。

六缸图明确标为Relative slide stroke，它是移动副相对行程，不是整根支链绝对长度。实际力和限速/补偿预算独立于轨迹目标；参考峰值速度不能当作实际速度硬保证。
'''
    (ROOT/'docs/40_Phase2C3A经典控制器性能对比.md').write_text(performance,encoding='utf-8')
    legs='\n'.join(f'- Leg{i}：fourDlead慢速L50实际相对行程范围{float(dynamics50[dynamics50.method=="fourDlead"][f"leg{i}_actual_range_mm"].iloc[0]):.6f}mm。' for i in range(1,7)) if len(dynamics50[dynamics50.method=='fourDlead']) else '该工况未完成，不填写虚构值。'
    answers=f'''# Phase2C-3A验收报告

日期2026-11-03。结论：**{decision}**。

大位移真实动力学能力已按本模型验证，最多成功测试的单程参考幅值{max_amp*1000 if max_amp else 0:g}mm；新大位移RL仍应先完善强传统基线与明确机械/连接内部覆盖限制。本轮不训练PPO。

## 实际完成

101点±100mm完整链几何扫描；160个保护范围角点查询；独立大范围配置；Home等价与录像不变性；{len(comparison)}次正式真实动力学回合，其中成功{int(comparison.success.sum())}、不成功{int((~comparison.success).sum())}。包括30/50mm低速、通过门禁后的速度条件和滞后/加载条件，全部结果保留于文档40及CSV。

本run共{steps}次真实积分，含观察器3600步；pytest临时动力学测试另计。原历史文件核验{len(frozen)}个，改变{len(changed)}个。独立指标复算通过：{recalc['passed']}。最终测试：`{test_summary}`，xfail保持历史规则，不计普通通过。

## 25项研究问题的回答

1–2. 正/负Z在已扫描的0..+100与0..−100mm采样点均满足模型内几何条件；没有找到真实边界，不能把扫描端点当机构极限。负向动态未执行。

3–4. +30/+50mm通过完整链IK、现有原生限位、Jacobian及非连接几何凸包间隙的**仿真前检**；被动机械硬止挡与关节内腔未认证，不能称完整机械安全审计通过。

5–6. 扫描范围内没有出现首次限位/条件数/已覆盖间隙违规。{nohard} 原native ncon=0不作为无碰撞证据。

7–8. 慢速30/50mm真实执行结果见文档40：6回合中{int(slow.success.sum())}/6成功。既有C0、C4、fourDlead全部按真实力与mj_step运行，原生模型无改变。

9. 50mm平台位移与六缸行程不同，以下为实际记录：
{legs}

10–12. 各控制器完成率、误差和代价均在文档40逐工况给出；不能把fourDlead与C4的微小差异宣称显著改进。没有重新整定任何一个冻结控制器。

13–15. 名义模型变化与误差机制独立核查。静态局部偏差在50mm约{local.static_z_model_error_N.iloc[-1]:.6f}N，不能将全部运动误差归因于它。固定速度时幅值、固定幅值时速度的对照分别保留，现有参考阻尼/惯性前馈不足是重要候选原因，不是已证实的唯一原因。

16–19. 正式回合最大条件数{comparison.max_jacobian_condition.max():.6f}，最大实际力{safety['maximum_force_N']:.6f}N，原生饱和步数{safety['native_saturation_steps']}。最大weld平移{safety['maximum_weld_translation_m']:.3g}m、旋转{safety['maximum_weld_angle_rad']:.3g}rad。各轴姿态漂移、补偿限幅、离散间隙及失败事件均保留；发生失败不记成功。最大实际力变化率{comparison.max_force_rate_N_s.max():.6f}N/s不是共享补偿20N/s的同一指标。

20. 当前成功动态测试最大参考幅值为{max_amp*1000 if max_amp else 0:g}mm，实际峰值最大{comparison.actual_z_peak_m.max()*1000:.6f}mm。参考和真实值分别记录，没有证明更大幅值的动态能力。

21–22. 已执行的5/10ms滞后与5ms+−5.5N加载组合见文档40，不把单因素通过当作组合自动通过。未执行10ms+负载，不用它作结论。

23. 后续候选为已验证的50mm往返、先较慢时序，再选择已通过的滞后/小负载条件；但须先建立参考动态前馈的强传统基线，重新冻结训练/验证/最终集。历史3mm策略不部署、不重训。

24. 纯Z任务先保留单Fz补偿权限；没有证据仅因机构六自由度就需要六维Actor。XY或姿态问题若成为主导，先验证相应输入可控性，不让Fz替代横向控制。

25. **需要独立大范围强基线研究。** 这不是对原C4参数的静默修改。本阶段已有动力学可达证据，但不据此自动进入RL。

## 视频与复现

`assets/videos/stewart_phase2c3a_large_motion.mp4`来自执行现场RGB，标记LARGE-WORKSPACE DYNAMIC MOTION VALIDATION；{video['decoded_frames']}帧、{video['seconds']:.2f}秒、2倍播放。状态哈希逐源帧核验通过{video['passed']}。它没有PPO内容。

```powershell
Set-Location __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\watch_phase2c3a.py --trial fourDlead_L50_T10_tau0_load0
Start-Process .\\assets\\videos\\stewart_phase2c3a_large_motion.mp4
```

该GUI播放真实归档RGB与同步曲线，不重新积分。Windows窗口已实际测试自动播放与正常关闭，截图`results/phase2c3a_gui_validation.png`。

独立复算入口`compare_phase2c3a_controllers.py`。`phase2c3a_dynamic_tracking.csv`每20ms汇总，并保留每回合末行；全部1ms原始数据仍在各回合`physics.csv.gz`及`integration_states.npz`，不从降采样CSV计算主指标。

未执行：负Z动力学、超过50mm动态、全姿态工作空间、10ms+外载、未建模关节止挡/内部CAD碰撞、传统新前馈控制器、任何RL训练、实机部署与安全认证。完成后暂停。
'''
    (ROOT/'docs/41_Phase2C3A验收报告.md').write_text(answers,encoding='utf-8')
    files={}
    paths=list(RUN.rglob('*'))+list((ROOT/'scripts').glob('*phase2c3a*.py'))+list((ROOT/'src/workspace').glob('*.py'))+list((ROOT/'tests').glob('*phase2c3a*.py'))+list((ROOT/'docs').glob('*Phase2C3A*'))+list((ROOT/'assets/plots').glob('phase2c3a*'))+[ROOT/'assets/videos/stewart_phase2c3a_large_motion.mp4',ROOT/'configs/phase2c3a_large_workspace.json']+list((ROOT/'results').glob('phase2c3a*.json'))+list((ROOT/'results').glob('phase2c3a*.csv'))
    for f in paths:
        if f.is_file() and f.name!='phase2c3a_artifact_manifest.json':files[str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    write_json(ROOT/'results/phase2c3a_artifact_manifest.json',dict(count=len(files),files=files));print(decision);print(safety);print(test_summary)
if __name__=='__main__':main()
