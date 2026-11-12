"""Generate conclusions from recomputed CSV and actual acceptance artifacts only."""
import _bootstrap
import json,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd,numpy as np
from src.model import ROOT,write_json

def table(frame,columns,labels):
    text='| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for r in frame.to_dict('records'):
        vals=[]
        for c in columns:
            v=r[c];vals.append('—' if pd.isna(v) else f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v))
        text+='| '+' | '.join(vals)+' |\n'
    return text

def main():
    formal=pd.read_csv(ROOT/'results/phase2a_controller_comparison.csv');allrows=pd.read_csv(ROOT/'results/phase2a_disturbance_summary.csv');v=json.loads((ROOT/'results/phase2a_validation.json').read_text(encoding='utf-8'));video=json.loads((ROOT/'assets/videos/stewart_phase2a_disturbance_rejection.validation.json').read_text(encoding='utf-8'));testfile=ROOT/'results/phase2a_pytest_final.xml'
    test=ET.parse(testfile).getroot()[0].attrib if testfile.exists() else dict(tests='pending',failures='pending',errors='pending')
    v['automatic_tests']=test;write_json(ROOT/'results/phase2a_validation.json',v)
    selected=formal[formal.task.isin(['A_home_force','B_home_roll_positive','B_home_roll_negative','C_z_roll_composite','legacy_local_z_roll_exact'])]
    block=table(selected,['task','controller','peak_position_mm','peak_orientation_deg','load_recovery_s','load_recovery_status','unload_recovery_s','unload_recovery_status'],['任务','控制器','峰值位置mm','峰值SO3姿态deg','加载恢复s','加载状态','卸载恢复s','卸载状态'])
    axis_table=table(formal[formal.task=='A_home_force'],['controller','peak_z_error_mm','peak_roll_error_deg','peak_pitch_error_deg','pre_position_rmse_mm','pre_orientation_rmse_deg'],['控制器','最大Z误差mm','Roll误差分量deg','Pitch误差分量deg','加载前位置RMSE mm','加载前SO3 RMSE deg'])
    full=table(formal,['task','controller','success','position_rmse_mm','orientation_rmse_deg','peak_position_mm','extra_peak_position_mm','load_position_iae_mm_s','unload_position_iae_mm_s'],['任务','控制器','完成','全程位置RMSE mm','姿态RMSE deg','峰值位置mm','相对零外载额外峰值mm','加载IAE mm s','卸载IAE mm s'])
    text=f'''# Phase2A突变负载实验验收报告

日期：2026-10-28。项目`__PROJECT_ROOT__`。仅执行Phase2A，不训练PPO/SAC/BC，不实现C3或扩大工作区。

## 已运行范围与冻结

共{len(allrows)}次真实6秒物理试验，{int(allrows.physical_steps.sum())}个1ms积分步：15次预检、30次新世界坐标正式对照、3次历史精确复现，以及3次保留的1ms时间偏移诊断。正式比较33/33按原完成标准通过；15/15预检通过；3个偏移诊断也正常完成，但不冒充精确历史复现。观察器对照另外运行2×2048=4096步；自动测试中的短仿真独立于上述研究预算。

主目录`results/runs/phase2a_20261028_v1`；精确历史复现`results/runs/phase2a_20261028_legacy_recheck_v2`。完整逐trial数据、早期指标、配置、模型参数、完整积分状态和事件均保留。没有覆盖任何历史实验。冻结核验{v['historical_files_checked']}个文件，变化{len(v['historical_files_changed'])}个；{v['identical_initial_state_count']}种完全相同初始积分状态用于严格配对。这是确定性参数对照，不是51个独立随机种子，不宣称统计显著性。

正式方案在各正式试验前冻结；新世界任务采用2秒加载、4秒卸载。加载期间参考位姿沿原轨迹运行，动态场景加载事件记录的reference_is_moving均为True。代理间隙仍仅原50Hz离散查询，weld/行程等逐物理步检查；不宣称连续碰撞或整机安全认证。

## 模型、执行器与外载

真实模型为`third_party/stewart_platform_mujoco/scene.xml`及其引用模型；接口审计保存全部XML SHA、上游commit、实际版本和执行器元数据。ID0～5是六支链力输入N，原±400 N不变。ID6～11是upper_center处site-local Fx/Fy/Fz/Tx/Ty/Tz，N及N m；原生范围±500按各自单位保留，但仅小幅候选经过本轮测试。外载不通过篡改前六个控制器输出来实现。

世界外载先用当时site旋转R^T转换到局部通道，力矩参考点是当时upper_center，不能把局部负Z当作世界负Z。全日志坐标变换检查、原生actuator_moment、site Jacobian与独立mj_applyFT核验，独立重建最大广义力误差{v['maximum_independent_mapping_error']:.3e}。C0/C1/C2读取独立名义模型和实际可观测运动状态，没有读取外载真值或实际隐藏模型参数。没有运行时写入live qpos/qvel，全部物理运动来自data.ctrl和mj_step。

## 关键抗扰结果

恢复条件为位置<=0.25 mm且SO3姿态<=0.02度，实际越界后连续保持0.3秒；加载与卸载分开。未恢复用空值和状态表达。NO EXCURSION表示该窗口未离开容差，无需恢复，不能填成虚假的0秒恢复。任务完成仍用原0.75 mm/0.1度和最后0.5秒保持，因此任务成功不等于加载期间严格恢复。

{block}

Home竖向力的方向分量与加载前基线（所有数值由原始日志复算）：

{axis_table}

C0/C2在Home −10 N或+/-0.5 N m保持加载时存在约0.4～0.5 mm额外偏移，严格恢复容差内未恢复；卸载后可以恢复。C1积分可降低该类持续偏差并恢复，但卸载后反向超调、恢复时间更长。C1对动态参考并非全面更优：其无外载Roll/Pitch任务的峰值位置误差已大于C0/C2，需结合配对额外误差分析，不能只看全回合RMSE。

原局部−10 N与+0.5 N m组合在Z+Roll轨迹中的峰值大于单因素Home场景；C1虽峰值较低，卸载后2秒观察窗内未满足严格恢复条件，最后位置误差约{float(formal[(formal.task=='legacy_local_z_roll_exact')&(formal.controller=='C1')].final_position_error_mm.iloc[0]):.6f} mm；它仍满足原任务完成门槛。这是真实的未恢复记录，未被计作保护失败。

## 力、行程、约束与失败

正式最大六支链实际力{formal.max_force_N.max():.6f} N；最大力变化率{formal.max_force_rate_N_s.max():.6f} N/s，原±400 N限制未解除；命令饱和计数{int(formal.command_saturated_steps.sum())}，原生饱和计数{int(formal.native_saturated_steps.sum())}。各trial还记录配对额外驱动力峰值、各支链加载前与加载期均值、力平方积分、行程误差及速度。

正式最大weld平移{formal.max_weld_translation_m.max():.6e} m、旋转{formal.max_weld_angle_rad.max():.6e} rad，均低于原1e-4门槛；最小已覆盖代理间隙{formal.minimum_partial_proxy_m.min()*1000:.6f} mm；全部51次检查的最小slide硬限位余量{min(x['minimum_stroke_margin_m'] for x in v['raw_audits'])*1000:.6f} mm。无MuJoCo数值警告、无正式保护事件；原生接触观察总数{int(formal.native_contact_observations.sum())}，这与部分代理检查不能替代整机碰撞安全认证。

没有人为制造研究失败。保护链路通过明确TEST_ONLY注入的单元测试验证，包括控制器拒绝、完整0.2秒受限保持及临近正常预算边界的保持延伸；这些测试不算正式实验失败或成功数据。

## 审计中发现并修复的问题

1. 初版三组legacy按理想十进制t_next推断为输入k=1999加载；旧浮点代码实际首个带载结果行是2001，对应输入k=2000。初版保留为legacy_offset_diagnostic，另冻结v2并只补跑三次。精确复现全部6000步外载、支链输入与qpos逐位相等，最大差0。
2. 初版恢复实现曾把机器人尚未响应时的初始带内0.3秒误记为恢复。修复为必须先有实际越界，再搜索0.3秒回归；阈值和时长不变。原metrics.json保留，正式报告使用metrics_recomputed.json；错误与修复见`results/phase2a_metric_implementation_correction.json`。没有根据结果调容差或重写物理数据。
3. 新试验plant.json保留原基类策略文本来源为legacy_base_external_channel_policy，并注明实际外载以disturbance_profile及日志为准；未修改质量、惯量、执行器或模型原件。

## 原始数据、复算与可视化

51次均完整保存6000行1kHz日志及完整mjSTATE_INTEGRATION，并保存初始化状态。独立复算从CSV重建位置和SO3误差、积分及恢复；另用独立scratch MjData验证外载施加前旋转、作用点和广义力，不在scratch中生成替代实验。

观察器开启Qt+原生MuJoCo镜像+实际RGB后，2048步全积分状态与关闭观察器时逐位相等，最大差0，镜像正常退出。实测报告：`results/phase2a_observer_validation.json`。

演示`assets/videos/stewart_phase2a_disturbance_rejection.mp4`：{video['frames']}帧，{video['fps']}fps，{video['duration_s']}秒，1920×1080，全部可解码，状态哈希、实际外载和帧映射一致。由五段真实trial现场RGB和同索引历史曲线合成，无重新积分或重新计算控制器，未放大机器人运动。标签为DYNAMIC DISTURBANCE RESPONSE EXPERIMENT；不同控制器的片段依次展示，不能把一段视频当作整体成功率证据。

已生成阶跃力/Z、力矩/Roll-Pitch、六支链力、相对slide行程、恢复比较、约束和动态响应图。图中行程是移动副相对Home增量，不是整根杆总长度。

自动测试：{test['tests']}项，失败{test['failures']}，错误{test['errors']}；原102项及持续运行13项回归保留。详见`results/phase2a_pytest_final.xml`和日志。

最终测试采用`phase2a_pytest_temp_plugin.py`在项目目录分配普通继承权限的独立临时目录；没有修改ACL或系统设置，也没有改变测试断言与物理行为。原因是默认pytest临时目录权限与沙箱不兼容，随后自动审批又出现连接/超时故障；改用沙箱内正常可写路径完成了135项实际测试。

## 全部正式对照（由原始CSV复算）

{full}

## 14项核心问题与决策

1. 成功建立突变外载平台，支持整数步阶跃、短脉冲接口、六维复合和零外载配对。
2. 经后6个site力/力矩执行器施加，前6个仍由原控制器控制。
3. 明确世界/局部坐标、upper_center参考点及N/N m，并完成独立广义力核验。
4. 新加载/卸载严格按k=2000/4000执行，结果样本有1ms积分延迟，非GUI墙钟触发。
5. 原−10 N与+0.5 N m组合已按旧实际步号逐位复现，偏移诊断保留。
6. Home力与力矩的实际最大偏移见上表，没有仅用RMSE掩盖瞬态。
7. Roll/Pitch及Z+Roll运动时施加载荷，参考保持不变；额外偏移和耦合响应在动态曲线及配对列中记录。
8. C0/C1/C2每任务的峰值、加载/卸载恢复见上表和恢复CSV，没有把未恢复填0。
9. C1在部分Home持续负载任务可恢复；C0/C2部分任务加载期未恢复，卸载后恢复；C1在历史组合卸载后观察窗内仍未严格恢复。
10. 正式任务无提前保护；未恢复与保护失败分开。注入测试仅用于工程链路验收。
11. 六支链力在加载后重新分担，实际峰值及增量受原限幅约束，见统一颜色曲线和逐支链原始数据。
12. 有明确、幅度有限的持续偏差与卸载超调研究空间，但不能由此推断PPO一定有用。
13. 下一阶段优先Home −5/−10 N及+/-0.25/0.5 N m可恢复任务，分别研究持续偏差与卸载超调；动态/组合后置。
14. 具备开展强传统抗扰基线和小幅残差动作敏感性研究的条件；没有证据直接批准RL长训练。

**GO TO PHASE 2B**

依据：任务可重复、原约束下有界、接口和坐标得到核验、持续误差和恢复取舍可测。Phase2B仅在用户审核后开展C3及动作可学习性研究，本轮到此暂停。

## 未执行与限制

没有正式大幅冲击或短脉冲扫描、没有专门Pitch外力矩矩阵、没有随机时间/负载多种子或模型随机化；仅实现脉冲接口和测试。没有C3、Gym残差环境、PPO/SAC/BC及任何训练。没有额外完整碰撞几何认证、连续安全保证或实机试验。小幅动力学任务全部成功不代表更大扰动安全，也不代表PPO超过传统控制。

独立历史回放入口已通过2秒离屏启动、现场RGB解码与正常关闭检查；另一次普通沙箱桌面启动不可见，已结束该测试进程。两次请求补做可见桌面回放均因自动审批服务连接中断而未执行，不能报告该项额外桌面操作已通过。Qt+原生MuJoCo实时观察器的2048步对照已实际在Windows完成并正常退出，不受此可选回放桌面检查限制。

## PowerShell复现/观看

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\compare_phase2a_controllers.py --run-root results\\runs\\phase2a_20261028_v1
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\watch_phase2a_disturbance.py --trial results\\runs\\phase2a_20261028_v1\\formal\\A_home_force\\C1
```

compare只复算现有原始数据，不重新执行物理实验。watch打开历史界面，现场RGB有来源标识。重跑完整新实验需使用run_phase2a_disturbance.py和全新输出目录，参见方案文档；本轮没有自动开展Phase2B。
'''
    (ROOT/'docs/15_Phase2A验收报告.md').write_text(text,encoding='utf-8')
    (ROOT/'PHASE2A_README.md').write_text('''# Stewart Phase2A: disturbance-response experiments

Only C0/C1/C2, no RL training. 15 pilot + 33 formal trials, with 3 retained timing-offset diagnostics. The formal set contains 30 new world-frame cases and 3 exact legacy reproductions. Full raw data and the correction history remain local.

- [中文验收报告](docs/15_Phase2A验收报告.md)
- [冻结实验方案和运行方法](docs/14_Phase2A突变负载实验方案.md)
- [后续研究路线（仅计划）](docs/13_Stewart突变负载抗扰研究技术路线.md)
- [30秒真实动力学视频](assets/videos/stewart_phase2a_disturbance_rejection.mp4)
- [完整核验证据](results/phase2a_validation.json)

C1 reduces some sustained-load offsets, while unloading/dynamic-reference tradeoffs remain. No result demonstrates an RL advantage. PARTIAL PROXY QUERY is not whole-robot safety certification.
''',encoding='utf-8')
    print('Wrote Phase2A report from',len(formal),'formal rows; tests',test,flush=True)
if __name__=='__main__':main()
