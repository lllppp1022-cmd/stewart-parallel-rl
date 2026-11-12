"""Generate final Chinese acceptance only from real recomputed CSV/validation data."""
import _bootstrap
import json,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha

def table(df,columns,headers):
    out='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in df.to_dict('records'):
        values=[]
        for key in columns:
            v=row[key];values.append('—' if pd.isna(v) else f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v))
        out+='| '+' | '.join(values)+' |\n'
    return out

def main():
    root=ROOT/'results/runs/phase2b1_20261028_v1';f=pd.read_csv(ROOT/'results/phase2b1_controller_comparison.csv');d=pd.read_csv(ROOT/'results/phase2b1_development_comparison.csv');v=json.loads((ROOT/'results/phase2b1_validation.json').read_text(encoding='utf-8'));params=v['frozen_parameters'];lat=pd.read_csv(ROOT/'results/phase2b1_observer_latency_details.csv');ob=pd.read_csv(ROOT/'results/phase2b1_observer_validation.csv');online=pd.read_csv(ROOT/'results/phase2b1_online_observer_validation.csv');video=json.loads((ROOT/'assets/videos/stewart_phase2b1_robust_control_comparison.validation.json').read_text(encoding='utf-8'));test=ET.parse(ROOT/'results/phase2b1_pytest_acceptance.xml').getroot()[0].attrib
    v['automatic_tests']=test;v['replay_desktop_smoke']=dict(passed=True,automatic_close_s=2,exitcode=0,log='results/phase2b1_replay_desktop.log');v['video']['physical_joint_motion']=video['physical_joint_motion'];write_json(ROOT/'results/phase2b1_validation.json',v)
    force=f[f.task=='benchmark_force10'].copy();force['order']=force.controller.map({'C0':0,'C1':1,'C2':2,'C1-L':3,'C3':4});force=force.sort_values('order');c3=f[f.controller=='C3'];c1l=f[f.controller=='C1-L'];first=json.loads((root/'leak_selection.json').read_text(encoding='utf-8'));selected_latency=lat[lat.bandwidth_rad_s==params['bandwidth_rad_s']]
    home_table=table(force,['controller','load_peak_position_mm','unload_peak_position_mm','unload_reverse_z_mm','load_position_iae_mm_s','unload_position_iae_mm_s','load_recovery_s','load_recovery_status','unload_recovery_s','unload_recovery_status'],['方法','加载峰值mm','卸载峰值mm','卸载反向Z峰值mm','加载IAE mm s','卸载IAE mm s','加载恢复s','加载状态','卸载恢复s','卸载状态'])
    leak_table=table(d[(d.controller=='C1-L')&(d.task!='dev_zero')],['task','leak_per_s','load_position_iae_mm_s','unload_position_iae_mm_s','objective'],['开发任务','lambda s^-1','加载位置IAE','卸载位置IAE','冻结综合目标'])
    full=table(f,['task','controller','success','load_peak_position_mm','load_peak_orientation_deg','unload_peak_position_mm','load_position_iae_mm_s','unload_position_iae_mm_s'],['任务','方法','完成','加载峰值mm','加载峰值SO3 deg','卸载峰值mm','加载位置IAE','卸载位置IAE'])
    obs_table=table(online,['task','split','force_rmse_N','roll_rmse_Nm','pitch_rmse_Nm','force_steady_error_N','roll_steady_error_Nm'],['最终任务','划分','Fz RMSE N','Mx RMSE N m','My RMSE N m','Fz稳态误差N','Mx稳态误差N m'])
    latency_table=table(selected_latency,['task','axis','rise10_delay_s','rise90_delay_s','unload_decay10_delay_s'],['开发任务','分量','10%响应s','90%响应s','卸载衰减至10% s'])
    text=f'''# Phase2B-1验收报告：积分泄放与局部动量抗扰

日期：2026-10-28。工程`__PROJECT_ROOT__`。

## 结论与范围

独立C1-L与C3已实现并实际验证。原C0/C1/C2、模型、力限、安全门槛及Phase0～2A实验结果未修改。C1-L体现加载补偿和卸载残留的取舍；C3在本轮理想测量、名义物理模型及局部范围内改善了两阶段误差，但力变化率更高，反向Z超调也并非对每个基线占优。不能将其称为真实硬件外力传感器、全空间鲁棒控制或正式稳定性保证。

本轮没有Gymnasium封装、PPO/SAC/BC训练、RL动作敏感性实验或扩大工作区。最终建议仅允许审核后进行有界Phase2B-2可行性检查，不表示需要或批准RL训练。

## 实际预算、冻结与数据隔离

研究运行76次×6000步={v['research_physical_steps']}个物理步：6次只安全预检（含新Pitch力矩两档前置检查）、15次C1-L开发、3次C0开发数据、1次C1全回合等价基准、6次C3补偿开发、45次最终评价。另做当前实现6000步复现及Qt/原生显示/录像开关2×2048步对照，不当作独立科学样本。自动测试的短仿真另计。观测器9组带宽评价是已有真实开发轨迹的因果回放，没有新积分或伪造运动。

主目录：`results/runs/phase2b1_20261028_v1`。最终25次为已查看的Phase2A历史基准，20次为4个未用于整定的安全预检后冻结测试条件；不是完全操作者盲测或独立随机种子统计。所有控制器共享相同初态、参考、外载、dt、实际模型、原生限制和保护语义。

`protocol_before_development.json`、`independent_test_frozen.json`、`controller_parameters_frozen.json`、`final_protocol_frozen.json`和每trial中的哈希形成冻结证据。独立测试只允许预检安全与可执行性的通过/失败信号进入是否执行的门禁，性能不进入参数选择。没有读最终结果后追加整定。

{v['historical_files_checked']}个历史文件SHA全部不变。15组C0/C1/C2历史场景的全部6000步完整积分状态与Phase2A逐位一致；C1-L lambda=0与C1的6000步亦逐位一致。独立实际重跑当前带额外输入/限速可行性检查的C3代码，其6000步完整状态仍与保存的最终轨迹逐位一致，证明开发期这些检查没有改变本轮有效工况的动作。

## 冻结控制参数和方法

C1-L选择lambda={params['leak_per_s']} s^-1，仍使用P=5000 N/m、D=200 N s/m、Ki=3000 N/(m s)、积分限幅±0.005 m s。非零lambda每步先泄放，再执行冻结C1的误差积分/限幅/条件抗饱和；推向饱和的误差积分仍被拒绝，泄放继续。lambda=0直接调用原C1代码。

C3选择局部等效广义动量残差，带宽{params['bandwidth_rad_s']} rad/s、补偿增益{params['compensation_gain']}；观测与控制1kHz，Jacobian仍100Hz，非30Hz观测。估计力上限±20N，力矩±0.8N m；附加支链力±5N、变化率20N/s，总力仍原生±400N。当前测量与前一步实际下发的总指令用于更新，当前补偿不读取外载真值、加载时刻、实际mass/bias/constraint force或下一状态。

6维Home局部M0/D0/K0由独立名义模型和36维闭链约束切空间推导；T满足E T=0、J_site T=I，质量矩阵正定。C3保留完整6维模型耦合，但仅补偿Z/Roll/Pitch，实际限幅/限速后的完整广义力及非目标轴误差均记录。详细推导、可测性假设和UNKNOWN项见文档16/17；不把姿态误差乘增益伪装成完整外力观测。

## 开发整定及未改善结果

候选lambda=0/0.5/1/2/5、观测带宽5/15/30 rad/s、补偿增益0.5/1，均来自开发前冻结范围。没有调整Ki、增益P/D、安全标准或最终任务。

{leak_table}

Home −5N时，所有非零泄放都使加载位置积分误差高于lambda=0；卸载误差则显著减少。Roll +0.25N m开发工况中，lambda=0.5/1/2的加载积分误差较lambda=0减少，lambda=5反而更差。卸载最小值未必与加载最小值对应。lambda=1按预定双阶段、位置/姿态归一化综合目标选中，不能宣称它每个指标都最优。

三档带宽在三个C0开发日志上的门禁结果共{len(ob)}行，全部保留；选中带宽的延迟如下（真实外载仅离线评价）：

{latency_table}

## Home −10N严格配对

{home_table}

恢复沿用位置<=0.25mm、SO3<=0.02度且连续0.3秒，并必须先发生实际越界。NO EXCURSION表示该窗口没有越界，恢复时间不适用；没有把它填成0。加载/卸载窗口分别计算峰值，区别于Phase2A报告中“加载后直到结束”的总峰值。

C1-L的加载峰值与C1非常接近，但持续加载IAE更大；卸载峰值及IAE更低，这是泄放降低积分残留的实测取舍。C3将加载与卸载的三维误差范数峰值和积分误差降低，且在此例中未越出严格容差；但它的卸载反向Z峰值约0.147365mm，大于C1-L的0.132946mm，也大于C0/C2约0.014mm。因此不能声称C3所有卸载指标都最优。不能将NO EXCURSION解读为无误差或瞬时恢复。

## C3观测真实性与泛化边界

名义模型验证使用旧无外载Home/Roll/Pitch/Z+Roll真实日志，不拟合扰动标签，已验证实际约±0.402度及Z约2.5mm局部范围。约束切空间误差约3.3e-16，名义平衡广义力误差约2.6e-6；Home等效Fz偏置约8e-5N。软闭链模式、变惯性和高阶速度耦合仍是模型残差，不是被完整精确建模。

以下最终估计误差是“局部等效扰动”和真实外载的离线比较；不能据此称为真实力测量。全过程RMSE包含加载/卸载滤波延迟，稳态误差单独给出：

{obs_table}

当前理想反馈下，方向、稳态与衰减均通过开发门禁；模型系数仅来自名义模型。若真实质量/增益或传感器发生偏差，它们会进入等效残差；本轮没有验证这些不确定性、噪声和延迟下的闭环表现。冻结C0仍依赖全链关节反馈，被动关节传感器和平台实际6D测量工程均UNKNOWN；不能默认具备实机可观测性。

## 安全、控制代价和剩余误差

45/45最终评价按完整6秒、无保护、最后0.5秒位置<0.75mm和SO3<0.1度完成，未出现新的实际保护失败或MuJoCo数值警告。C3九个最终工况的加载和卸载均为NO EXCURSION，最高加载位置误差仍有{c3.load_peak_position_mm.max():.6f}mm、最高卸载位置误差{c3.unload_peak_position_mm.max():.6f}mm，加载最高姿态误差{c3.load_peak_orientation_deg.max():.6f}度，故并非完美补偿。

全部最终试验最大实际支链力{f.max_force_N.max():.6f}N，最小原生力余量{f.actuator_force_margin_N.min():.6f}N。C3附加力实际最大{c3.max_compensation_N.max():.6f}N，实际最大附加力变化率{c3.max_compensation_rate_N_s.max():.6f}N/s；这是附加力约束，不是总控制力变化率上限。

Home −10N的总力最大变化率：C1-L {force[force.controller=='C1-L'].max_force_rate_N_s.iloc[0]:.6f} N/s，C3 {force[force.controller=='C3'].max_force_rate_N_s.iloc[0]:.6f} N/s；C3更快抗扰伴随更快力变化。实际力平方积分和非目标XY/Yaw误差保存在完整CSV，没有以更低位姿误差掩盖控制代价。

最终评价最大weld平移{f.max_weld_translation_m.max():.6e}m、旋转{f.max_weld_angle_rad.max():.6e}rad，低于原1e-4；已覆盖代理最小间隙{f.minimum_partial_proxy_m.min()*1000:.6f}mm。代理仍是原50Hz PARTIAL PROXY QUERY，其余状态检查为1kHz；不能据此声称全机连续碰撞安全或实机认证。

保护通过TEST_ONLY测试验证：控制器拒绝、非法Jacobian/模型、无效观测、原生余量与附加变化率冲突、0.2秒受限保持。没有故意削弱控制器或制造“研究失败”以显示新方法优越。

## 自动验收、失败记录与可视化

最终{test['tests']}项实际测试，失败{test['failures']}、错误{test['errors']}、跳过{test['skipped']}；原135项回归保留。记录`results/phase2b1_pytest_acceptance.xml/.log`。首轮v1有4项新增测试失败、148项通过；失败源于测试漏掉Home IK参考与实际行程之间约1e-7m的真实小误差，导致积分期望值偏差约1.4e-10 m s。已把真实误差项加入公式期望，并使用更严格容差；控制器、物理模型和保护阈值没有因此修改。v1失败XML/日志及修正轮记录均保留。

Qt+原生MuJoCo镜像+真实RGB开启/关闭对照各2048步，全部积分状态逐位一致，误差0；原生显示退出码0，未额外推进物理或改变输入。独立历史回放在Windows打开2秒后正常关闭，见`results/phase2b1_replay_desktop.log`。

视频`assets/videos/stewart_phase2b1_robust_control_comparison.mp4`：{video['frames']}帧、25fps、{video['duration_s']}秒、1920×1080；四段C0/C1/C1-L/C3的相同Home −10N实际现场RGB，合成时只读取原日志和帧，不重新积分控制器。全部帧可解码、qpos哈希/外载/时间匹配，机器人关节位移按完整状态独立核对，没有放大机械运动。标注CLASSICAL DISTURBANCE REJECTION CONTROL COMPARISON，不是训练视频。所有六支链颜色沿用既有统一映射。

六类响应图及双阶段指标图已生成：Home阶跃、Roll力矩/非目标轴、C1与C1-L积分、C3局部等效扰动、六支链基础/附加/实际力、加载—卸载比较。当前代码完整物理重现额外6秒验证已计入工程成本，不纳入调参或最终成功率。

## 全部最终结果（独立复算）

{full}

## 20项核心问题逐项回答

1. C1按I+e dt积分、先积分限幅，再依据原始力和误差方向条件拒绝推向饱和的更新，最后原生限力。
2. C1-L只增加可控积分泄放，饱和时仍泄放但不继续错误累积，P/D/Ki不改。
3. lambda=0完整6000步与C1逐位相同。
4. Home −5N的持续加载积分误差未因非零泄放改善；Roll +0.25N m中0.5/1/2改善，5退化，见开发表。
5. 卸载改善与加载改善不总一致；Home −5N的2 s^-1卸载IAE最小，但综合目标选择1 s^-1。
6. 存在加载补偿不足，泄放减少稳态积分增益，不能宣称全面优于C1。
7. C3采用Home局部6轴广义动量残差的一阶有界观测，不是姿态误差增益或完整全空间DOB。
8. 新观测模块仅用平台位姿/速度及前一步已知下发命令、名义Jacobian和名义系数；原C0反馈接口保留。
9. 控制器完全不读取真实外载/事件时刻；真值仅在独立评价代码与展示中使用。
10. 模型依据名义MJCF惯性、被动阻尼/重力及闭链切空间，局部验证范围与未建模项见文档17。
11. 开发门禁方向与稳态通过；30 rad/s下约4ms达到10%、78ms达到90%并在卸载约78ms衰减至10%，最终误差见观测表。
12. 将3个补偿分量嵌入完整世界6维广义力，解J_l^T F_comp=W_comp，再限幅/限速并记录实际广义力。
13. 五种控制器同条件45次完整结果见表，9个条件各5种方法。
14. C1-L降低卸载残留，但部分加载IAE更差；C3本轮两阶段三维误差范数和IAE较低，但Home −10N卸载反向Z峰值高于C1-L及C0/C2，已单独列出，不能据全程RMSE掩盖该取舍。
15. 有控制代价及方向指标取舍：C3力变化率更高，部分反向Z超调更大；非目标轴与完整力努力均记录，未声明全面占优。
16. 参数由三个开发任务选择，四个最终测试条件仅安全预检后冻结，最终45次前已冻结控制参数与矩阵；不是完全盲测。
17. 没有新的实际保护失败，所有未改善开发候选和工程测试失败日志保留。
18. C3仍有最高约{c3.load_peak_position_mm.max():.3f}mm瞬态误差和较快的力变化，但当前所有九个条件均满足严格带内要求；不能仅因非零误差推断有学习必要。
19. 可以进入有界、只测可控性与代价取舍的Phase2B-2，验证是否存在超出强C3基线的额外收益；若没有，应停止引入学习方法。C3的±5N附加力配置不是任意RL动作幅值已获安全认证的证据。
20. 仍禁止直接开展PPO/SAC/BC、Gym环境或长训练；下一阶段需用户另行审核授权。

**GO TO PHASE 2B-2**

此GO只表示有实际约束、因果接口和可测剩余瞬态/控制代价可供有限动作敏感性研究，不表示RL必要或有效。首先保留强C3对照；若小幅附加动作不能在不恶化安全和控制代价的条件下改善，不应继续搭建训练环境。本轮到此暂停。

## 未完成与限制

未开展模型质量/惯量/增益失配、实际测量噪声/延迟、实机传感器可用性、全空间运动、更多复杂随机外载或严格闭环稳定性证明。未实施RL残差动作试验、Gymnasium和任何网络训练。带宽选择处于本轮候选上边界30 rad/s，不能把有限无噪声整定称作最终鲁棒最优参数。

## 实际验证的PowerShell入口

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\compare_phase2b1_controllers.py --run-root results\\runs\\phase2b1_20261028_v1
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\watch_phase2b1.py --trial results\\runs\\phase2b1_20261028_v1\\final\\benchmark_force10\\C3
```

观看入口的Windows实测使用相同命令另加`--validate-seconds 2`自动关闭，正常退出。compare只复算归档数据和图，不重新执行试验。完整研究重跑入口tune_phase2b1_controllers.py必须提供新的run目录，预算最多6次预检+25次开发/等价+45次最终，不覆盖已有目录；本轮已经真实运行，不为收尾重复全部长矩阵。
'''
    (ROOT/'docs/18_Phase2B1验收报告.md').write_text(text,encoding='utf-8')
    (ROOT/'PHASE2B1_README.md').write_text('''# Stewart Phase2B-1 — Classical disturbance rejection

C1-L leaky PID and a Home-local nominal-model momentum residual observer on frozen C0. No reinforcement learning, no Gym environment or RL action-sensitivity experiments.

- [设计和接口审计](docs/16_Phase2B1传统抗扰控制器设计.md)
- [名义模型与观测器依据](docs/17_Phase2B1扰动观测器模型与参数依据.md)
- [完整验收报告](docs/18_Phase2B1验收报告.md)
- [24秒真实对比视频](assets/videos/stewart_phase2b1_robust_control_comparison.mp4)
- [机器可读验收](results/phase2b1_validation.json)

45 paired final trials completed. C1-L reduces unloading residuals with a sustained-load tradeoff. C3 improves the tested ideal-sensing nominal-model response, with higher force slew. No claim of global robustness, true-force sensing, hardware safety or an RL benefit.
''',encoding='utf-8')
    print('Report written from',len(f),'final rows; tests',test,flush=True)
if __name__=='__main__':main()
