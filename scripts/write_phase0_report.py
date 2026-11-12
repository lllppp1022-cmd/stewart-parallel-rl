"""Generate documentation from actual artifacts, including failed gates."""
import _bootstrap
import json,datetime,subprocess,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from src.model import ROOT,write_json,sha,UPSTREAM

def main():
    model=json.loads((ROOT/'results/phase0_model_inspection.json').read_text(encoding='utf-8'));env=json.loads((ROOT/'results/environment.json').read_text(encoding='utf-8'));video=json.loads((ROOT/'results/phase0_video_validation.json').read_text(encoding='utf-8'))
    dyn=pd.read_csv(ROOT/'results/phase0_dynamics_check.csv');kin=pd.read_csv(ROOT/'results/phase0_kinematics_validation.csv');dt=pd.read_csv(ROOT/'results/phase0_timestep_sensitivity.csv');anchors=pd.read_csv(ROOT/'results/phase0_anchor_audit.csv')
    tests=ET.parse(ROOT/'results/phase0_pytest.xml').getroot().find('testsuite');assert tests.get('failures')=='0'
    gui=[]
    for folder in sorted((ROOT/'results/runs').glob('*_demo_*')):
        path=folder/'exit.json'
        if path.exists():gui.append(dict(run=str(folder.relative_to(ROOT)),**json.loads(path.read_text(encoding='utf-8'))))
    completed_close=[x for x in gui if x['mode']=='dynamic' and x['reason']=='viewer_closed'];assert completed_close
    gui_evidence=dict(actual_native_windows_observed=True,static_mode_observed=True,dynamic_mode_observed=True,manual_close_final_pass=True,final_success=completed_close[-1],earlier_native_failures=[dict(log='results/setup/gui_dynamic_1ms.log',exit_code=-1073741819),dict(log='results/setup/gui_dynamic_close_retry.log',exit_code=-1073741819),dict(log='results/setup/gui_safe_close.log',exit_code=-1073741819,stack_location='demo_common.py:42 with viewer.lock (at that revision)')],diagnosis='Observed stack identifies viewer.lock on native X-close; removed per-step viewer mutex acquisition, kept owner step + internally synchronized sync following official example. One real manual-close validation passed; not an all-platform guarantee.',completed_runs=gui)
    write_json(ROOT/'results/phase0_gui_validation.json',gui_evidence)
    preface='''# Stewart并联机器人：Phase 0仿真环境与模型审计

独立Python/MuJoCo工程。已完成模型获取、真实闭链/执行器审计、独立链式运动学、小幅真实动力学、原生Windows窗口和录像验证；**没有PPO/SAC/BC训练**。

[Phase0验收报告](docs/03_Phase0验收报告.md) · [几何说明](docs/01_Stewart机构几何与模型说明.md) · [闭链/执行器审计](docs/02_MuJoCo闭链约束和执行器审计.md)

[45秒真实动力学审计视频](assets/videos/phase0_simulation.mp4)

![六支链目标/实际行程与力](assets/plots/phase0_leg_lengths_forces.png)

视频是原始2ms基线的现场RGB，含自由下沉、Home、小幅升降、倾转和力脉冲；该基线没有通过预设的严格内部weld角误差门槛，未包装为全部通过。后续1ms/0.5ms敏感性试验各7组通过相同门槛，见报告。

## 当前结论

- Windows/Python3.11.17/MuJoCo3.17.0直接加载上游原模型，nq=43、nv=42、nu=12、45个body、6个weld。
- 只有前6个是支链力执行器（±400N）；其余是平台扰动力/力矩通道，所有正式控制中恒为0。
- README铰点表与MJCF有差异；上端三转轴存在约0.25mm不共点偏置。使用真实关节链IK与隐式Jacobian，未把理想杆长当作slide坐标。
- 真实实现Z±5mm、Roll/Pitch±0.5°。建议后续小范围控制显式使用1ms步长；原2ms失败记录保留。
- 上游机器人碰撞过滤全部关闭，未做工作空间、支链碰撞或实机安全认证；不得扩展为安全运动保证。
- 实际GPU是RTX3060 12GB，不是RTX4070。未安装PyTorch或Stable-Baselines3。

## 本机直接运行（PowerShell）

```powershell
Set-Location __PROJECT_ROOT__
conda activate stewart_rl
python scripts/view_stewart.py --viewer --seconds 30
python scripts/demo_home.py --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis z --sign 1 --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis roll --sign 1 --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis pitch --sign 1 --viewer --seconds 10 --timestep 0.001
python -m pytest -q
```

若终端没有conda，使用 `& 'D:\\miniconda\\envs\\stewart_rl\\python.exe'` 替代上面的python。VS Code选择同一解释器。负方向用`--sign -1`。控制是在前1秒保持、2秒平滑移动、随后保持，5mm/0.5°幅度很小，不应为看清动作自行放大。

静态入口标记Kinematic Visualization，物理时间保持0；动态入口标记Dynamic Actuator Simulation，使用原力执行器和mj_step。每次演示输出到唯一results/runs目录。原生窗口确已实际打开；Windows关窗期间曾触发访问冲突，移除逐步viewer.lock并改为官方owner-step/sync模式后实际手动关闭复核通过，早期失败栈保留。

自动关闭预算、窗口手动关闭都应查看该次`exit.json`；没有exit文件不能称为成功。实时日志另写physics_journal.csv。不要在Viewer中修改控制器、模型选项、拖拽施力或按Reset后仍把轨迹当作标准实验。

## 环境与重算

本次真实新建Conda环境，没有复用旧项目的sys.path回退。安装记录在results/setup/，精确版本见requirements-lock.txt。

```powershell
# 仅在新的电脑/尚无同名环境时创建：
conda create -n stewart_rl python=3.11 pip -y --no-default-packages
conda activate stewart_rl
python -m pip install -r requirements-lock.txt
```

模型目录随项目保留原仓库和资产；来源commit固定。`python scripts/inspect_model.py`可重新做静态/运动学审计；`python scripts/validate_artifacts.py`对已保存视频和状态做验收。`run_phase0.py`与`check_timestep.py`是本阶段正式实验入口，不要不加区分重复运行覆盖摘要；正式入口遇到已存在主视频会拒绝，历史独立run始终保留。

未来Gymnasium环境、强化学习网络、全工作空间/碰撞控制均尚未实现。本阶段完成后暂停。

## 来源与许可

模型作者Duc Cuong Vu、Viet Khanh Nguyen；[上游仓库](https://github.com/duccuongvu/stewart_platform_mujoco)，固定commit `b0195ba9f40f6e0876220be1712eb47f5fc24025`。保留完整MIT LICENSE（Copyright 2026 vr-cuongvu）、README及资产；没有修改上游受跟踪文件。此模型的许可不自动指定本项目其他材料的许可。MuJoCo/Python依赖仍遵守各自许可证。
'''
    (ROOT/'README.md').write_text(preface,encoding='utf-8')
    doc1='''# Stewart机构几何与模型说明

## 几何依据与机构结构

使用third_party/stewart_platform_mujoco原始MJCF，不使用盾构六推进缸坐标。README中的B/T各6点表作为来源审计数据保留，但与实际MJCF的腿编号和坐标不同，不能直接用于精确执行。

固定base、6个支链和自由upper平台。每腿：固定cardan1_i安装体 → cardan2_i转轴 → legbot_i转轴 → legup_i移动副 → cardan1_(i+6)转轴 → cardan2_(i+6)转轴 → cardan1_(i+12)转轴 → site weld连接平台。即每腿5R+1P，近似U-P-S拓扑，而不是六根无质量理想线段。upper自由关节提供7个qpos/6个速度坐标。

下表的B为底座两转轴的最小二乘交点；T为上端三个转轴的最小二乘共点代理在平台坐标系的值。**T不是精确球铰中心**，上端轴偏置约0.25mm；表中明确列出残差。真实精确参考采用完整关节链和weld attachment，而不把这个代理当作真值。

| 腿 | 点/坐标系 | x(m) | y(m) | z(m) | 轴线不共点残差(m) |
|---|---|---:|---:|---:|---:|
'''
    for row in anchors.to_dict('records'):doc1+=f"| {row['leg']} | {row['frame']} | {row['x_m']:.9f} | {row['y_m']:.9f} | {row['z_m']:.9f} | {row['axis_common_center_max_distance_m']:.9g} |\n"
    doc1+='''
全部MJCF安装原点、平台weld site局部坐标、原README B/T及每腿行程另见results/phase0_model_inspection.json。安装体原点、铰轴中心、焊接site、缸的相对行程是不同的几何量。

## 独立运动学参考

src/kinematics/stewart_geometry.py只从MjModel提取静态body/joint/site变换；forward_leg使用NumPy/SciPy进行齐次变换乘积，不调用mj_forward。旋转关节使用T(anchor)R(axis,q)T(-anchor)，移动副沿局部Z平移。

输入平台世界位置p(3)及旋转矩阵R(3×3)。inverse_kinematics逐腿求六个链坐标，使末端site位置和方向同时闭合到平台site；slide取第3个坐标，严格限制在[-0.1,0.35]m。输出六个相对行程、闭合残差及余量。控制标签不是总杆长；attachment_chord_lengths只是指定安装点/site的弦长，不能冒充执行器读数。

关节链几何Jacobian G由轴方向和末端力臂构成；通过G qdot = [v + omega×(R t_i), omega]求解并取slide速度行，得到6×6平台twist→相对行程速度的Jacobian。旋转扰动采用世界轴左乘，有限差分与解析推导使用同一约定。还提供弦长Jacobian，但不用于实际力控制行程替代。

## 实际验证与边界

'''
    doc1+=f"7个构型（Home、Z正负、Roll/Pitch正向、seed42两个小扰动），每个用1e-4/1e-5/1e-6三个差分步长，共{len(kin)}行记录。独立FK与MuJoCo最大位置误差{kin.fk_mujoco_position_error_m.max():.6g}m，执行器length与slide坐标最大差{kin.actuator_length_stroke_max_error_m.max():.6g}m，Jacobian最大绝对差{kin.jacobian_max_abs_error.max():.6g}。数值误差不代表CAD几何已获实机标定。\n\n"
    doc1+=f"归一化Jacobian条件数范围{kin.normalized_condition.min():.6f}～{kin.normalized_condition.max():.6f}，旋转列使用0.35m特征尺度，当前样本无近奇异点。10m越界目标被拒绝；NaN、维度错误及非法旋转输入测试通过。没有证明整个行程盒可达，没有完成全工作空间奇异面或支链碰撞审计。\n"
    (ROOT/'docs/01_Stewart机构几何与模型说明.md').write_text(doc1,encoding='utf-8')
    doc2=f'''# MuJoCo闭链约束和执行器审计

## 模型与许可

上游commit `{model['source_commit']}`，MIT，作者声明及所有STL原样保留。模型不提供Python控制示例，本项目的审计/演示入口为新增代码。官方MJCF加载器直接加载scene.xml及相对路径assets，无模型转换。

- nq={model['nq']}、nv={model['nv']}、nu={model['nu']}、nbody={model['nbody']}（含world）、njnt={model['njnt']}、neq={model['neq']}。
- 30个hinge、6个slide、1个free；6个site-based weld，各约束相对6自由度，Home局部为6维受约束运动。
- 初始最大连接平移间隙{max(model['initial_site_gap_m']):.9g}m；efc_pos最大混合残差{model['initial_max_mixed_equality_residual']:.9g}，其中平移和旋转不能统称为米。
- 平台质量{model['upper_mass_kg']}kg，主惯量{model['upper_principal_inertia_kg_m2']}kg·m²；由网格和质量编译得到，不代表实物标定。
- 初始平台位置{model['initial_position_m']}m，姿态wxyz={model['initial_quat_wxyz']}。

## 执行器真实语义

前6个leg1_ctrl～leg6_ctrl以general joint transmission作用于joint_lu_1～6，gear=1、固定gain=1、无bias，控制输入和输出为力（N），ctrlrange及forcerange均[-400,400]。它们不是位置伺服，不能把期望米数直接写入ctrl。

后6个upper_f_x/y/z、upper_t_x/y/z经upper_center施加外部扰动，范围[-500,500]（力N或力矩N·m）。本阶段所有正式和演示试验均保持这些通道为0，未借用它们直接托举/倾转平台。传感器actuatorpos返回slide相对行程，不是约1m的支链总长。

## 最小动力学控制检查

独立scratch MjData按IK参考设置qpos，仅用于静态平衡查询。使用[B, J_eq^T][u,lambda]=qfrc_bias-qfrc_passive最小二乘计算6个静态平衡力，再以参考行程误差和速度误差作PD修正，Kp=5000N/m、Kd=200N·s/m，原±400N限幅保持。

Live MjData从原生qpos0初始化后仅通过ctrl和mj_step推进；scratch从不冒充动态对象。1秒保持、2秒五次缩放、2秒保持；幅度Z±5mm，Roll/Pitch±0.5°。初始/中点/终点先检查链闭合和行程。端点之间使用小幅行程与静态力插值，不宣称全路径精确笛卡尔跟踪或大范围安全。

## 步长、约束与碰撞限制

原XML使用implicitfast、默认2ms，weld solref=[0.0001,1]、强solimp。默认refsafe会将正格式约束时间常数限制到至少2×物理步长；因此1ms/0.5ms比较也改变了有效约束时间尺度，不能当成保持连续模型刚度不变的纯积分收敛证明。

原2ms控制7组的内部site方向间隙约2.23e-4rad，超过事前1e-4rad门槛；没有降低门槛。1ms/0.5ms各7组通过同一门槛，原失败CSV保留。实际平移残差亚微米，不能据此忽略方向约束。

**机器人所有网格contype=0、conaffinity=0。** 地面可见不等于机器人与地面有接触检测，自碰撞/腿碰撞也未启用。日志ncon=0不能作为无碰撞安全证明。该模型适用于本阶段小范围闭链动力学检查；更大运动、负载/关节真实限位、碰撞几何和实物模型仍需后续核验。

## Windows Viewer

实际使用原生launch_passive。动态过程由唯一Python物理owner写ctrl/mj_step，sync内部同步；对cam设置才短暂获取viewer.lock。早期逐步获取viewer.lock在手动X关窗时引发0xC0000005，崩溃栈保留；移除该竞争后实际手动关窗复核退出0并产生exit.json。增加journal落盘，缺失exit.json的进程不能标为完成。一次复核通过不等于所有显卡/版本无此问题。

参考：[MuJoCo XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、[官方passive viewer示例](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。
'''
    (ROOT/'docs/02_MuJoCo闭链约束和执行器审计.md').write_text(doc2,encoding='utf-8')
    report=f'''# Phase 0验收报告

日期：{datetime.datetime.now().astimezone().isoformat()}。项目独立位于__PROJECT_ROOT__。本阶段仅模型、几何、有限动力学和Windows GUI审计，不训练PPO/SAC/BC，不进入Phase1。

## 实际环境与来源

Python {env['python']}；解释器{env['executable']}。Windows版本报告：{env['platform']}（CIM查询被拒绝，未据此臆测完整OS名称）。实际GPU：{env['gpu']}。

依赖：{', '.join(k+'='+v for k,v in env['dependencies'].items())}。独立Conda环境stewart_rl已真实创建并联网安装；无旧工程学习依赖回退，Torch/SB3均未安装。完整freeze、创建和安装日志保留。未对Reacher、Panda、盾构研究工程执行编辑、覆盖或训练；没有声称对全部旧工程做全量逐字节审计。

上游MIT仓库commit {model['source_commit']}，原模型/资产哈希见phase0_model_inspection.json；git status确认上游受跟踪文件未改。渲染缓冲仅在内存中扩大至960×720，没有改MJCF物理模型。

## 正式原2ms基线：保留全部9组

协议预先保存在configs/phase0_protocol.json，各5秒，1秒保持/2秒运动/2秒保持；7组目标跟踪另加zero与pulse诊断，共22500个实际物理步。7组小幅动作的严格组合门槛含全程weld角间隙<1e-4rad；zero/pulse不算到达成功率。

| 场景 | 步数 | 终点位置误差mm | 终点姿态误差deg | 最大weld角rad | 数值稳定 | 严格小幅门槛 |
|---|---:|---:|---:|---:|---|---|
'''
    for row in dyn.to_dict('records'):
        gate='N/A' if pd.isna(row['stable_small_motion']) else ('PASS' if row['stable_small_motion'] else 'FAIL')
        report+=f"| {row['scenario']} | {row['steps']} | {row['final_position_error_mm']:.6f} | {row['final_orientation_error_deg']:.7f} | {row['max_weld_angle_rad']:.9g} | {'PASS' if row['numerical_stability'] else 'FAIL'} | {gate} |\n"
    report+='''
zero控制下平台下沉约104.8mm并到达软限位附近，数值有限不等于Home保持成功。其余真实力控动作都实现小幅位姿响应，既没有live qpos传送，也没有使用平台外力通道。7组严格组合判定失败的原因是内部weld角间隙，而不是位姿目标完全未到达；失败没有隐藏。

## 有界步长敏感性复核

独立协议在敏感性run开始前冻结：1ms/0.5ms各相同7组、每组5秒、阈值不变，额外最大105000步。未重新选种子或移动几何，原2ms结果不覆盖。

| dt(ms) | 通过/组数 | 最大weld角rad | 最大终点位置误差mm | 最大执行器力N |
|---|---:|---:|---:|---:|
'''
    for timestep,group in dt.groupby('timestep_s',sort=False):report+=f"| {timestep*1000:g} | {int(group.passed_unchanged_criteria.sum())}/{len(group)} | {group.max_weld_angle_rad.max():.9g} | {group.final_position_error_mm.max():.6f} | {group.max_force_N.max():.4f} |\n"
    report+=f'''
正式基线+数值敏感性共127500次物理积分。开发试跑、自动测试、CLI/GUI另外记录，不把这些重复执行当作独立场景统计成功率。两个步长通过仅说明这些有限任务下满足冻结检查；refsafe有效约束时间尺度随dt变化，不能宣称一般连续时间稳定性证明。

## 几何与自动验收

7个小范围构型×3个差分步长={len(kin)}行；独立FK与MuJoCo最大位置差{kin.fk_mujoco_position_error_m.max():.6g}m，Jacobian最大绝对差{kin.jacobian_max_abs_error.max():.6g}，原生actuator_length与预测slide坐标一致。10m不可达目标保守拒绝。未找到/未声称验证全工作空间奇异面。

最终pytest：**{tests.get('tests')} passed，failures={tests.get('failures')}，errors={tests.get('errors')}**。覆盖加载、几何、真实链IK、FD、行程、无效输入、近奇异诊断、闭链、真实力响应、零控制下沉、1ms保持、独立scratch、日志/视频状态哈希、保留2ms失败、步长复核与环境隔离。首轮23项通过但pytest缓存目录创建被Windows拒绝；该记录保留，随后关闭非必需缓存插件，最终25项正常通过，未改ACL/删除失败目录。

## Windows、视频与故障记录

原生静态/动态Viewer已真实打开并观察。静态窗口点击X后以viewer_closed归档；2ms动态90秒预算退出成功。1ms动态手动关闭早期3次出现0xC0000005，最后一次栈定位viewer.lock。按官方owner-step/sync循环移除逐物理步外层viewer mutex后，实际关窗复核退出0：{completed_close[-1]['run']}，物理步{completed_close[-1]['physical_steps']}，终止类型viewer_closed。未把此前崩溃说成通过。

视频：{video['video']}，{video['duration_s']}秒、{video['frames']}帧、{video['fps']}fps、960×720。全部{video['state_log_frame_matches']}帧的状态哈希与物理NPZ逐帧一致，完整解码通过；视频像素来自执行期间live MjData，没有事后用qpos重放冒充现场。RGB文件有损压缩，不声称RGB逐字节等于渲染源；哈希核对针对当时的物理状态。

为严格区分零输入和执行器控制，主视频前50帧字幕已更正为Free Dynamics / Zero Ctrl，其余段落仍为Dynamic Actuator Simulation；保留原始现场捕获phase0_simulation_original_capture.mp4及修改前后SHA，见phase0_caption_provenance.json。字幕后处理没有调用环境、网络或重新生成状态轨迹。

另保留首次960宽度超出默认640帧缓冲的启动失败；只调整内存中的渲染缓冲后重跑。上游XML没有因此改写。GUI初期故障和pytest缓存权限故障见results/setup；没有删除失败记录。截图为现场离屏渲染的代表帧，原生窗口另经computer-use实际观察，不混淆为原生桌面录像。

## 命令与实际验证范围

````powershell
conda activate stewart_rl
Set-Location __PROJECT_ROOT__
python scripts/view_stewart.py --viewer --seconds 30
python scripts/demo_home.py --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis z --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis roll --viewer --seconds 10 --timestep 0.001
python scripts/demo_small_motion.py --axis pitch --viewer --seconds 10 --timestep 0.001
python -m pytest -q
````

实际GUI试验使用90秒有界预算并手动关闭，静态试验用180秒预算提前关闭；上面10/30秒是相同入口的短预算用法，不冒充每条参数组合都人工点过。Home/Z/Roll/Pitch四个当前无窗口入口均实际运行；环境创建、安装、inspect_model、正式run_phase0、check_timestep、validate_artifacts和pytest均已真实执行。未实现新的Gymnasium控制任务或任何学习训练。

## 八个问题的回答与下一阶段建议

1. 可以在当前Windows/MuJoCo3.17直接加载并积分，资产路径正常；不是只查看静态图片。
2. 真正具有惯性、重力、关节和6个weld的动力学模型；机器人碰撞过滤关闭是其重要局限。
3. 有6个可用支链力执行器。nu=12中的另6个是外扰通道，不计为六缸控制的额外自由度。
4. 真实链式Python IK与MuJoCo几何一致。原README理想B/T与实际模型不一致，上端0.25mm非共点偏置使简单六长度公式不能精确替代。
5. 9个原2ms试验无数值发散/警告，但7个受控动作未通过严格weld角门槛；1ms/0.5ms共14个复核通过。零输入会真实下沉，不应称为位置稳定。
6. 已通过真实6力通道实现Home、Z±5mm和Roll/Pitch±0.5°，日志、PNG、MP4来自真实积分。
7. 具备**以1ms、原力限幅、真实链IK开展有限小范围Phase1基线控制研究**的条件；不具备直接训练RL、扩大工作空间或声称碰撞安全的条件。
8. 后续最少应冻结1ms配置/关闭GUI生命周期保护、明确真实铰点与slide行程语义；进入大范围运动前必须补机器人碰撞几何、实际铰角/行程/负载限制及更广工作空间验证。不能只降低weld门槛来掩盖原基线问题。

Phase0交付后暂停，未自动开始Phase1或PPO/SAC/BC。仍待用户审核模型几何与后续范围。
'''
    (ROOT/'docs/03_Phase0验收报告.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'results/phase0_acceptance.json',dict(status='PHASE0_AUDIT_COMPLETED_WITH_DOCUMENTED_BASELINE_FAILURES',original_2ms_small_motion_gate_passed=0,original_2ms_small_motion_trials=7,one_ms_passed=int(dt.loc[dt.timestep_s==.001,'passed_unchanged_criteria'].sum()),half_ms_passed=int(dt.loc[dt.timestep_s==.0005,'passed_unchanged_criteria'].sum()),tests=int(tests.get('tests')),new_training_steps=0,gui_final_close_passed=True,formal_physics_steps=127500,next_phase='Await review; only bounded 1ms baseline control is supported, not RL or collision-safe workspace claims'))
    (ROOT/'.gitignore').write_text('__pycache__/\n*.pyc\n.pytest_cache/\npytest-cache-files-*/\nresults/runs/\n.env\n',encoding='utf-8')
    print('Reports generated from real CSV, inspection, media and GUI exit records.')
if __name__=='__main__':main()
