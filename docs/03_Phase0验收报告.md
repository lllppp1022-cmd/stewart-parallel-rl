# Phase 0验收报告

日期：2026-10-26T17:58:31.798226+08:00。项目独立位于__PROJECT_ROOT__。本阶段仅模型、几何、有限动力学和Windows GUI审计，不训练PPO/SAC/BC，不进入Phase1。

## 实际环境与来源

Python 3.11.17 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:16:54) [MSC v.1942 64 bit (AMD64)]；解释器D:\miniconda\envs\stewart_rl\python.exe。Windows版本报告：Windows-10-10.0.26200-SP0（CIM查询被拒绝，未据此臆测完整OS名称）。实际GPU：NVIDIA GeForce RTX 3060, 560.94, 12288 MiB。

依赖：mujoco=3.17.0, numpy=2.4.6, scipy=1.17.1, gymnasium=1.4.0, matplotlib=3.11.2, pandas=3.1.0, pytest=9.1.1, imageio=2.38.1, imageio-ffmpeg=0.6.0, pillow=13.0.0。独立Conda环境stewart_rl已真实创建并联网安装；无旧工程学习依赖回退，Torch/SB3均未安装。完整freeze、创建和安装日志保留。未对Reacher、Panda、盾构研究工程执行编辑、覆盖或训练；没有声称对全部旧工程做全量逐字节审计。

上游MIT仓库commit b0195ba9f40f6e0876220be1712eb47f5fc24025，原模型/资产哈希见phase0_model_inspection.json；git status确认上游受跟踪文件未改。渲染缓冲仅在内存中扩大至960×720，没有改MJCF物理模型。

## 正式原2ms基线：保留全部9组

协议预先保存在configs/phase0_protocol.json，各5秒，1秒保持/2秒运动/2秒保持；7组目标跟踪另加zero与pulse诊断，共22500个实际物理步。7组小幅动作的严格组合门槛含全程weld角间隙<1e-4rad；zero/pulse不算到达成功率。

| 场景 | 步数 | 终点位置误差mm | 终点姿态误差deg | 最大weld角rad | 数值稳定 | 严格小幅门槛 |
|---|---:|---:|---:|---:|---|---|
| zero | 2500 | 104.812736 | 0.0100698 | 0.00103009433 | PASS | N/A |
| home | 2500 | 0.001121 | 0.0000515 | 0.000222575083 | PASS | FAIL |
| z_up | 2500 | 0.001491 | 0.0000570 | 0.000222643845 | PASS | FAIL |
| z_down | 2500 | 0.001238 | 0.0000467 | 0.00022435398 | PASS | FAIL |
| roll_pos | 2500 | 0.007538 | 0.0001001 | 0.000222589051 | PASS | FAIL |
| roll_neg | 2500 | 0.013420 | 0.0002209 | 0.000226145495 | PASS | FAIL |
| pitch_pos | 2500 | 0.011990 | 0.0001680 | 0.000224071983 | PASS | FAIL |
| pitch_neg | 2500 | 0.011569 | 0.0001764 | 0.000222575083 | PASS | FAIL |
| pulse | 2500 | 0.058332 | 0.0008916 | 0.000222965954 | PASS | N/A |

zero控制下平台下沉约104.8mm并到达软限位附近，数值有限不等于Home保持成功。其余真实力控动作都实现小幅位姿响应，既没有live qpos传送，也没有使用平台外力通道。7组严格组合判定失败的原因是内部weld角间隙，而不是位姿目标完全未到达；失败没有隐藏。

## 有界步长敏感性复核

独立协议在敏感性run开始前冻结：1ms/0.5ms各相同7组、每组5秒、阈值不变，额外最大105000步。未重新选种子或移动几何，原2ms结果不覆盖。

| dt(ms) | 通过/组数 | 最大weld角rad | 最大终点位置误差mm | 最大执行器力N |
|---|---:|---:|---:|---:|
| 1 | 7/7 | 5.65162585e-05 | 0.013561 | 89.9906 |
| 0.5 | 7/7 | 1.41278081e-05 | 0.013574 | 89.9907 |

正式基线+数值敏感性共127500次物理积分。开发试跑、自动测试、CLI/GUI另外记录，不把这些重复执行当作独立场景统计成功率。两个步长通过仅说明这些有限任务下满足冻结检查；refsafe有效约束时间尺度随dt变化，不能宣称一般连续时间稳定性证明。

## 几何与自动验收

7个小范围构型×3个差分步长=21行；独立FK与MuJoCo最大位置差1.11998e-15m，Jacobian最大绝对差3.71059e-09，原生actuator_length与预测slide坐标一致。10m不可达目标保守拒绝。未找到/未声称验证全工作空间奇异面。

最终pytest：**25 passed，failures=0，errors=0**。覆盖加载、几何、真实链IK、FD、行程、无效输入、近奇异诊断、闭链、真实力响应、零控制下沉、1ms保持、独立scratch、日志/视频状态哈希、保留2ms失败、步长复核与环境隔离。首轮23项通过但pytest缓存目录创建被Windows拒绝；该记录保留，随后关闭非必需缓存插件，最终25项正常通过，未改ACL/删除失败目录。

## Windows、视频与故障记录

原生静态/动态Viewer已真实打开并观察。静态窗口点击X后以viewer_closed归档；2ms动态90秒预算退出成功。1ms动态手动关闭早期3次出现0xC0000005，最后一次栈定位viewer.lock。按官方owner-step/sync循环移除逐物理步外层viewer mutex后，实际关窗复核退出0：results\runs\dynamic_demo_1793008231496790300，物理步36833，终止类型viewer_closed。未把此前崩溃说成通过。

视频：assets\videos\phase0_simulation.mp4，45.0秒、450帧、10.0fps、960×720。全部450帧的状态哈希与物理NPZ逐帧一致，完整解码通过；视频像素来自执行期间live MjData，没有事后用qpos重放冒充现场。RGB文件有损压缩，不声称RGB逐字节等于渲染源；哈希核对针对当时的物理状态。

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
