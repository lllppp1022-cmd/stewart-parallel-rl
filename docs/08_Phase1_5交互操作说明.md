# Stewart Phase 1.5：交互操作说明

本阶段只有传统力控制演示，没有 PPO、BC 或 SAC 训练。请先阅读安全边界，再操作。项目根目录为 `__PROJECT_ROOT__`。

## 1. 安装与启动（Windows PowerShell）

已有 `stewart_rl` 环境已实际安装并验证。新增的最小图形依赖为 PySide6-Essentials 6.12.0、shiboken6 6.12.0、pyqtgraph 0.14.0；原依赖版本保持不变。没有安装包含 WebEngine 的完整 PySide6 套件。

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe -m pip install -r requirements-phase1_5-lock.txt
& D:\miniconda\envs\stewart_rl\python.exe scripts\teleop_stewart.py --record
```

第二条安装命令用于重建依赖；本机依赖已安装，无需每次运行。进入 VS Code 后选择解释器 `D:\miniconda\envs\stewart_rl\python.exe`，打开项目根目录，在集成 PowerShell 执行第三条命令。不要用其他项目的 Python。

程序打开两个窗口：

- **MuJoCo 原生窗口**显示当前真实执行状态的独立镜像，可用鼠标调相机。
- **Stewart Phase1.5 Control Panel**接收运动命令并显示实时曲线。请先点击该控制面板，再按运动键。

原生 Viewer 的 `sync` 会接收 GUI 输入，因此不能把它与正在积分的 live MjData 无隔离地共享。本实现由唯一物理进程执行六缸力控制；原生窗口使用另一个 MjData 显示物理进程的完整状态快照，标注 **LIVE STATE MIRROR**，不额外积分。现场录像则直接来自物理进程的 live MjData。镜像验证见验收报告。

## 2. 键盘与滑块

| 控制面板按键 | 实际功能 |
|---|---|
| W / S | Z 目标增加 / 减少 0.25 mm |
| A / D | Roll 目标增加 / 减少 0.05° |
| I / K | Pitch 目标增加 / 减少 0.05° |
| R | 平滑返回 Home，不传送 qpos |
| Space | 以当前真实位姿、六行程建立受限保持，清除待执行目标 |
| P | 暂停 / 继续物理积分；暂停时仿真时间与曲线冻结 |
| V | 切换三个相机视角 |
| Q / Esc | 退出并等待日志、视频归档 |

系统忽略 Qt 自动重复键事件。原生窗口中的 W/S 等键不作为运动指令；原生窗口另支持 Q/Esc 提交退出请求。使用录屏或远程桌面时，操作系统的 Esc 也可能先被外部工具截获，可使用 Quit 或系统关闭按钮。

Z/Roll/Pitch 三个滑块是**目标**，不是对真实状态的直接赋值。键盘目标会同步回滑块。改变一个滑块时，其他两个保持显示的目标值。快速变化的目标合并到一个待执行目标；当前已验证的运动段会先完成。Home/Hold 可取消待执行目标。

允许的目标椭球为 `(Z/4mm)^2+(Roll/0.4°)^2+(Pitch/0.4°)^2<=1`。因此三个滑块同时拉满会被拒绝。X、Y、Yaw 按钮保持禁用。参数默认值见 `configs/phase1_5_interface.json`；当前控制器实际采用的限值以源码与 Phase 1 冻结协议为准，修改显示配置不自动构成新的工作区验收。

每段用 2 秒五次时间缩放和 SO(3) 旋转插值，在 41 个节点执行完整链式 IK、条件数、行程和代理间隙预检，并对六行程做 1 ms 插值范围检查。预检期间仿真时钟暂时冻结，控制面板仍可响应。这个教学系统不承诺硬实时。

## 3. 控制器、保护与 Reset

当前执行控制器在 ACTIVE 字段显示。选择 Next controller 后，必须按 **Reset**，才会建立新的 episode、物理进程和 MjData。不能在原 episode 内直接改控制器或传送真实状态。

保持原 Phase 1 的 1 ms 积分、±400 N 原生执行器限制、slide [-0.1, 0.35] m、位姿保护、weld 平移/旋转 1e-4 阈值及已覆盖代理的 5 mm 门槛。受限保持不是绝对安全停止，实际机构可能继续运动。

目标越出受限椭球：拒绝该目标并保持已有目标。IK 预检失败、实际位姿越界或约束异常：锁存 FAILED，记录原因，执行 0.2 秒受限保持，此后停止继续积分等待 Reset/退出。失败不自动清除。

asymmetric_gain 在 Home 保持期已知会失败，默认拒绝自由遥操作。仅可运行显式锁定诊断：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts\teleop_stewart.py --condition asymmetric_gain --diagnostic-asymmetric --record --max-sim-seconds 1 --max-wall-seconds 120
```

**PARTIAL PROXY QUERY** 表示只检查 Phase 1 的 86 个代理配对，不含完整 Cardan 接触几何，不是整机碰撞认证。

## 4. 看曲线和录制

主界面依次显示世界 XYZ（mm）、Roll/Pitch/Yaw（deg）、六移动副相对行程（mm）、六实际力与指令（N）。虚线是参考/指令，实线是实际值。XYZ 同轴时 Z 的约 1100 mm 会压缩 X/Y 变化，可用 Position plot 单独选 X、Y 或 Z。小姿态自动缩放；有些坐标轴用科学计数或自动 SI 前缀，请同时看单位和倍率。

Leg 选择器支持全六根或单根；Ref/Actual 可以隐藏；Leg error 显示 actual-reference。时间窗口可选 5/10/20 秒。鼠标移到曲线上可查看最近物理采样值。Passive joints 页每次仅显示选定支链的五个被动转动关节，单位 deg。

`actuatorpos` 是移动副的**相对行程**，不是整根支链长度或连接点弦长。绘图颜色与日志中的六缸顺序保持一致。F_cmd 与 F_actual 分开记录，执行器增益失配时可能不同。

Record 只切换现场 RGB 采集，完整物理日志始终保存。录制会降低墙钟运行速度，不改变 1 ms 物理精度。每次启动生成独立 `results/runs/phase1_5_session_<时间戳>/episode_000/`。Reset 递增 episode。

每个 episode 保存：`physics.csv.gz`、`integration_states.npz`、`events.jsonl`、`config.json`、`plant.json`、`summary.json`；开启录制还会保存 `actual_rgb.mp4` 和 `frames.jsonl`。窗口镜像有 `mirror_validation.json`。使用 Screenshot 保存纯本项目面板截图；Export CSV 建议在停止后或 Replay 中使用，避免读取尚未关闭的 gzip。

## 5. 预设、无窗口、回放和视频

```powershell
# 实际测试过的 Z 预设，显示双窗口并录制
& D:\miniconda\envs\stewart_rl\python.exe scripts\view_stewart_dashboard.py --preset z --record --max-sim-seconds 6 --close-on-complete

# 高速无窗口执行，保留真实物理记录；不额外录制RGB
& D:\miniconda\envs\stewart_rl\python.exe scripts\teleop_stewart.py --headless --unpaced --preset z --max-sim-seconds 6

# 回放已实际产生的键盘/滑块 episode
& D:\miniconda\envs\stewart_rl\python.exe scripts\replay_stewart.py results\runs\phase1_5_session_1793093569084997300\episode_000

# 查看所有选项
& D:\miniconda\envs\stewart_rl\python.exe scripts\teleop_stewart.py --help
```

Replay 中 P 或 Play/Pause 控制播放，右箭头或 Frame > 前进一帧，右下选择 0.25/0.5/1/2/4 倍速度。Open Replay 选择其他 episode 文件夹；文件夹对应的 config/summary 可用于区分控制器和工况。换文件夹即选择不同实验，不会重跑控制器。

有现场视频时，机器人窗口标注 ACTUAL EXECUTION CAPTURE；仅有历史状态时标注 RECORDED STATE REPLAY。旧 Phase 1 的 qpos/qvel/ctrl 日志可以重建可视状态，但不能声称恢复了当时未保存的完整物理内部状态。两种回放均不调用 mj_step。

已交付 `assets/videos/stewart_phase1_5_teleop.mp4` 与 `stewart_phase1_5_dynamic_dashboard.mp4`。若重新合成，必须选新的输出文件名，程序拒绝覆盖：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts\render_stewart_dashboard_video.py results\runs\phase1_5_session_1793093569084997300\episode_000 assets\videos\my_teleop_review.mp4 --start 9 --end 34
```

完成 Phase 1.5 后暂停；本系统尚未接入 RL 环境，界面中的奖励、Actor 和策略版本均为 N/A。
