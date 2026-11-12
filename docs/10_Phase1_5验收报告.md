# Stewart Phase 1.5 验收报告

本阶段已完成受限交互、双窗口、真实执行记录、历史回放及同步视频。没有训练 PPO/BC/SAC，没有修改 Phase 0/Phase 1 控制器、模型或正式结果。工作目录：`__PROJECT_ROOT__`。

## 真实运行证据

- 自动测试：**85 项，失败/错误 0 项**；包含原 57 项回归，详见 `results/phase1_5_pytest.xml`。
- 冻结历史：698 个旧文件逐一 SHA256 复核，变化 0 个。
- 录制开关：同一固定输入的 8000 步完整 integration state 逐位一致，最大差异 0。
- 最终双窗口：8000 步、每步 458 个 integration state 数值，与无窗口基线逐位一致；终止语义相同。
- 原生镜像：1084 次样本，qpos/qvel/ctrl/平台位置最大差异 0；没有额外 mj_step。
- 失配诊断：首次保护 0.126s，原因 `restricted_orientation_envelope`，保持至 0.326s。不是静态安全停止，保持期间真实位姿继续变化，完整日志保留。
- 运动中关闭原生窗口：实际执行 24360 步 / 24.360s，保存 609 帧，归档原因 `native_viewer_closed`；owner 与 mirror 退出码均 0。

本阶段交互/验收的全部 session 摘要见 `results/phase1_5_run_summary.csv`；固定预设及失配测试见 `phase1_5_backend_validation.json`。这些是界面验收，不替代或扩充旧 Phase 1 正式成功率。未筛掉失败尝试。

## 操作验收

Windows 实测了 W/A/I、Z 滑块、SPACE 真实状态保持、R 平滑 Home、P 暂停、Reset 新 episode 和 C2、原生窗口关闭、录制中关闭、暂停时关闭及失败后关闭。操作记录位于各 episode 的 events.jsonl；进程退出码位于 session.json。

原生窗口 W 不发送平台运动指令。自动重复键拒绝及 100 个连续目标的合并由新增自动测试覆盖；不把它们说成已经人工按住全部按键的穷尽测试。原生窗口 Q/Esc 另提供退出请求；本轮系统 Close 按钮已实测。所有测试只支持当前 Windows/驱动/模型条件，不保证任意远程桌面环境无故障。

历史回放已实际打开并操作逐帧/播放。现场 RGB 与仅状态重建明确区分，Replay 不额外执行物理步。日志缺少的旧内部求解器状态不被虚构。

## 视频与图表

| 文件 | 帧数 | 时长 | 分辨率 / 帧率 | 来源 |
|---|---:|---:|---|---|
| stewart_phase1_5_teleop.mp4 | 625 | 25.0s | 1920×1080 / 25fps | Windows 键盘与滑块现场 RGB，截取物理时间9～34s |
| stewart_phase1_5_dynamic_dashboard.mp4 | 150 | 6.0s | 1920×1080 / 25fps | C0 真实 Z 预设执行 |

两个视频所有帧均实际解码通过，画面有变化，真实 qpos 有位移；状态索引与哈希一致，曲线没有未来数据。逐帧证据见 `results/phase1_5_video_frame_audit.json` 及视频同名 `.frame_audit.json`。视频中的小运动未被人为放大。

截图：`assets/plots/phase1_5_native_window.png`、`phase1_5_control_panel.png`、`phase1_5_failure_panel.png`。曲线：`phase1_5_six_slide_strokes.png`、`phase1_5_platform_attitude.png`。截图只含项目窗口；其他应用的遮挡画面未作为交付截图。

## 最终十项回答

1. 可以用键盘控制上平台受限 Z/Roll/Pitch，须聚焦控制面板。
2. 可以用三个滑块设定目标；组合目标须通过受限椭球及完整 IK 预检。
3. 物理 owner 确实通过六执行器力输入和 mj_step 运动。原生 Viewer 是该真实状态的独立同步镜像，现场录像来自 live owner。
4. 六移动副相对行程可实时观察，支持单缸、参考/实际、误差、5/10/20秒。
5. Roll/Pitch/Yaw 均有参考和实际曲线，控制误差继续使用 SO(3)。
6. 可以按支链查看五个被动关节。
7. 两个最终视频同步展示机器人、真实曲线和状态。
8. 曲线时间来自 MuJoCo 物理时间，所有数据来自实际运行。
9. 当前固定输入的录制/窗口开关对照逐位一致；墙钟耗时会改变。该结论不推广到不同人工输入时刻。
10. 后续 Gymnasium/PPO 可以复用遥测和记录结构，但尚未接入 RL，当前奖励/策略字段为 N/A。

## 限制与未开展事项

保持 PARTIAL PROXY QUERY：86配对不等于完整碰撞安全。未开放 X/Y/Yaw，未作整机接触认证、连续碰撞证明或实机测试。GUI/视频是低速教学工具，非硬实时控制器。没有开展新算法、RL 训练或扰动自由遥操作。额外控制器对比视频为可选项，本轮未制作。

保留的失败尝试及修复经过见 `phase1_5_interaction_validation.json`。其中早期 Alt+F4 不能证明运动中原生关闭；最终用原生系统关闭按钮完成了独立验收，不能把前一次墙钟预算结束混记为关闭成功。

## 运行命令

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts\teleop_stewart.py --record
& D:\miniconda\envs\stewart_rl\python.exe scripts\view_stewart_dashboard.py --preset z --record --max-sim-seconds 6 --close-on-complete
& D:\miniconda\envs\stewart_rl\python.exe scripts\replay_stewart.py results\runs\phase1_5_session_1793093569084997300\episode_000
& D:\miniconda\envs\stewart_rl\python.exe -m pytest tests -p no:cacheprovider
```

这些入口及参数已实际运行；安装重建见文档08。操作说明见 `08_Phase1_5交互操作说明.md`，数据所有权与视频来源见 `09_Phase1_5实时曲线与视频设计.md`。

**完成 Phase 1.5 后暂停，等待审核，不自动开展 RL。**
