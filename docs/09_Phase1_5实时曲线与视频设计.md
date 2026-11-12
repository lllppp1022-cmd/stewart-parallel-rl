# Phase 1.5：实时曲线与视频设计

## 数据所有权

`src/visualization/owner.py` 是唯一 live 物理 owner。命令队列最多 64 项；Qt 只发送命令并读取不可用于改写物理数据的快照。物理 owner 使用原六缸力控制器，将 N 单位的力指令写入前六个 data.ctrl，其他六路外力通道归零，再调用 mj_step。每步 0.001 s，随后 mj_forward 与真实状态监控。没有动态赋值 live qpos/qvel 的动画逻辑。

Reset 退出并归档旧进程，再创建新进程和 MjData。原生 Viewer 位于第三个进程，其独立显示对象接收 mjSTATE_INTEGRATION，mj_setState + mj_forward 重建当前画面。它不调用 mj_step。原生 GUI 的重置、鼠标扰动等只影响这个显示对象，下次快照会恢复物理 owner 状态，不能回传物理控制。实际目标/状态在控制面板及原生窗口文字中分别显示。

这是为了隔离 MuJoCo `sync` 的双向 GUI 交互以及 Windows 窗口关闭风险。参考官方 [passive viewer 文档](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。不能仅因为设置 state_only=True 就假定 GUI 不会影响共享数据。本实现没有逐物理步争抢 viewer.lock；只有相机更改使用短锁。

## 采样、绘图与存储

物理日志为 1000 Hz 全量 gzip CSV；完整 integration state 为 NPZ。平台位姿、六相对行程、qvel、ctrl、实际力、参考值、weld、代理间隙、失败原因与时间逐步对应。代理距离查询沿用 Phase 1 的 50 Hz 周期；界面显示最近一次查询值，不伪称每毫秒重算全部 86 配对。

遥测每 10 步发送一次（100 Hz），显示队列 64 项、镜像队列 8 项，采用丢弃旧显示帧保留最新帧的非阻塞策略。显示缓冲最多 2200 个采样，GUI 目标刷新 20 Hz。慢电脑、远程桌面和录制情况下可能达不到该墙钟刷新率，但不能丢弃原始物理日志或增大物理 dt。物理与 GUI 都有有限运行预算。

曲线只根据快照的物理仿真时间绘制。暂停时没有新物理步，图线保留。显示可以跳过旧快照，离线分析使用完整日志。六缸按原 MJCF 顺序，不按视觉左右重新编号。三个姿态角只用于显示；控制和姿态误差保持原 SO(3) 定义。

Replay 从已保存 CSV 和现场 RGB 读取。曲线固定按 10 ms 下采样并缓存，使用二分查找取当前时刻之前的窗口。不存在为了显示而重新采样策略、重新调用控制器或积分环境的路径。

## 现场 RGB 与合成视频

`recording.py` 每 40 个真实物理步获取一幅 1216×912 RGB，即 25 fps；来源始终为物理 owner 的 live MjData。渲染/编码错误被记录并关闭录制，不中止正常控制。退出时依次关闭日志、编码器、Renderer 并保存状态归档。最终遥测发送异常也不会跳过归档。

`frames.jsonl` 包含原始帧号、state_index、step、物理时间、episode、控制器、运行状态与 qpos SHA256。`integration_states.npz` 的完整状态可核对该索引。原生显示镜像另核验 qpos/qvel/ctrl/平台位置；这些核验不向物理系统回写任何内容。

合成程序只读取原始 RGB 和日志，输出 1920×1080、25 fps、H.264：左侧 1216 像素（63.3%）为真实机器人，右侧 704 像素为三组曲线，底部 168 像素为状态。编码 macro_block_size=8，避免 1080 被自动补成 1088。

每帧曲线只取 `time_s<=frame_time` 的过去数据，窗口最大 10 s。仿真截取从 t=9 s 开始的视频仍可以显示这之前已真实发生的过去数据，不会展示未来数据。合成器以独立 scratch MjData 解包保存状态核对哈希，**不调用控制器、不调用 mj_step**。逐帧映射保存到同名 `.frame_audit.json`。

早期编码器补边产生的 1088p 尝试保留为 `phase1_5_attempt_1088p.*`，不是最终交付视频。失败记录不删除。最终视频分辨率、可解码帧数、图像变化、qpos 变化和过去数据约束均由验收脚本重新检查。

## 保护语义

输入目标超过受限椭球属于目标拒绝，不改变当前参考。完整 IK/行程/代理预检失败或真实状态保护触发后，锁存 FAILED，取消待执行目标，并以当前实际构型执行原 0.2 s 受限保持。保持不是机械制动，实际状态仍可继续偏离；报告保留这个过程。只允许 Reset 新建 episode 清除锁存。

保持六缸 ±400 N、原行程、原位姿和 weld 阈值。PARTIAL PROXY QUERY 不覆盖完整原生接触/铰接几何；离散参考预检、50 Hz 代理查询和仿真成功不构成连续安全或实机认证。

## 训练扩展接口

`telemetry.py` 定义 Episode、Decision Step、Policy Version、Actor Mean、Sampled Action、Applied Action、Last Completed Reward、Terminal Reason 字段。奖励只能在完成真实环境 step 后由生产者提供；结构会拒绝未标记 completed step 的非空奖励。当前所有训练字段为 None，显示 N/A。

来源标签分别为 DYNAMIC CONTROL DEMONSTRATION、ACTUAL STOCHASTIC TRAINING、DETERMINISTIC POLICY EVALUATION、RECORDED STATE REPLAY。未来 Gymnasium 可以复用命令/遥测/录像结构，但必须在实际 step、策略采样与更新入口完成集成和非侵入验收。本阶段只提供框架，不能声称已经实现或验证 PPO 训练观察。

## 验证和复现

`scripts/validate_phase1_5.py` 执行有限固定输入、六种预设和失配诊断；`configs/phase1_5_equivalence_events.json` 保存物理步编号固定的输入。无录制、开启录制、真实双窗口开启的比较使用相同固定事件。人工操作按墙钟输入，天然受人和界面延迟影响，不宣称任意两次人工操作可逐位重复。

最终验收数据为 `results/phase1_5_interaction_validation.json` 与 `results/phase1_5_video_frame_audit.json`。历史冻结清单为 `phase1_5_frozen_history.json`；脚本按原始文件哈希核对，不把新结果冒充原 Phase 1 实验。
