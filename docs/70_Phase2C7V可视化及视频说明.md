# Phase 2C-7V：可视化及视频使用说明

## 直接复制运行：只读训练分析

```powershell
Set-Location '__PROJECT_ROOT__'
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_training.py
```

选择seed、All stages/A/B/C、指标。Raw data显示真实点/数据；Display trend只控制显示平滑，不覆盖CSV。Reward五项可切换绝对代价或占比；Actor/Critic、Value loss、EV、KL、clip、标准差、价值目标/优势和固定开发评价都可选择。鼠标缩放/拖动曲线，右键可自动恢复范围。checkpoint选择显示该实际文件训练步数及SHA，只读、不加载训练器。67个文件/seed均可查看，只有9个已评价点有新开发曲线。

## 真实动力学冻结策略Viewer

```powershell
Set-Location '__PROJECT_ROOT__'
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --seed 42
```

窗口选择F_roll/F_pitch/F_mix/F_30、冻结combined baseline/Classic50/PPO及seed后点Reset。Pause/resume暂停/继续；Single decision暂停并推进一次50Hz决策，即20个真实1ms积分；拖动机器人视图调整相机，滚轮缩放。每个回合按冻结任务时长结束，窗口保留结果，可Reset运行新真实回合。关闭窗口归档，不会改变历史目录。八条实时曲线包含Z、Roll、Pitch、SO3、外载、请求/接受力矩、六支链相对行程和驱动力。只允许三个已验证final和四个已验证任务；早期checkpoint不任意开放到最终任务。

```powershell
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --classical
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_policy.py --task F_pitch --seed 123
```

Viewer是新的冻结确定性评价，不是历史随机训练现场。Qt主线程持有live MjData并做data.ctrl/mj_step，绘制读取同一状态，GUI不改qpos或控制限制。

## 增强视频

- `assets/videos/stewart_phase2c7_training_analysis.mp4`：90秒、2250帧、1920×1080/25fps。原seed42六段真实随机训练RGB，不重新predict、不重造动作。左上机器人/当前值，右上姿态/外载，左下完整Return历史，右下请求/边际接受/支链力/物理代价。Current checkpoint显示当时采样策略版本×512，当前决策可能位于其后。完整回合跨更新边界时版本随真实日志变化。
- `assets/videos/stewart_phase2c7_policy_comparison.mp4`：56秒、1400帧、同分辨率。F_roll相同任务classic50、seed42、123、2026各14秒真实确定性执行。该任务预先选作已知加载/卸载权衡示例，不是事后挑最佳任务。

训练视频明确POST-PROCESSED TRAINING VISUALIZATION；已完成Return为蓝色，未来数据灰色，截断片段x。任何完整回合要等结束后才算历史点，不把当前片段提前当完整回报。训练RGB原档只有seed42，另外两个seed只有完整数字日志，未伪造它们的训练RGB。本轮三seed对比是新冻结评价录像。

每个源帧核对frames.jsonl时间、integration_states qpos SHA和对应物理行；输出视频全帧解码通过。验证文件为training_video_validation.json、comparison_video_validation.json。姿态扰动角很小，机器人保持真实几何比例，不能为了肉眼显著而放大qpos；看右侧角度/误差曲线判断细微抗扰变化。

```powershell
Start-Process '__PROJECT_ROOT__\assets\videos\stewart_phase2c7_training_analysis.mp4'
Start-Process '__PROJECT_ROOT__\assets\videos\stewart_phase2c7_policy_comparison.mp4'
```

## TensorBoard

原标签`attitude/reward`、`attitude/episode_return`及五项分量仍可看。派生目录逐决策完整记录，完整回合标签为`training/complete_episode_return`，片段独立`training/incomplete_fragment_return`，优化器标签以`optimizer/`开头，固定任务以`evaluation/`开头。三类曲线不要混读。

```powershell
& 'D:\miniconda\envs\stewart_rl\python.exe' -m tensorboard.main --logdir_spec 'original:results/runs/phase2c7_20261107_v1/training,derived:results/phase2c7v_tensorboard' --host 127.0.0.1 --port 6033
```

浏览器打开http://127.0.0.1:6033。端口6032原服务不修改。Qt单独即可看曲线，不要求同时开三个窗口。数据已冻结，浏览器刷新不会自动增加训练步数。该启动命令已经实际运行，HTTP 200；服务加载完成后确认原6个run及派生3个seed均可见，验证在tensorboard_http_validation.json。首次打开要等待后台读取事件文件，不能把首次加载不全误认为缺少记录。

## 已实际执行的验收入口

```powershell
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_training.py --validate
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --seed 42 --validate-steps 1000
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/view_phase2c7v_policy.py --task F_roll --classical --validate-steps 1000
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/evaluate_phase2c7v_checkpoints.py collect
& 'D:\miniconda\envs\stewart_rl\python.exe' scripts/plot_phase2c7v.py
& 'D:\miniconda\envs\stewart_rl\python.exe' -m pytest tests/test_phase2c7v.py -q
```

分析窗口实际检查144个seed×stage×指标组合。两个真实物理GUI测试均与无窗口记录前1000步逐位一致，包含暂停与20步单次推进。截图保存在7V运行目录。初版Qt控件命名覆盖metric虚函数及阶段列缺失已修复，失败日志保留。没有通过修改环境解决GUI问题。
