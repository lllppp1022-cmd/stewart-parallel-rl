# 真实升降、PPO训练与回放

工程：`__PROJECT_ROOT__`。解释器：`D:\miniconda\envs\stewart_rl\python.exe`。

## 先看这段视频

`assets/videos/phase2c2_actual_training_visible_v2.mp4`由seed42实际随机训练的第2、8、16回合RGB生成。每回合物理六秒，视频四倍慢放。右侧是同一帧的四倍像素裁剪，机器人没有被人为放大位移；下方是实测Z与参考Z。原视频在各`actual_training/episode_XXXX/actual_rgb.mp4`，原CSV和状态也一起保留。

```powershell
Set-Location __PROJECT_ROOT__
Start-Process .\assets\videos\phase2c2_actual_training_visible_v2.mp4
```

位移是参考轨迹驱动的真实三毫米升降；PPO只叠加小幅力修正，不应将整段运动归功于PPO。平台全尺寸约一米，三毫米在全景中仍较小，因此必须结合近景和毫米曲线。原画面中的数值量尺是示意，不能误读为放大后的物理机器人。

## 交互查看一次真实训练回合

以下入口只读历史训练，不开始新训练。打开后自动播放；底部Play/Pause暂停/继续，Frame >逐帧，速度下拉框控制播放倍率。机器人画面窗口与曲线窗口分开。关闭窗口正常退出。

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts\watch_phase2c2.py --seed 42 --episode 2
```

可换`--episode 8`或`--episode 16`。没有现场RGB的回合会明确标记状态日志重建，不称为现场录像。第2回合跨越首次256决策PPO更新，顶部策略版本会从0变成1。不要假设每个回合只对应一份策略权重。

注意旧物理Owner的局部`episode`字段始终为0，原RGB内也沿用了这个局部编号；这是旧记录层的标号限制，不能拿它分组训练回合。正确的全局回合编号来自`training_samples.csv`和`episode_XXXX`目录。新回放程序顶部和综合视频使用这个正确编号。物理时间每个回合从0开始，全局决策编号连续递增。

完成本轮记录后，已为后续新环境修复Owner编号同步，并新增重置后的编号/初始观测一致性测试。历史原CSV和RGB保持原样，因此仍需按上面的全局目录读取。这项修复只影响元数据，不进入控制或观测特征。

## 查看学习曲线和更新

本轮TensorBoard使用**6032**，6031是旧任务。训练时每256个20Hz决策收集一次rollout，然后暂停环境采样更新Actor/Critic。机器人每个动作实际执行50次1ms积分，不是每个物理步都更新神经网络。

打开：`http://127.0.0.1:6032/#timeseries`。

服务器未运行时：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe -m tensorboard.main --logdir __PROJECT_ROOT__\results\runs\phase2c2_20261030_v1 --host 127.0.0.1 --port 6032
```

查看`seed42`或`seed123`。`audited_complete_updates`是由真实CSV补齐的全部更新，包括SB3常规输出未及时dump的末次更新，不能当成另一次训练。重点同时查看独立评价误差、安全状态和训练奖励。

```powershell
Get-Content .\results\phase2c2_training_console.log -Tail 20
Import-Csv .\results\phase2c2_evaluation_comparison.csv |
  Select-Object label,task_id,success,z_rmse_mm,xy_peak_mm,minimum_partial_proxy_m
```

## 已有验收命令

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts\watch_phase2c2.py --episode 2 --validate-seconds 6
& D:\miniconda\envs\stewart_rl\python.exe -m pytest -q -p phase2a_pytest_temp_plugin -p phase2c1_pytest_policy
```

前者进行了真实Windows Qt窗口的自动播放与关闭测试，截图保存在`results/phase2c2_replay_*playing.png`。它不是新的物理执行。后者保留Phase0“不得安装学习库”的历史断言为明确的xfail；后续阶段已授权Torch/SB3，不能把这项计作普通通过。

本轮没有把实时GUI附着到训练物理线程，采用训练现场RGB加独立回放，且用真正PPO验证录像不改变随机采样和最终参数。新的连续实时GUI是未执行项。训练入口`train_phase2c2_paired.py`有目录存在保护，不应为了看机器人而重新启动训练；直接使用以上回放和视频。
