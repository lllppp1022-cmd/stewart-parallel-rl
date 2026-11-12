# Phase2C-2训练复现

## 1. 回看已经发生的随机训练

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts\watch_phase2c2.py --seed 42 --episode 2
```

自动播放原训练记录，Play/Pause暂停，Frame >逐帧，速度下拉框调速。可选第8、16、32回合。第2回合跨第一次PPO更新，策略版本变化可见。有原RGB的回合播放训练现场帧；其他回合明确标记状态重建。此操作不进行训练。

```powershell
Start-Process .\assets\videos\phase2c2_actual_training_visible_v2.mp4
```

综合视频保留真实3毫米位移，4倍慢放和同帧像素近景均有标注。

## 2. 从头重跑相同配置的有限训练

独立入口为`scripts/reproduce_phase2c2.py`。不要直接运行绑定原目录的旧训练入口，也不要删除原目录来腾出名字。

先只检查：

```powershell
Set-Location __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts\reproduce_phase2c2.py --run-name phase2c2_reproduce_01 --check-only
```

本命令已实际执行通过：检查了训练源文件、当前环境/PPO配置与保存配置的一致性、旧安全门禁及新目录未占用；没有创建训练目录或采样。

确认要重新消耗训练计算后，去掉`--check-only`：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts\reproduce_phase2c2.py --run-name phase2c2_reproduce_01
```

这会真正从零均值Actor开始采样，依次进行两种子42/123的256、2048、最多4096累计决策；共同2048门禁未通过则一起停止。正式上限每种子204800次1ms物理积分，两个种子合计409600步；阶段独立评价另外计数。旧测试结果不复制成新训练结果。

本轮回复只验证了复现入口，没有再次实际执行这条长训练命令。原始两种子训练已经真实执行并验收。重跑后会自动生成`reproduction_parameter_comparison.json`，比较同预算下最终参数哈希。应优先保持相同Windows机器、Conda解释器、库版本和CPU线程设置。不同软件/硬件下不能预先承诺逐位相同，时间戳ZIP文件的哈希也不适合作为网络参数等价依据。

新输出：

- `results/runs/phase2c2_reproduce_01/`：实际采样、状态、录像、更新与验证。
- `models/phase2c2/phase2c2_reproduce_01/`：初始、每次更新、各阶段及最终模型。

输入目录和模型目录任一存在就拒绝启动。下次改用新的`--run-name`。新训练沿用已经验证的基线/观察器门禁，来源写入`reproduction_provenance.json`，不宣称这些检查被重新开展。

## 3. 看重跑的学习曲线

另开PowerShell：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe -m tensorboard.main --logdir __PROJECT_ROOT__\results\runs\phase2c2_reproduce_01 --host 127.0.0.1 --port 6033
```

浏览器访问`http://127.0.0.1:6033/#timeseries`，选择seed42/seed123。6032是原已完成实验。每256次决策才更新一次网络，不能期待每个物理步都有新的Actor/Critic指标。

训练不默认弹出实时物理GUI；它记录选定真实训练回合的现场RGB和全部动作/状态，避免显示线程扰动采样。归档完成一个回合后可独立打开：

```powershell
& D:\miniconda\envs\stewart_rl\python.exe scripts\watch_phase2c2.py --run-root __PROJECT_ROOT__\results\runs\phase2c2_reproduce_01 --seed 42 --episode 2
```

这是刚刚训练发生过的回合回放。窗口会显示真实动作、时间与策略版本，不会重新调用策略产生另一条轨迹。

## 4. 判断重跑是否完成

检查新目录里的`seed42/training_result.json`、`seed123/training_result.json`、`joint_2048_gate.json`及`final_evaluation_frozen.json`。更新日志是各种子的`ppo_update_events.csv`，完整动作记录是`actual_training/training_samples.csv`。

网络更新次数应与决策预算匹配：2048对应8次、4096对应16次。4096决策对应204800物理步，正常6秒回合有120次决策。最后可能是预算截断的部分回合，不能标为完整成功或失败。原实验得到小幅动态收益但未达到扩大预算门槛，复现不自动增加8192/16384预算。
