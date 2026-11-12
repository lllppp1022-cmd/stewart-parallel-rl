# Phase2C-4：PPO真实训练与超参数

## 冻结预算及生效参数

| parameter | value |
| --- | --- |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| learning_rate | 0.0003 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| log_std_init | -1.0 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| target_kl | 0.03 |
| net_arch | {"pi": [32, 32], "vf": [32, 32]} |
| device | "cpu" |
| torch_threads | 1 |
| seeds | [42, 123, 2026] |
| maximum_per_seed | 4096 |
| smoke_seed | 42 |
| smoke256_included_in_seed42_4096 | true |
| checkpoint_every_rollout | true |
| selection | "Last4096 checkpoint for every seed; no best-test selection" |
| record_seed42_episodes | [0, 6, 12] |


小型Tanh两层32单元Actor/Critic，Actor最终均值输出层置零，高斯log_std初始−1，随机探索仍保留，初始确定性均值接近零残差。使用版本`{"python": "3.11.17 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:16:54) [MSC v.1942 64 bit (AMD64)]", "platform": "Windows-10-10.0.26200-SP0", "mujoco": "3.17.0", "torch": "2.15.0+cpu", "sb3": "2.9.2", "gymnasium": "1.4.0", "numpy": "2.4.6", "pandas": "3.1.0"}`，CPU、Torch单线程；没有超参数搜索。

## 真正运行的三个种子

| seed | decisions | physical_steps | updates | optimizer_steps | complete_episodes | safety_terminations | wall_s | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 4096 | 204800 | 16 | 640 | 13 | 0 | 955.036116 | be200c06b19b504e490483b1ae799e0468eec97877d35e81dff42bfc56d9d934 |
| 123 | 4096 | 204800 | 16 | 640 | 13 | 0 | 940.3619 | df9b0bbae172d0cac6794f0613779164ba48c34ac0b6b3737e63c420faca15ae |
| 2026 | 4096 | 204800 | 16 | 640 | 13 | 0 | 942.01439 | b75d91f20fbe6c256f9ae88b527688567eb296f80d3d2172f56bb696fc8fe0f6 |


合计12288次PPO决策、614400次真实MuJoCo积分。42号种子256步冒烟是4096的前缀，接着3840步，不重复计费；另外两个各4096，均16次rollout优化。每回合300决策，256更新边界不等于回合边界，同一episode会跨策略版本。每个种子保存initial、最后模型及每256步checkpoint；按最后4096模型评价，未筛选最佳种子或最佳测试checkpoint。

冒烟保存/重载参数与确定性动作一致；在进行重载验证时保存并恢复Python/NumPy/Torch RNG，避免诊断消耗采样随机流。Actor和Critic每次参数差范数均记录并验证非零。每个优化期间实际物理步数不变。

| seed | decisions | raw_action_mean | raw_action_std | box_clip_fraction | budget_projection_fraction | accepted_residual_rms_N | safety_terminations | full_sample_reward_mean | last_parameter_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 4096 | -0.0345353479 | 0.374139709 | 0.00659179688 | 0.000620117188 | 0.0372863002 | 0 | -1.30436941 | d7dc8dbde0b5e4fafbee87b47e023dbbe41121933a6305cec5af203969757f34 |
| 123 | 4096 | 0.0151282339 | 0.377412142 | 0.00927734375 | 0.000688476563 | 0.0374598419 | 0 | -1.31273574 | 04027b8d2e130913d9773b53f37f75b9b2533eec3c29442f785aeb342ca9ba45 |
| 2026 | 4096 | -0.011262866 | 0.378164292 | 0.00903320312 | 0.000551757813 | 0.0374329904 | 0 | -1.27774764 | 903ac3c067db98289f1eb05b55a5b9586938fbd275c9e8f5f435002cc13c8d9c |


## 看训练与复盘

TensorBoard：`http://127.0.0.1:6034/#timeseries`，日志目录`__PROJECT_ROOT__\results\runs\phase2c4_20261105_v1\training`。采样窗口CSV含raw Gaussian、Box后动作、请求Fz、接受命令、预算投影、Z/XY/SO(3)、奖励分量。`ppo_update_events.csv`含Policy/Value loss、KL、PPO clip、explained variance、GAE及价值目标分布、参数变化和checkpoint哈希；`rollout_*.npz`保留训练buffer，真实物理日志逐1ms保存。

图：`phase2c4_learning_curves.png`、`phase2c4_action_vs_applied.png`、`phase2c4_update_timeline.png`、`phase2c4_reward_components.png`，都在assets/plots。完整回合回报与末尾未完成前缀分开；不以提前结束后较低RMSE证明优势。

真实训练录像取42号种子episode0/6/12，早中后期预先指定，不按表现挑选。原始RGB由训练进程实际积分时采集，其他episode默认不渲染；视频是这些RGB的后期加字，绝非最终策略确定性重新生成。逐帧qpos哈希与实际积分状态比对。ACTUAL STOCHASTIC PPO TRAINING与独立评价明确区分。加载真值只用于离线视频/指标，不输入Actor或奖励。

可见的50mm运动主要由冻结FF-VA基线跟踪参考完成，PPO只施加微小附加力；不能把整段大位移运动算成PPO独自学会。微小残差效果靠曲线和严格配对指标判断，不放大机器人实际位移。物理逐步动作有效率见`phase2c4_physical_action_effectiveness.csv`，接受的是已知命令差，不是直接测得的独立RL物理力。

训练命令实际运行过`train_phase2c4_ppo.py --seed 42/123/2026`；原路径存在时会拒绝覆盖，不要直接重跑这些命令写历史目录。用户复现实验须显式设置新的独立run及models目录、复制冻结配置；本轮没有额外授权更长训练。

程序已提供`--output-root`和`--model-root`用于将同一冻结实验写入新目录；参数入口实际执行过`--help`，以下完整重训示例没有再次执行，不计为本轮结果，也不会自动启动：
```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\miniconda\envs\stewart_rl\python.exe'
$newStudy = Join-Path '__PROJECT_ROOT__\results\runs' ('phase2c4_reproduce_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py scripts/train_phase2c4_ppo.py --seed 42 --output-root "$newStudy/training" --model-root "$newStudy/models"
foreach ($trialSeed in @(123,2026)) {
    & $py scripts/train_phase2c4_ppo.py --seed $trialSeed --output-root "$newStudy/training" --model-root "$newStudy/models"
}
```
同机CPU短采样的种子复现已测试，未额外重复4096决策全训练，也不声称跨硬件完全逐位复现。权重可以直接用Viewer加载，无需重训。

保护仍为原大工作区、每链相对C0总附加±5N/20N/s、原生±400N、weld及Jacobian和部分几何代理检查。887个网格凸包配对、103个连接组件排除和30个无原生硬止挡被动铰链的局限保留；无接触不等于全几何安全或实机认证。
