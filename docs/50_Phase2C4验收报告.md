# Phase2C-4 验收报告

2026-11-05，工程`__PROJECT_ROOT__`，实验`__PROJECT_ROOT__\results\runs\phase2c4_20261105_v1`。

## 真实完成及研究判断

**PILOT INCONCLUSIVE — REVISE TRAINING DESIGN**。本轮完成环境、完整零动作等价、随机预检、真实三种子PPO、24次独立闭环评价、真实训练/评价RGB视频及受限Viewer。没有预设PPO应胜出，原FF-VA和XY代价保留。

| seed | decisions | physical_steps | updates | optimizer_steps | complete_episodes | safety_terminations |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 4096 | 204800 | 16 | 640 | 13 | 0 |
| 123 | 4096 | 204800 | 16 | 640 | 13 | 0 |
| 2026 | 4096 | 204800 | 16 | 640 | 13 | 0 |


实际训练12288决策、614400个1ms积分，256烟雾已包含42号预算，没有免费重复前缀。预检、评价、GUI和自动测试是额外独立成本，不混入训练样本。

独立任务各方法平均相对改善（正为改善，基线FF-VA）：

| method | 加载IAE改善百分比 | 卸载IAE改善百分比 | 全程Z_RMSE改善百分比 |
| --- | --- | --- | --- |
| PPO123 | 1.592162 | 2.07850198 | 2.36660351 |
| PPO2026 | 0.117643954 | 1.75133455 | 1.51383907 |
| PPO42 | 2.80357901 | -1.40680357 | -0.591773354 |


绝对收益及全部24行在49号报告和CSV中。通过环境/优化验收不等于控制性能成功。多工况门槛通过=False，实际执行=False；没有绕过门槛启动随机化训练。

## 工程验收与追溯

完整pytest真实输出：`322 passed, 1 xfailed, 2 warnings in 322.71s (0:05:22)`。原历史xfail单列，未计普通通过。完整日志`results/phase2c4_pytest.log`。另有训练前12项检查通过；故障注入只用于单元测试终止语义，不计真实机器人安全失败。

7199个冻结历史文件SHA未变，结果`history_preservation.json`。全部实际模型哈希见training_summary及每个ppo_update_events；无覆盖历史模型。原模型/前馈参数/预算/工作区未放松。原始CSV可独立复算；`reward_training_independent.json`验证全部物理奖励及每决策积分。评价Z与预算用独立向量公式复算，安全审计通过=True。

首轮终止观测指针错误及其失败物理日志归档，修复后重新完整验证再训练。没有用直接写live qpos隐藏问题。Reset创建新的独立物理对象，GUI暂停/重置/退出均不额外生成策略训练动作。

过程偏差如实记录：启动门禁先完成在线公式敏感性、物理分量积分与回合奖励核对；更严格的独立向量公式复算在冒烟首轮期间补完，随后又复算全部训练档案，均通过，而非全部在首次采样前完成。未据此重训或追加预算。收尾补齐Gym `info.is_success`元数据，沿用原最后0.5秒<0.75mm/<0.1deg判据，不改变训练观测、奖励、动作、物理或done。详见`implementation_notes.json`。

## 可直接观看与复盘

```powershell
Set-Location '__PROJECT_ROOT__'
$py = 'D:\miniconda\envs\stewart_rl\python.exe'
# 新建独立日志，真实动力学运行最后checkpoint；不是训练现场回放
& $py scripts/view_phase2c4_policy.py --method PPO --model models/phase2c4/seed42/final.zip
# 原FF-VA对照
& $py scripts/view_phase2c4_policy.py --method FF-VA
# 只看TensorBoard，不启动训练
& $py -m tensorboard.main --logdir results/runs/phase2c4_20261105_v1/training --host 127.0.0.1 --port 6034
```
本轮已实际启动6034并验证HTTP200；若已运行，不重复占端口。暂停只暂停物理时钟；Reset显式新建MjData和新episode，不在运动中写qpos；Stop关闭、归档。完整15秒结束后停留显示，墙钟600秒上限。受限CLI只提供固定已预检任务，不允许界面扩载。

有界GUI开/关验证实际命令：
```powershell
& $py scripts/view_phase2c4_policy.py --method FF-VA --validate-steps 2000 --compare-headless
```
新PPO入口的有界加载检查命令见运行清单。默认完整GUI交互未作为额外训练样本。窗口为单线程Qt＋MuJoCo Renderer，物理与显示所有权一致；不是原生GLFW Viewer。历史2C3B窗口保留。

以下PPO加载命令分别实际运行至2000和8000物理步（8000步已覆盖真实50mm上升及顶部）：
```powershell
& $py scripts/view_phase2c4_policy.py --method PPO --model models/phase2c4/seed42/final.zip --validate-steps 8000
```
这是GUI预算截断的部分评价，不算完整15秒成功。完整结果使用正式24条全轨迹评价。


实际训练视频`assets/videos/stewart_phase2c4_ppo_training.mp4`，独立评价视频`assets/videos/stewart_phase2c4_ppo_evaluation.mp4`，均按现场RGB后期合成、2倍播放，逐帧qpos SHA对齐完整积分档案并验证解码/真实50mm位移。训练episode跨256更新边界按每帧真实policy_version展示，未假设整个episode策略固定。训练末尾未完成prefix不做完整成功视频。

只做离线分析的命令实际执行：`analyze_phase2c4_training.py`、`audit_phase2c4_reward.py --training`、`render_phase2c4_video.py training/evaluation`。不自动再次learn。所有新代码、模型、配置、报告和原始轨迹列入`results/phase2c4_artifact_manifest.json`。

## 25项问题

### 1. 50mm环境真实有效？

Gym检查、完整15000步零动作配对、真实随机动作和GUI对照均通过；真实ctrl和mj_step，非qpos动画。

### 2. Actor/Critic优化？

三个种子实际48次rollout更新；全部参数变化非零，checkpoint可重载。

### 3. 零动作严格复现？

完整积分、observer/governor/lag/12ctrl逐位一致，Reset前50步再次一致。

### 4. 奖励与目标一致？

独立复算、单项敏感性及失败折扣上界通过；但平方误差全程目标与1秒IAE并不完全相同，不能预设学会抗扰。

### 5. 识别加载/卸载？

123号在所测加载/卸载窗口均改善、请求呈异号；这支持有效反馈响应，仍不足以证明辨识了未知负载。42号存在相反窗口取舍，Actor没有外载标志。

### 6. 动作方向/幅值？

请求±0.1N；E1的123号约95%样本与当前Z误差反向，42号窗口平均请求同为负、卸载退化。实际接受和raw/Box另行审计，不简单按外载正负判动作。

### 7. 经常被预算抑制？

见动作表，预算投影率范围0.000551758～0.000688477，与高斯Box裁剪率不同。

### 8. 加载IAE改善？

42和123在两个独立组合均改善；2026一个改善、一个恶化。绝对变化全列49号报告。

### 9. 卸载IAE改善？

123和2026在两个独立组合均改善，42在两个组合均恶化；加载改善不能替代卸载结论。

### 10. 全程Z改善？

123和2026在两个独立组合均改善，42均恶化；平均百分比见表，绝对收益很小。

### 11. XY是否恶化？

42的独立任务XY峰值轻微增大，123和2026轻微减小；原FF-VA横向耦合仍存在，未解决。

### 12. 姿态是否恶化？

42的SO(3)轻微增加，其余两种子轻微降低，绝对变化很小；RPY日志保留，未触原姿态保护。

### 13. 控制力变化？

确定性PPO的峰值变化率相对FF-VA仅有极小变化，均未超过冻结的10%不利门槛；实际力/能量单列，附加20N/s不等于全力20N/s。

### 14. 安全失败？

训练安全终止0次，评价成功24/24；测试故障注入不冒充真实失败。

### 15. 相对C4？

匹配任务的PPO全程Z误差均更小，但主要收益来自FF-VA，不能全部算成PPO贡献。

### 16. 相对fourDlead？

匹配任务全程Z误差均更小，同样主要来自FF-VA，不能替代最强基线对照。

### 17. 相对FF-VA？

PILOT INCONCLUSIVE — REVISE TRAINING DESIGN；123通过共同门槛，2026加载收益不一致，42有卸载/全程退化，未形成跨种子稳定优势。

### 18. 三个种子一致？

共同门槛{'42': False, '123': True, '2026': False}，保留个体差异，不筛种子。

### 19. 已见和独立差异？

训练内train_fixed与E1/E2分开，加载时序/幅值有限泛化，非大范围鲁棒证明。

### 20. 单帧是否足够？

首轮仅验证可采样优化；包含因果observer状态，未证明一般任务Markov充分性。

### 21. 是否加历史？

159维接口保留但未自动运行；需在同物理预算的独立消融中检验，不能预设收益。

### 22. 是否多工况？

冻结门槛通过=False，实际执行=False；不达门槛就停止。

### 23. 20/50Hz？

本轮仅20Hz。可作为后续独立假设，同物理暴露、折扣/GAE时间尺度匹配，当前未验证。

### 24. 扩Fx/Fy？

未获本轮验证，不直接扩大Actor；先诊断名义耦合、控制权限与预算，再独立预检。

### 25. 扩大正式预算？

本轮不自动扩大；按最终决策先处理设计问题或停止扩展，不能用追加训练隐藏负结果。


## 未完成与边界

未开展十万步训练、SAC/TD3/BC、全6DoF残差、Fx/Fy探索、实机辨识/部署或安全认证。未把旧3mm策略直接放到50mm。159维历史/20vs50Hz/更广泛随机化如未执行就保留为未来假设，不能写成结果。新的独立任务仅两个组合，不能泛化为真实液压系统或所有负载扰动。

保护仍为原大工作区、每链相对C0总附加±5N/20N/s、原生±400N、weld及Jacobian和部分几何代理检查。887个网格凸包配对、103个连接组件排除和30个无原生硬止挡被动铰链的局限保留；无接触不等于全几何安全或实机认证。

本轮完成后暂停，不自动追加算法或训练预算。
