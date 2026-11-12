# Phase2A突变负载实验方案（冻结定义）

正式方案在运行正式矩阵前保存到`results/runs/phase2a_20261028_v1/formal_protocol_frozen.json`；原始先验方案为`configs/phase2a_disturbance_protocol.json`，各trial记录正式方案哈希。

## 模型和控制接口

入口`third_party/stewart_platform_mujoco/scene.xml`引用同目录`stewart_platform.xml`。12个执行器：ID0～5为leg1_ctrl～leg6_ctrl，标量支链力N，ctrl/force均±400；ID6～8为upper_f_x/y/z（N），ID9～11为upper_t_x/y/z（N m），后6个为upper_center的site-local直接力/力矩通道，原生ctrl/force范围均±500（不同单位，不代表这些极限已通过动力学安全验证）。

运行时独立创建nominal与actual MjModel，C0/C1/C2始终只接收名义几何和实际可观测状态/参考；外载真值仅实验器、日志和评价使用。世界外载用当前site旋转R的转置转换到局部通道，作用参考点为当时upper_center，不把世界原点力矩误当作site原点力矩。

每步在积分前计算site Jacobian的广义力、原生稀疏actuator_moment映射及独立mj_applyFT，三者交叉核验。外载通过独立后6个ctrl输入；前6个ctrl只由原控制器计算。全部运动由mj_step产生，未给live qpos/qvel赋值。

## 时间与窗口

dt=0.001 s，控制每步执行。第k个输入驱动[k*dt,(k+1)*dt)，结果CSV行step=k+1，time_s=(k+1)*dt。新阶跃在k=2000开始，k=4000撤销；事件分别在物理时间2和4秒记录，首个带载响应样本为2.001秒。该1 ms时间约定明确区别于GUI墙钟。

旧Phase1使用`external_command(condition,t_next)`，2<=t_next<4时输出局部[0,0,-10,.5,0,0]；初版按理想十进制时间推断为k=1999至3998，实际审计发现旧浮点累积使首个带载结果行为2001、末行为4000。初版三组保留为时间偏移诊断；独立v2复现按历史实际输入k=2000至3999执行，并在补跑前重新冻结协议。它单列legacy_local_z_roll，不与新世界坐标标准任务混同。

每次正式运行最多6秒仿真、180秒墙钟；保护触发仍保持原0.2秒受限保持逻辑并终止。没有无限扫描。

## 预检与正式矩阵

C0预检三条独立方向，每个方向按幅值递增做0.5秒加载，然后对最大通过幅值做1秒及2秒加载；最多15次，任何失败停止该方向扩大。预检与formal目录分开。

正式使用C0/C1/C2，同初态、模型、参考、采样和波形：

- zero_home、zero_roll、zero_pitch、zero_z_roll无外载对照。
- A_home_force：Home，世界Fz=-10 N。
- B_home_roll_positive/negative：Home，世界Tx=+/-0.5 N m。
- C_roll_force、C_pitch_force：已验收Roll/Pitch轨迹，世界Fz=-5 N。
- C_z_roll_composite：已验收Z+Roll轨迹，世界Fz=-5 N、Tx=+0.1 N m。
- legacy_local_z_roll：原局部坐标−10 N与+0.5 N m复现，单列历史对照。

实际正式矩阵由预检通过方向决定，上述完整版本为33次，未通过的方向不擅自补造结果。动态任务加载时应有非零参考速度，且必须先完成Home A/B实验。没有声称本轮覆盖任意Pitch力矩、所有脉冲或随机复合负载。

## 冻结评价

任务完成沿用原标准：完整6秒、无保护，最后至少0.5秒位置误差<0.75 mm，SO3姿态误差<0.1度。恢复标准更严格：位置<=0.25 mm且SO3姿态<=0.02度，同时连续保持0.3秒；加载和卸载窗口分别判定。加载前基线用加载前最后0.5秒。

位置误差为三维范数；姿态误差为Log(R_ref R_actual^T)的范数。Roll/Pitch分量只作显示/方向诊断，不用欧拉差值替代SO3总误差。容差只定义可解释评价，不改变原安全或成功门槛。

未恢复保存空数值并明确NOT RECOVERED DURING LOAD；提前停止保存RIGHT CENSORED，不能填0。恢复搜索必须在窗口内首次越出容差带之后开始，不能把加载后尚未响应的初始带内等待误记为恢复；整个窗口未越界标NO EXCURSION，恢复时间不适用。初版运行器的恢复实现曾误计初始延迟，原metrics.json保留，正式报告采用修复后独立复算结果，容差和保持时间均未调整。积分用所有原始样本矩形求和sum(error*dt)，单位mm s/deg s；同时报告实际观察长度和完整性，不能用短观察窗口的较低积分判定优越。

报告完整任务RMSE、扰动后峰值、与同控制器无扰动轨迹的额外偏移、加载和卸载误差积分、实际支链力/变化率/平方积分、饱和、weld、行程、条件数、代理间隙和保护。无扰动对照通过相同input_step及参考数组一致性检查后才能配对。

## 原始记录和复算

各trial含physics.csv.gz、integration_states.npz（完整mjSTATE_INTEGRATION）、initial_state.json、events.jsonl、config.json、plant.json、disturbance_profile.json、summary.json、metrics.json。外载指令、积分前旋转/作用点、世界外载和广义力逐步保存；积分后的qpos/qvel/ctrl、平台状态、实际力和约束保留1kHz。

`plant.json`原基类中的external_channel_policy来自旧Phase1说明；本轮以独立disturbance_profile和config定义为准，验收归档会保留该旧字段来源并加注Phase2A实际策略，不改变物理参数。

复算脚本从原始CSV重新计算SO3误差、积分和恢复，结果另存metrics_recomputed.json和汇总CSV，不覆盖旧阶段结果。视频用原trial现场RGB和相同state_index日志合成，核验qpos哈希、帧数、时间与外载，明确DYNAMIC DISTURBANCE RESPONSE EXPERIMENT。

## 运行命令

```powershell
cd __PROJECT_ROOT__
& D:\miniconda\envs\stewart_rl\python.exe scripts\run_phase2a_disturbance.py --stage all --run-root results\runs\一个新的独立目录
& D:\miniconda\envs\stewart_rl\python.exe scripts\compare_phase2a_controllers.py --run-root results\runs\phase2a_20261028_v1
& D:\miniconda\envs\stewart_rl\python.exe scripts\watch_phase2a_disturbance.py --trial results\runs\phase2a_20261028_v1\formal\A_home_force\C1
```

run命令会真实重新执行有界实验，不是观看入口；若只查看历史结果用compare/watch。已有输出目录不会被run覆盖。最终报告逐项列实际已运行命令。
