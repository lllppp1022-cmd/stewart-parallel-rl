# Phase 2C-7V：Actor控制行为解释

## 真实控制链

Dynamic Reference → C4因果观测器+fourDlead+FF-VA → 加入二维PPO力矩请求 → 名义Jacobian转六支链力 → 单一共享预算器 → 原生执行器 → MuJoCo闭链动力学。

Actor是58输入、2输出的连续高斯策略；训练采样随机动作，评价用确定性均值，再Box裁剪至[−1,1]。请求Mx/My分别等于0.05×裁剪动作，单位Nm。它不修改平台目标、支链行程或外载。1kHz物理控制，每20ms才产生一次新Actor动作；共享预算逐1ms处理。传统C4补偿器历史代码名C3，报告名称与代码名称需要对应。

## 58维输入的准确顺序（零起始、右端不包含）

| name | start | end_exclusive | dimension | unit | scale | source |
| --- | --- | --- | --- | --- | --- | --- |
| SO3 error | 0 | 3 | 3 | rad | [0.001, 0.001, 0.001] | measurement/reference |
| angular velocity | 3 | 6 | 3 | rad/s | [0.02, 0.02, 0.02] | measurement |
| reference angular velocity | 6 | 9 | 3 | rad/s | [0.02, 0.02, 0.02] | known reference |
| XYZ error | 9 | 12 | 3 | m | [0.0002, 0.0002, 5e-05] | measurement/reference |
| linear velocity | 12 | 15 | 3 | m/s | [0.01, 0.01, 0.01] | measurement |
| relative slides | 15 | 21 | 6 | m | [0.06, 0.06, 0.06, 0.06, 0.06, 0.06] | measurement |
| slide velocity | 21 | 27 | 6 | m/s | [0.03, 0.03, 0.03, 0.03, 0.03, 0.03] | measurement |
| known command | 27 | 33 | 6 | N | [5, 5, 5, 5, 5, 5] | known command minus home force |
| causal observer | 33 | 39 | 6 | N,Nm | [20, 20, 20, 0.8, 0.8, 0.8] | causal estimate |
| base compensation | 39 | 45 | 6 | N | [5, 5, 5, 5, 5, 5] | same-state known controller |
| previous request | 45 | 47 | 2 | Nm | [0.05, 0.05] | previous command |
| previous accepted | 47 | 49 | 2 | Nm | [0.05, 0.05] | known governor |
| reference rotation | 49 | 52 | 3 | rad | [0.01, 0.01, 0.01] | known reference |
| reference Z/v/a | 52 | 55 | 3 | m,m/s,m/s2 | [0.05, 0.03, 0.03] | known reference |
| reference angular acceleration | 55 | 58 | 3 | rad/s2 | [0.02, 0.02, 0.02] | known reference |

观测每维最终裁剪±10。参考姿态是相对home旋转向量；SO3误差是R_actual R_ref^T的旋转向量，输入单位rad，图上Roll/Pitch为deg。观测器输出来自因果控制器估计，不是实际未知外力；控制命令、历史请求/接受量及参考均已知。这里不是58个独立传感器。MuJoCo中的位姿和速度是假定可测状态，未来实机还需传感/状态估计验证。本轮没有把真实drive增益、质量、滞后参数、外载值及事件时刻提供给Actor。

## 四任务×三种子的动作审计

| seed | task_id | requested_rms_Nm | accepted_rms_Nm | requested_accepted_rms_ratio | request_accept_error_rms_Nm | budget_projected_fraction | actual_vs_nominal_mapping_max_Nm | restoring_sign_fraction | damping_sign_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | F_roll | 0.0202609 | 0.0132515 | 0.654043 | 0.00783127 | 0.000214286 | 3.99322e-07 | 0.376786 | 0.471 |
| 123 | F_roll | 0.00881855 | 0.00870942 | 0.987625 | 0.000267202 | 0.000142857 | 2.4928e-07 | 0.625286 | 0.500357 |
| 2026 | F_roll | 0.00674368 | 0.00674336 | 0.999952 | 4.13186e-05 | 0.000142857 | 1.83524e-07 | 0.527714 | 0.474429 |
| 42 | F_pitch | 0.0191078 | 0.0133161 | 0.696892 | 0.00661063 | 0.0028125 | 2.02772e-07 | 0.178813 | 0.446688 |
| 123 | F_pitch | 0.00865417 | 0.00864173 | 0.998562 | 0.000241499 | 0.00275 | 1.86623e-07 | 0.536312 | 0.495625 |
| 2026 | F_pitch | 0.00553649 | 0.00553259 | 0.999294 | 8.93119e-05 | 0.00275 | 1.37961e-07 | 0.44425 | 0.557 |
| 42 | F_mix | 0.0194297 | 0.0134173 | 0.690556 | 0.00689461 | 0.0043125 | 1.97132e-07 | 0.0146875 | 0.487312 |
| 123 | F_mix | 0.00863269 | 0.00863106 | 0.999811 | 9.41835e-05 | 0.0026875 | 1.83479e-07 | 0.472 | 0.518875 |
| 2026 | F_mix | 0.00636633 | 0.00636065 | 0.999107 | 0.000208344 | 0.0045625 | 1.42257e-07 | 0.357812 | 0.535 |
| 42 | F_30 | 0.0176097 | 0.0130562 | 0.741422 | 0.00531609 | 0.00492857 | 1.79822e-07 | 0.1285 | 0.471643 |
| 123 | F_30 | 0.00692022 | 0.00691721 | 0.999565 | 9.27683e-05 | 0.00464286 | 1.05457e-07 | 0.603286 | 0.523929 |
| 2026 | F_30 | 0.00545769 | 0.00545562 | 0.99962 | 0.000105058 | 0.00492857 | 8.73837e-08 | 0.595286 | 0.447 |

requested_accepted_rms_ratio是两个波形RMS的比，不是逐时刻标量接受率。restoring/damping符号比例只做描述，不是闭环稳定性证明；混合动态参考时不能要求每时刻残差都逆误差或逆速度。

最重要的补充发现：日志accepted等于J_nominal^T×(真实命令−同状态无RL反事实命令)的Roll/Pitch分量。无RL反事实同样继承**真实前一时刻**的共享预算状态，因此本身可能受限速。它不是独立“关闭PPO”的另一条物理轨迹，也不必等于请求。

| task_id | seed | command_max_error_N | counter_base_max_error_N | executed_request_limited_fraction | counterfactual_no_rl_limited_fraction | counterfactual_rate_limited_fraction | counterfactual_amplitude_limited_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F_roll | 42 | 1.42109e-14 | 7.54952e-15 | 0.000214286 | 0.576929 | 0.576929 | 0 |
| F_roll | 123 | 1.42109e-14 | 7.54952e-15 | 0.000142857 | 0.128929 | 0.128929 | 0 |
| F_roll | 2026 | 1.42109e-14 | 7.54952e-15 | 0.000142857 | 0.000285714 | 0.000285714 | 0 |
| F_pitch | 42 | 1.42109e-14 | 7.21645e-15 | 0.0028125 | 0.57875 | 0.57875 | 0 |
| F_pitch | 123 | 1.42109e-14 | 7.54952e-15 | 0.00275 | 0.051375 | 0.051375 | 0 |
| F_pitch | 2026 | 1.42109e-14 | 7.54952e-15 | 0.00275 | 0.0028125 | 0.0028125 | 0 |
| F_mix | 42 | 1.42109e-14 | 7.16094e-15 | 0.0043125 | 0.575875 | 0.575875 | 0 |
| F_mix | 123 | 1.42109e-14 | 7.32747e-15 | 0.0026875 | 0.0063125 | 0.0063125 | 0 |
| F_mix | 2026 | 1.42109e-14 | 7.32747e-15 | 0.0045625 | 0.003 | 0.003 | 0 |
| F_30 | 42 | 1.42109e-14 | 7.32747e-15 | 0.00492857 | 0.545571 | 0.545571 | 0 |
| F_30 | 123 | 1.42109e-14 | 7.32747e-15 | 0.00464286 | 0.000714286 | 0.000714286 | 0 |
| F_30 | 2026 | 1.42109e-14 | 7.32747e-15 | 0.00492857 | 0.0045 | 0.0045 | 0 |

用已保存F_C3、F_C0、fourDlead请求、FF请求与RL映射力，逐1ms独立重建含RL及无RL两次投影。总命令与无RL反事实最大复算误差分别1.421e-14N、7.55e-15N。seed42/F_roll接受/请求RMS约0.654，但真实总请求投影比例仅0.0214%，无RL反事实限速比例57.69%。不能把RMS差直接解读为“35%控制权被预算裁掉”。对本轮全部12项反事实审计，没有幅值超5N案例，主要为20N/s变化率约束。

日志的实际几何Jacobian与控制器名义Jacobian不同；用前者复算接受力矩最大差3.9932e-07Nm，属于此处几何近似/状态差异，不能冒充两个Jacobian严格一致。接口仍保留名义映射，不改变控制算法。

## 加载、持续、卸载和恢复

| seed | event | window | requested_along_load_Nm | accepted_along_load_Nm | first_recorded_request_change_ms |
| --- | --- | --- | --- | --- | --- |
| 42 | load | pre100 | -0.0357099 | -0.0211003 | 10 |
| 42 | load | post20 | -0.0362919 | -0.0337659 | 10 |
| 42 | load | post40 | -0.0365582 | -0.03098 | 10 |
| 42 | load | post60 | -0.0365897 | -0.0288488 | 10 |
| 42 | load | post100 | -0.0365112 | -0.0259835 | 10 |
| 42 | load | sustained | -0.0330002 | -0.0212962 | 10 |
| 42 | unload | pre100 | -0.0266828 | -0.0216289 | 10 |
| 42 | unload | post20 | -0.0244908 | -0.00819749 | 10 |
| 42 | unload | post40 | -0.0236922 | -0.0114677 | 10 |
| 42 | unload | post60 | -0.0234021 | -0.0135175 | 10 |
| 42 | unload | post100 | -0.0231184 | -0.0162957 | 10 |
| 42 | unload | recovery | -0.0121153 | -0.0120008 | 10 |
| 123 | load | pre100 | -0.0221071 | -0.0202691 | 10 |
| 123 | load | post20 | -0.0202698 | -0.0201799 | 10 |
| 123 | load | post40 | -0.0184753 | -0.0184303 | 10 |
| 123 | load | post60 | -0.0172731 | -0.0171442 | 10 |
| 123 | load | post100 | -0.0162774 | -0.0162 | 10 |
| 123 | load | sustained | -0.00970863 | -0.00970317 | 10 |
| 123 | unload | pre100 | -0.00122709 | -0.00122709 | 10 |
| 123 | unload | post20 | -0.00264855 | -0.00256834 | 10 |
| 123 | unload | post40 | -0.00462453 | -0.00458443 | 10 |
| 123 | unload | post60 | -0.00587064 | -0.00574673 | 10 |
| 123 | unload | post100 | -0.00673802 | -0.00666368 | 10 |
| 123 | unload | recovery | -0.00348364 | -0.00348364 | 10 |
| 2026 | load | pre100 | -0.00788551 | -0.00788551 | 10 |
| 2026 | load | post20 | -0.00791209 | -0.00791209 | 10 |
| 2026 | load | post40 | -0.00795841 | -0.00795841 | 10 |
| 2026 | load | post60 | -0.00794193 | -0.00785446 | 10 |
| 2026 | load | post100 | -0.0078181 | -0.00776561 | 10 |
| 2026 | load | sustained | -0.00422726 | -0.00422477 | 10 |
| 2026 | unload | pre100 | 0.000247473 | 0.000247473 | 10 |
| 2026 | unload | post20 | 0.000759865 | 0.000759865 | 10 |
| 2026 | unload | post40 | 0.00097949 | 0.00097949 | 10 |
| 2026 | unload | post60 | 0.00114567 | 0.00112872 | 10 |
| 2026 | unload | post100 | 0.00128941 | 0.00127924 | 10 |
| 2026 | unload | recovery | 0.00379214 | 0.00379167 | 10 |

上述along_load按该任务真实外载方向投影，仅用于离线解释，不输入Actor。F_roll事件位于决策栅格中间，首次记录请求变化为10ms；它可能也受正在变化的参考影响，不能称10ms外力估计延迟。其他任务/事件的实际机会及首变化见policy_behavior.csv。20/40/60/100ms是累计窗口均值，不是一个独立瞬时采样点。物理图有±100ms放大和每20ms决策标记。

seed42在加载前已存在明显偏置，卸载后前100ms和恢复窗仍保持与此前相同方向的补偿分量。加载收益与卸载代价的非对称，与该偏置、动态参考以及同状态反事实限速一致；目前没有在相同物理状态做去偏置/替换动作的受控闭环干预，所以不能断言其为唯一因果机制。也没有证据把所有退化都归因于传统控制器与PPO竞争。

## 局部输入敏感性

每seed使用最后一个真实rollout的64个等间隔观测；对各输入施加±0.01归一化单位，冻结Actor中央差分。输出按裁剪动作×0.05Nm计算。64×58×2次单seed输入扰动，没有运行新训练。

| seed | group | axis | mean_l2_sensitivity_Nm_per_normalized_unit | label | samples |
| --- | --- | --- | --- | --- | --- |
| 123 | previous request | Mx | 0.0113481 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 123 | causal observer | Mx | 0.0102261 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 2026 | previous accepted | Mx | 0.0093248 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 42 | previous accepted | My | 0.00781942 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 42 | previous accepted | Mx | 0.0075354 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 123 | previous request | My | 0.00697155 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 42 | previous request | My | 0.00621024 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 42 | slide velocity | Mx | 0.00609777 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 123 | previous accepted | My | 0.00606773 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 2026 | previous request | Mx | 0.00587428 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 2026 | XYZ error | My | 0.00400735 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |
| 2026 | previous accepted | My | 0.0039193 | LOCAL INPUT SENSITIVITY / OFF-MANIFOLD DIAGNOSTIC | 64 |

结果表明历史请求/接受量、因果估计等会改变输出，但组维数及归一化尺度影响L2幅值，不能当成无偏特征重要性排序。输入维之间有运动学关系，单维扰动通常离开真实状态流形。这里只能证明局部数值敏感性，不能证明Actor已经辨识或估计未知外力矩，也不能据此给出闭环因果贡献。

全部36张物理时序图位于`results/runs/phase2c7v_20261108_v1/behavior_plots`。经典反馈图线来自独立、同任务真实轨迹；增强视频中的same-state PD第三曲线是离线公式诊断且未执行，二者不可混淆。
