# Phase 2C-9 未知动力学泛化评价

完成了同一参数与外载下的闭环诊断。下表只比较22 kg工况的冻结传统控制与50 Hz经典附加补偿。经典方法使用因果姿态和角速度PD，加泄漏积分（1/s，Ki=10，积分限幅±0.025 N·m），饱和时不增加积分；总残差仍经原二维动作和共享预算。系数为本轮工程诊断设定，未做独立调参。

| method | reference_id | kind | value | success | steps | load_iae_deg_s | unload_iae_deg_s | so3_peak_deg | omega_peak_rad_s | xyz_rmse_mm | force_peak_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_zero_residual | home | mass | 1.1 | True | 15000 | 0.001570675 | 0.001576328 | 0.01140989 | 0.004585232 | 0.04787051 | 93.6508 |
| frozen_zero_residual | combined | mass | 1.1 | True | 15000 | 0.009458466 | 0.004058863 | 0.01461761 | 0.006787186 | 0.1402558 | 95.23448 |
| classical_leaky_pi_diagnostic | home | mass | 1.1 | True | 15000 | 0.001540809 | 0.00154173 | 0.01135009 | 0.004583287 | 0.04777886 | 93.6508 |
| classical_leaky_pi_diagnostic | combined | mass | 1.1 | True | 15000 | 0.009419437 | 0.003974824 | 0.01450911 | 0.006786836 | 0.1394604 | 95.23358 |

独立泛化矩阵未开放。这里的两个工况来自准入任务，尚无未见参数、方向、时刻或组合上的策略泛化结论。逐条1 ms日志复算了SO(3)、IAE、角速度、位置与执行器指标。

