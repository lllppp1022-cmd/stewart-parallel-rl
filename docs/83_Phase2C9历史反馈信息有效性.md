# Phase 2C-9 历史反馈信息有效性

使用真实预检观测作离线二分类。标签是名义Plant或单因素变化Plant，只给离线诊断器。每种条件每条轨迹抽取相同的决策时刻，每隔5决策采样，排除前10决策。按整个参考工况交叉留出，两个方向是Home训练→联合验证与联合训练→Home验证。岭回归归一化只拟合训练折，窗口长度1、5、10，当前帧完全共享。5帧的首尾间隔80 ms，10帧180 ms；窗口并不推迟最新观测。

| kind | value | held_out_reference | frames | balanced_accuracy |
| --- | --- | --- | --- | --- |
| mass | 1.03 | combined | 1 | 0.7080537 |
| mass | 1.03 | home | 1 | 1 |
| mass | 1.03 | combined | 5 | 0.7348993 |
| mass | 1.03 | home | 5 | 1 |
| mass | 1.03 | combined | 10 | 0.738255 |
| mass | 1.03 | home | 10 | 1 |
| mass | 1.1 | combined | 1 | 0.8557047 |
| mass | 1.1 | home | 1 | 1 |
| mass | 1.1 | combined | 5 | 0.9865772 |
| mass | 1.1 | home | 5 | 1 |
| mass | 1.1 | combined | 10 | 0.8288591 |
| mass | 1.1 | home | 10 | 1 |
| com_x | 0.0005 | combined | 1 | 0.6375839 |
| com_x | 0.0005 | home | 1 | 1 |
| com_x | 0.0005 | combined | 5 | 0.647651 |
| com_x | 0.0005 | home | 5 | 1 |
| com_x | 0.0005 | combined | 10 | 0.6442953 |
| com_x | 0.0005 | home | 10 | 1 |
| com_x | 0.0015 | combined | 1 | 0.738255 |
| com_x | 0.0015 | home | 1 | 1 |
| com_x | 0.0015 | combined | 5 | 0.7449664 |
| com_x | 0.0015 | home | 5 | 1 |
| com_x | 0.0015 | combined | 10 | 0.7483221 |
| com_x | 0.0015 | home | 10 | 1 |
| lag | 0.002 | combined | 1 | 0.5 |
| lag | 0.002 | home | 1 | 0.5100671 |
| lag | 0.002 | combined | 5 | 0.4966443 |
| lag | 0.002 | home | 5 | 0.4932886 |
| lag | 0.002 | combined | 10 | 0.5 |
| lag | 0.002 | home | 10 | 0.5100671 |
| lag | 0.006 | combined | 1 | 0.5 |
| lag | 0.006 | home | 1 | 0.5 |
| lag | 0.006 | combined | 5 | 0.5 |
| lag | 0.006 | home | 5 | 0.5033557 |
| lag | 0.006 | combined | 10 | 0.5 |
| lag | 0.006 | home | 10 | 0.5067114 |

研究准入规则由本次实现设定：both references safe; at least one attitude metric degrades >=5% in each; history balanced accuracy >=65% and gain >=5 percentage points in both held-out reference folds。

22 kg条件下，5帧在留出联合任务将准确率从85.57%提升至98.66%，10帧降至82.89%。留出Home的三者均100%，没有提升空间；Home姿态指标退化也不到5%。这是单折的可区分信息增益，不能据此断言历史在所有工况无效。每类仅两种任务和单一外载波形；没有跨参数值、时刻或方向的充分证据。未来需要增加能留出不同联合运动的组，并对单帧接近满分的工况单独设评价口径。

