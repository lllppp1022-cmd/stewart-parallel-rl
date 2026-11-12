# MuJoCo闭链约束和执行器审计

## 模型与许可

上游commit `b0195ba9f40f6e0876220be1712eb47f5fc24025`，MIT，作者声明及所有STL原样保留。模型不提供Python控制示例，本项目的审计/演示入口为新增代码。官方MJCF加载器直接加载scene.xml及相对路径assets，无模型转换。

- nq=43、nv=42、nu=12、nbody=45（含world）、njnt=37、neq=6。
- 30个hinge、6个slide、1个free；6个site-based weld，各约束相对6自由度，Home局部为6维受约束运动。
- 初始最大连接平移间隙2.31858748e-07m；efc_pos最大混合残差2.92060988e-07，其中平移和旋转不能统称为米。
- 平台质量20.0kg，主惯量[1.578160552589395, 0.7907915349877266, 0.7907857503367506]kg·m²；由网格和质量编译得到，不代表实物标定。
- 初始平台位置[0.0002058, 0.0051264, 1.1]m，姿态wxyz=[1.0, 0.0, 0.0, 0.0]。

## 执行器真实语义

前6个leg1_ctrl～leg6_ctrl以general joint transmission作用于joint_lu_1～6，gear=1、固定gain=1、无bias，控制输入和输出为力（N），ctrlrange及forcerange均[-400,400]。它们不是位置伺服，不能把期望米数直接写入ctrl。

后6个upper_f_x/y/z、upper_t_x/y/z经upper_center施加外部扰动，范围[-500,500]（力N或力矩N·m）。本阶段所有正式和演示试验均保持这些通道为0，未借用它们直接托举/倾转平台。传感器actuatorpos返回slide相对行程，不是约1m的支链总长。

## 最小动力学控制检查

独立scratch MjData按IK参考设置qpos，仅用于静态平衡查询。使用[B, J_eq^T][u,lambda]=qfrc_bias-qfrc_passive最小二乘计算6个静态平衡力，再以参考行程误差和速度误差作PD修正，Kp=5000N/m、Kd=200N·s/m，原±400N限幅保持。

Live MjData从原生qpos0初始化后仅通过ctrl和mj_step推进；scratch从不冒充动态对象。1秒保持、2秒五次缩放、2秒保持；幅度Z±5mm，Roll/Pitch±0.5°。初始/中点/终点先检查链闭合和行程。端点之间使用小幅行程与静态力插值，不宣称全路径精确笛卡尔跟踪或大范围安全。

## 步长、约束与碰撞限制

原XML使用implicitfast、默认2ms，weld solref=[0.0001,1]、强solimp。默认refsafe会将正格式约束时间常数限制到至少2×物理步长；因此1ms/0.5ms比较也改变了有效约束时间尺度，不能当成保持连续模型刚度不变的纯积分收敛证明。

原2ms控制7组的内部site方向间隙约2.23e-4rad，超过事前1e-4rad门槛；没有降低门槛。1ms/0.5ms各7组通过同一门槛，原失败CSV保留。实际平移残差亚微米，不能据此忽略方向约束。

**机器人所有网格contype=0、conaffinity=0。** 地面可见不等于机器人与地面有接触检测，自碰撞/腿碰撞也未启用。日志ncon=0不能作为无碰撞安全证明。该模型适用于本阶段小范围闭链动力学检查；更大运动、负载/关节真实限位、碰撞几何和实物模型仍需后续核验。

## Windows Viewer

实际使用原生launch_passive。动态过程由唯一Python物理owner写ctrl/mj_step，sync内部同步；对cam设置才短暂获取viewer.lock。早期逐步获取viewer.lock在手动X关窗时引发0xC0000005，崩溃栈保留；移除该竞争后实际手动关窗复核退出0并产生exit.json。增加journal落盘，缺失exit.json的进程不能标为完成。一次复核通过不等于所有显卡/版本无此问题。

参考：[MuJoCo XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、[官方passive viewer示例](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。
