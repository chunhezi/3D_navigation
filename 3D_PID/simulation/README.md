# 三维位置 PI 基线仿真

新增了论文 DRNN-EID 抗扰算法的项目适配版。原 PI 基线入口保持不变；三组同扰动对比运行：

```bash
python 3D_PID/simulation/compare_hierarchical_control.py
```

输出在 `output/comparison/`：`original_pi` 为原 PI，`paper_rough` 为仅粗学习，`paper_hierarchical` 为分层学习。原文公式对应与适配限制见 [HIERARCHICAL_LEARNING_NOTES.md](HIERARCHICAL_LEARNING_NOTES.md)。

在仓库根目录运行：

```bash
conda activate 3d_pid
python 3D_PID/simulation/position_pid_sim.py
```

不需要相机、MVS SDK、Arduino、泵或电流源。结果保存在本目录 `output/`：

- `trajectory.png`：三维参考和实际轨迹。
- `response.png`：三轴位置、误差、磁力指令和示例扰动力。
- `baseline_states.csv` / `baseline_forces.csv`：无外加扰动的数据。
- `demo_disturbance_states.csv` / `demo_disturbance_forces.csv`：示例扰动的数据。
- `summary.json`：实际使用的参数快照与误差指标。

初始位置由 `config.json` 设置，默认 `[0, 0, 0.5] cm`，初速度为零。默认运行 90 秒，采样周期 0.15 秒（对应 `main.py` 当前设置）。采样周期是仿真配置值，不会随 `main.py` 修改自动改变。

将已有 CSV 结果制作成 GIF 动图：

```bash
python 3D_PID/simulation/animate_position_pid.py
```

生成 `output/trajectory.gif`，左侧动态展示目标点和两种情景的三维轨迹，右侧展示三维位置误差。默认 7.5 倍速、15 帧/秒，可通过 `--speed`、`--fps` 修改；`--input` 可指定其他仿真结果目录。动画读取保存的数据，不会重新计算或改变仿真参数。

## 复用哪些原代码

直接加载以下独立模块，避免导入硬件/GUI 初始化代码：

| 原文件 | 仿真使用内容 |
| --- | --- |
| `core/controllers/pid_3d_controller.py` | 原位置控制器、P/I 增益及积分分离阈值 |
| `core/reference_generator_3d.py` | 原定点、阶跃、螺旋参考生成器 |
| `core/gravity_compensation.py` | 磁珠直径、密度、流体密度、重力与补偿力 |

原名为 PID 的控制器实际只有 **P 和 I，没有 D 项**。仿真保留其行为：误差在阈值内才更新积分，超过阈值时保留已有积分。控制输入为原 PI 输出加原重力前馈。

默认使用项目当前参考（目前为 60 秒、3 圈、半径 0.75 cm、Z 从 0.5 到 1.2 cm 的螺旋，随后保持终点）。与运动模式一致，每次控制先推进参考，再计算位置误差，因此首次控制的螺旋参考时钟为 `dt`。终端记录仅观测状态并保持最后参考，不再执行控制。

切换参考或延长阶跃实验：

```bash
python 3D_PID/simulation/position_pid_sim.py --reference fixed --output 3D_PID/simulation/output/fixed
python 3D_PID/simulation/position_pid_sim.py --reference step --duration 300 --output 3D_PID/simulation/output/step
```

阶跃保持计数沿用原来的 500 个有效采样，0.15 秒采样对应约 75 秒一个点。每个仿真采样均视为有效测量，无丢帧和测量噪声。

## 被控对象与适用范围

内部采用 m、s、N，进入控制器前换算为 cm。提供两种明确区分的模型，均未经过实物辨识。

默认 `plant_mode: "equilibrium"`：以“现有前馈对应悬浮平衡指令”为工作点假设，测试该工作点附近的位置环：

```text
F_command = F_PI + F_feedforward
F_equilibrium_command = F_feedforward    # 工作点假设，不是实际重力
m * p_ddot = (F_command - F_equilibrium_command) - c * p_dot + F_disturbance
```

这里仍假设增量指令到增量实际力的增益为 1，平衡指令暂设为空间常量。它不等于已辨识的磁力模型，也不保证能复现实物精度。该模型不计算磁场、电流和真实磁力，不能解释为什么实物需要 1.5 倍经验前馈；它也不把这 1.5 倍解释成真实重力。

原先版本对应 `plant_mode: "ideal_force"`，假设总期望力等于总实际磁力，并独立使用理论净重力：

```text
m * p_ddot = F_PI + F_feedforward - W_effective * ez - c * p_dot + F_disturbance
m = rho_object * V
W_effective = (rho_object - rho_fluid) * V * g
c = 3 * pi * viscosity * diameter
```

采用球体 Stokes 线性阻力近似；默认动力黏度 `0.0485 Pa·s` 参考 `magnetic_model.py/update_fx` 中 `50e-6 * 970` 的数值，并在这里按 SI 单位解释。这是可替换的名义参数，**尚未经过实验辨识**。旧模型另一处 `resistance` 的数值不同，仿真没有混用二者。默认质量约 `3.927e-6 kg`、阻力系数约 `4.571e-4 N·s/m`。

控制力在一个采样周期内保持不变，用线性方程的解析离散解更新位置和速度，避免 0.15 秒显式欧拉积分对快速阻尼动力学造成数值不稳定。

两种模型均暂未包含六线圈磁场/电流分配、电流限幅、磁矩姿态变化、相机延迟、壁面接触和非线性流体效应。因此它们用于检验位置反馈结构和比较控制器，不能据此认定实物可达到相同误差或磁力。

原重力补偿系数目前为 **1.5**。只有在 `ideal_force` 的理想执行器假设下，模型才产生约 `16.73 µN` 的向上余力和约 `0.837 cm` 的 Z 偏差。该结果说明模型假设与已报告的实物表现不符，不能用来认定实物参数有误。当前默认的平衡点模型通过显式工作点假设去掉这一预置失配，并未对真实执行器完成标定。

复现旧版结果：

```bash
python 3D_PID/simulation/position_pid_sim.py --plant-mode ideal_force --output 3D_PID/simulation/output/ideal_force_check
```

两种模型都保留原 PI 增益、积分分离和 1.5 倍重力前馈，未修改实物代码。检查结论及后续辨识所需数据见 `MODEL_AUDIT.md`。

## 扰动与参数修改

默认同时运行无外加扰动基线和示例周期扰动。后者从 20 秒起允许扰动，在全局采样计数的弱阶段结束后施加强阶段力。弱/强阶段沿用 `int(6/dt)`、`int(2/dt)` 的计数方式；默认 40/13 个采样，总周期 7.95 秒。弱阶段设为零，强阶段默认 `[2, -1, 0] µN`。这只是控制器对比输入，**没有泵流量到磁珠受力的标定含义**，也不是串口泵控制程序。

`config.json` 可改运行时间、采样周期、初值、黏度和示例扰动力。总时间必须为采样周期的整数倍。原控制增益、参考路径和重力补偿读取项目源文件，修改那些文件会同时影响实物控制和后续仿真，请注意区分；本次仿真未修改它们。

如在新环境运行，安装本目录 `requirements.txt`。仅导出数据可加 `--no-plots`。运行数值模型测试：

```bash
python -m unittest discover -s 3D_PID/simulation -p 'test_*.py' -v
```
