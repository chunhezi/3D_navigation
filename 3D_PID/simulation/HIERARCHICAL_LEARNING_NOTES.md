# 分层在线学习抗扰算法：实现与适配说明

## 原文与实现范围

Pan Yu, Qiang Wu, Kang-Zhi Liu, *Hierarchical optimization-based online learning control for nonlinear systems with unknown disturbances*, Automatica 185 (2026), 112744，DOI：[10.1016/j.automatica.2025.112744](https://doi.org/10.1016/j.automatica.2025.112744)。在线发表日期为 2025-12-17。

已读取根目录用户提供的 `1-s2.0-S0005109825006429-main.pdf`，并检查第 3–5 页公式图像。当前实现是**过阻尼位置环适配版**，不是旋转倒立摆实验复现，也未经过实物验证。实物入口没有切换，原 PI 和重力补偿源文件未修改。

| 原文 | 文件/实现 |
| --- | --- |
| (4)、(5)、(6) | `../core/controllers/hierarchical_learning_controller.py`：观测器使用原反馈输入 uf，实际控制为 uf 减 EID 估计 |
| (7)、(12)、(13)、(22) | 残差线性支路与单隐层 tanh 支路；NN 只进入滤波支路 |
| (23)、(27) | sigma 微分状态实现指数遗忘积分，Gamma 和 Phi 取标量乘单位阵 |
| (34)、(35)、(36) | NN 和定理 2 敏感度梯度；输出权重达到边界时冻结 |
| (24)、(25)、(47) | `design_hierarchical_controller.py`：CVXPY/CLARABEL 离线 SDP |

粗学习从起始启用，精学习默认在第 5 秒后的实际采样时刻启用；仅粗学习组始终关闭 NN 输出。未启用可选正则化 (44)、(45)。每次实验有独立状态和固定随机种子。

## 模型适配与理论边界

现有二阶仿真保留质量与线性阻力：`m*p_ddot=F_incremental-c*p_dot+F_disturbance`。其状态为 `[p,v]`、输出为位置，因此 `C*B=0`。

若对该纯位置二阶对象直接要求原文 (24) 对应的 `P1*Bs=Cpsi.T` 且 `P1>0`，则左乘 Bs.T 得 `Bs.T*P1*Bs=Cpsi*Bs=0`，迫使 Bs=0，与非零位置输出映射不相容。因此不能直接取得这组正实性证书。

默认 `m/c≈0.00859 s`，小于 0.15 s 控制间隔，本次用过阻尼近似设计名义控制器：

```text
y_dot ≈ u_normalized + disturbance_normalized
y: cm, u_normalized: cm/s
F_incremental = (c/100)*u_normalized  # N
A=0, B=C=I
```

三轴共享标量离线增益，在线残差权重是三维矩阵。**对比的实际植物仍是原二阶模型**。该近似不是辨识结果；连续名义模型证书不覆盖惯性失配、采样、时变参考和实物。输出日志明确记录 `full_plant_certified=false`。

最初配置 L=4、AF=-2、BF=CF=1，故 Ae=-1；gamma=1，SDP 结果约为 `Kf=-2, Kl=-1.800824, Kh=-1, chi=3.000011`。当前调参后 L=12、反馈强度上界为 5，重新求解得到 `Kf≈-5, Kl≈-2.292951, Kh≈-1`。令 R=Bg=0 后，(25) 的独立负定块可消去，保留左上耦合块；epsilon4=0.01。

为限制任意尺度和采样下反馈强度，另外加入 Q1[0,0]=1、矩阵大小界、`-max_feedback_gain<=Kf<=-0.1` 和 `Kl<=-0.02`。这是本项目额外约束与数值选择，不是论文倒立摆参数，也不是离散稳定性证明。优化结果含 Q1/Q2/U、KYP 等式残差和不等式最大特征值；不可行或残差检查失败则报错，不回退到手写增益。

## 原文与数值实现的细节

1. 按 (12)、(16) 直接组装残差输入块 `[-B*Kh; BF*Kl]`。第 3 页 (15) 的 Ke=[Kl;Kh] 与 (17) 的 Bpsi=diag(-B,BF) 直接相乘会交换通道；代码遵循实际微分方程，明确记录此约定。
2. 定理 2 印刷为 Xi=I-h*h.T，多个隐层神经元时不同于通常逐元素 tanh 导数。实现保留原表达式，默认一个隐层神经元（三输入、三输出），此时两者一致。有限差分测试验证默认结构的冻结误差代理目标梯度，不代表证明动态闭环全局最优。
3. 观测器用过去一个区间的 uf 与已到达的两个位置端点作一阶保持积分；滤波器和 sigma 对过去残差作零阶保持解析更新。权重用显式梯度步，没有使用未来数据或真速度。
4. 连续律 (36) 到边界冻结，离散一步可能越界，因此增加仅在越界时向边界投影，记录触发次数。
5. 原文主要讨论原点调节；这里扩展成 uf=Kf*(x_hat-reference)，不加参考导数前馈。不能默认继承调节问题对任意轨迹的全部结论。

## 运行、公平性与输出

```bash
conda activate 3d_pid
python 3D_PID/simulation/compare_hierarchical_control.py
python 3D_PID/simulation/animate_position_pid.py --input 3D_PID/simulation/output/comparison --cases original_pi paper_hierarchical
```

控制参数见 `hierarchical_config.json`，仿真条件仍读 `config.json`。对比结果在 `output/comparison/`：

- original_pi：原位置 PI，同一示例扰动。
- paper_rough：残差粗学习＋EID＋状态反馈适配版。
- paper_hierarchical：同一设计＋NN 精学习适配版。

各组共享初始位置、初速度、参考、采样、二阶植物、重力前馈和平衡假设、扰动时序及幅值。控制器只接收位置、参考和 dt，不读取真扰动力、真速度。扰动力属于匹配扰动，不能用本实验声称验证了原文的非匹配抗扰能力。

新算法 CSV 的 compensation 是被减去的 EID 力估计，满足 `command=feedback-compensation+feedforward`，不是积分项；估计也可能包含名义模型失配。学习 CSV 保存残差、权重范数和精学习启用标志；summary.json 保存各组参数及最终权重；paper_implementation.json 保存配置与本地原文 SHA256。

调参前，90 秒螺旋加周期扰动的全程三维 RMSE：原 PI 约 0.133802 cm，粗学习约 0.142599 cm，分层学习约 0.142601 cm，未优于原 PI。调参后分层学习约为 0.057819 cm，同时峰值增量力增加；精学习相对粗学习仍没有显著改善。参数、搜索约束与补充检查见 [TUNING_RESULTS.md](TUNING_RESULTS.md)。这些只是名义仿真结果，仍需结合辨识、噪声和执行器限制评估。
