# 仿真参数调整结果

修改 `hierarchical_config.json`，原配置保留在 `hierarchical_config.before_tuning.json`。原 PI、重力补偿、质量/阻尼模型、采样间隔、初值、参考与扰动设置均未修改；没有启用实物运行。

| 参数 | 原值 | 当前值 |
| --- | ---: | ---: |
| observer_gain（L） | 4 | 12 |
| max_feedback_gain | 2 | 5 |
| gamma_learning（残差学习 Gamma） | 300 | 30000 |
| eta1、eta2（NN 学习率） | 5、5 | 100、100 |

滤波速率仍为 1、Phi 仍为 1、mu=0、NN 在 5 秒后启用、输出权重界为 4、隐层神经元为 1、随机种子为 2026。重新求解离线 SDP 得到 Kf≈-5、Kl≈-2.292951、Kh≈-1，不直接手改这些优化结果。

## 搜索方法

`tune_hierarchical_control.py` 固定读取调参前配置，使用 18 组离线设计和残差学习增益网格，再对前三名细调遗忘系数及 NN 学习率，共评估 81 个组合。训练指标为：

```text
0.5*(全程 RMSE / 原 PI 全程 RMSE)
+ 0.5*(扰动启用后 RMSE / 原 PI 扰动启用后 RMSE)
```

搜索要求数值有限、SDP 证书通过、没有触发权重边界投影；训练场景的峰值增量力模长不得超过原 PI 的 3 倍（约 18.0 μN）。67 个组合通过首轮约束。之后按训练得分依次检查补充场景，筛除了 11 个条目后选中当前配置；筛除包括数值发散或权重投影，因此这些补充场景参与了选择，不能称为独立无偏测试集。

搜索只模拟理想位置测量，没有执行器电流限制。这一力预算只是仿真选择约束，不是线圈安全性或实际可实现力的保证。

## 结果

默认 90 秒螺旋、相同周期扰动，全程三维 RMSE：

| 控制器 | RMSE（cm） |
| --- | ---: |
| 原 PI | 0.133802 |
| 分层学习，调参前 | 0.142601 |
| 分层学习，调参后 | 0.057819 |

相对原 PI 全程误差下降约 56.8%，扰动启用后的 RMSE 从 0.135851 cm 降为 0.048585 cm。代价是峰值增量力模长由约 6.0 μN 增至 17.1 μN，约为原 PI 的 2.86 倍。

补充检查的全程三维 RMSE：

| 场景 | 原 PI（cm） | 当前参数（cm） |
| --- | ---: | ---: |
| 0.5 倍扰动，90 秒螺旋 | 0.096805 | 0.053827 |
| 1.5 倍扰动，90 秒螺旋 | 0.177793 | 0.061289 |
| 定点，90 秒 | 0.095608 | 0.029141 |
| 阶跃，300 秒 | 0.107340 | 0.040753 |

以上场景均未触发权重投影。相同设置的粗学习和完整分层学习 RMSE 均约 0.057819 cm，NN 精学习没有明显额外收益；不能把改善全部归因于 NN，更不能据此认定论文算法普遍优于调优后的 PID。这里比较的是调参后的适配版和保留原参数的 PI，反馈强度也发生了变化。

## 文件与复现

- `output/comparison/`：当前参数的图、GIF、CSV 和参数快照。
- `output/comparison_before_tuning/`：本轮调参前的结果备份。
- `output/tuning/search_report.json`：全部候选、失败原因、目标函数与补充检查。
- `output/tuning/selected_config.json`：选中的参数。

```bash
python 3D_PID/simulation/compare_hierarchical_control.py
python 3D_PID/simulation/animate_position_pid.py --input 3D_PID/simulation/output/comparison --cases original_pi paper_hierarchical
```

重复搜索运行 `python 3D_PID/simulation/tune_hierarchical_control.py`，只写搜索报告和候选配置，不自动覆盖当前配置。恢复旧参数对比可运行：

```bash
python 3D_PID/simulation/compare_hierarchical_control.py --learning-config 3D_PID/simulation/hierarchical_config.before_tuning.json --output 3D_PID/simulation/output/reproduce_before_tuning
```
