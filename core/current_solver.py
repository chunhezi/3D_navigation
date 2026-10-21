"""
文件功能：
    三维期望力到六线圈电流的求解器。

设计原则：
    本文件只负责电流分配数学求解（磁场方向 + 线性方程组）。
    不计算 LQR/ESO 控制力，不参与运动模式决策。
    不允许 import motion.py。

与其他模块关系：
    被 core/current_allocator.py 调用。
    magnetic_model 由 RuntimeContext 或调用方通过参数传入。

迁移说明：
    从 motion.py 的 current_control1() 原样迁移，算法不变。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# z 轴力分量安全下限，与 motion.py / current_allocator 一致
_MIN_FZ = 1e-6

# z 轴 3A 电流产生的磁场幅值基准（原 motion.current_control1 常数）
_BH_SCALE = 0.001159 * 2


def _resolve_mvm(magnetic_model: Any) -> float:
    """
    获取磁矩幅值 MVm。

    优先使用 magnetic_model.MVm（system_factory 创建时缓存）；
    否则调用 return_MVm()。
    """
    if hasattr(magnetic_model, "MVm"):
        return float(magnetic_model.MVm)
    return float(magnetic_model.return_MVm())


def current_control1(
    force_3d,
    magnetic_model: Any,
    angle_scheme: int = 0,
    norm_mode: int = 0,
    *,
    mvm: Optional[float] = None,
):
    """
    根据期望三维控制力求解线圈电流。

    参数：
        force_3d: 期望三维力向量 [Fx, Fy, Fz]，可 reshape 为 (3,)；
        magnetic_model: 磁模型对象，需实现 return_matrix()；
        angle_scheme: 角度配置方案，0=主方案（cal_theta2=2*fr/fz），1=备用方案；
        norm_mode: 归一化/求解策略，0=二范数/最小二乘 pinv，1=无穷范数；
        mvm: 可选，显式传入磁矩幅值；默认从 magnetic_model 读取。

    返回：
        current: 六维线圈电流命令，shape=(6,)。

    注意：
        该函数从原 motion.py 中迁移而来。
        迁移过程中不允许改变原数学逻辑。
    """
    f = np.asarray(force_3d, dtype=float).reshape(-1,)
    if f.shape[0] < 3:
        f = np.concatenate([f, np.array([_MIN_FZ], dtype=float)])

    # 防止除 0 / NaN（对应原 RuntimeWarning 与 SVD 不收敛）
    eps = _MIN_FZ
    if (not np.isfinite(f[2])) or abs(f[2]) < eps:
        f[2] = eps

    MVm = float(mvm) if mvm is not None else _resolve_mvm(magnetic_model)

    # 设置期望磁场幅值
    b_h = _BH_SCALE

    i = int(angle_scheme)
    j = int(norm_mode)

    if i == 0:
        theta1 = np.arctan2(-f[1], -f[0])
        fr = np.sqrt(f[1] ** 2 + f[0] ** 2)
        cal_theta2 = 2 * fr / f[2]
        theta2 = np.arctan(cal_theta2)
    else:
        theta1 = np.arctan2(-f[1], -f[0])
        fr = np.sqrt(f[1] ** 2 + f[0] ** 2)
        cal_theta2 = fr / f[2]
        theta2 = np.arctan(cal_theta2)

    B_h = np.array([
        np.sin(theta2) * np.cos(theta1),
        np.sin(theta2) * np.sin(theta1),
        np.cos(theta2),
    ])

    B_des = B_h * b_h
    M_theta = MVm * B_h * np.identity(3)

    tran = np.block([
        [np.identity(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), M_theta],
    ])

    des = np.concatenate((B_des, f))
    magnetic_matrix = magnetic_model.return_matrix()
    matrix_b = tran @ magnetic_matrix

    # 脏值检查：避免 pinv 的 SVD 不收敛
    if (not np.all(np.isfinite(matrix_b))) or (not np.all(np.isfinite(des))):
        return np.zeros((6,))

    # 特解
    i_p = np.linalg.pinv(matrix_b) @ des

    if j == 0:
        current = i_p
    else:
        # 无穷范数解（保留原实现）
        U, S, Vh = np.linalg.svd(matrix_b)
        nullspace = Vh[5, :].T
        i_g = np.reshape(nullspace, (6,))
        i_p = np.reshape(i_p, (6,))

        I_matrix = []
        for k in range(2):
            for n in range(4 - 2 * k):
                a1 = (i_p[k] - i_p[k * 2 + n + 2]) / (i_g[k * 2 + n + 2] - i_g[k])
                a2 = (i_p[k] + i_p[k * 2 + n + 2]) / (-i_g[k * 2 + n + 2] - i_g[k])
                a3 = (i_p[k + 1] - i_p[k * 2 + n + 2]) / (i_g[k * 2 + n + 2] - i_g[k + 1])
                a4 = (i_p[k + 1] + i_p[k * 2 + n + 2]) / (-i_g[k * 2 + n + 2] - i_g[k + 1])

                I1 = i_p + a1 * i_g
                I2 = i_p + a2 * i_g
                I3 = i_p + a3 * i_g
                I4 = i_p + a4 * i_g
                I_matrix.extend([I1, I2, I3, I4])

        I_infinite = [np.max(np.abs(v)) for v in I_matrix]
        min_norm = min(I_infinite)
        min_indices = [m for m, norm in enumerate(I_infinite) if norm == min_norm]
        I = I_matrix[min_indices[0]]
        current = np.reshape(I, (6,))

    return current
