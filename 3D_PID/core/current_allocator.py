"""
文件功能：
    三维控制力到线圈电流的公共分配模块。

设计原则：
    1. 集中实现三维力 → current_control1 链路；
    2. 运动模式共用，避免重复代码；
    3. 不包含模式特有的控制律。

与其他模块关系：
    被运动模式调用。
    依赖 core.current_solver.current_control1。
    不依赖任何 motion_modes 具体模式文件，不 import motion.py。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.current_solver import current_control1

# 三维力 z 分量安全下限，避免 current_control1 中除零
_MIN_FZ = 1e-6


def _max_current_from_cfg(cfg: Optional[Any]) -> Optional[float]:
    """
    从配置对象或字典读取电流限幅值。

    若未配置则返回 None，表示不在本模块内限幅（由 main/Drive 层处理）。
    """
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        if "max_current" in cfg:
            return float(cfg["max_current"])
        return None
    if hasattr(cfg, "max_current"):
        return float(cfg.max_current)
    return None


def zero_current(num_coils: int = 6) -> np.ndarray:
    """
    输出零电流命令，用于安全停止或无效测量等场景。

    参数：
        num_coils: 线圈数量，默认 6。

    返回：
        全零电流向量。
    """
    return np.zeros(int(num_coils), dtype=float)


def force_3d_to_current(
    force_3d: np.ndarray,
    magnetic_model: Any,
    cfg: Optional[Any] = None,
    angle_scheme: int = 0,
    norm_mode: int = 0,
) -> np.ndarray:
    """
    将三维力向量转换为线圈电流命令。

    参数：
        force_3d: 三维力向量；
        magnetic_model: 磁学模型（current_control1 内部会使用）；
        cfg: 可选配置，含 max_current 时在本模块做限幅；
        angle_scheme: current_control1 角度方案，默认 0；
        norm_mode: current_control1 范数模式，0=二范数/最小二乘。

    返回：
        current: 线圈电流命令。

    """
    f = np.asarray(force_3d, dtype=float).reshape(-1)
    if f.shape[0] < 3:
        f = np.concatenate([f.flatten(), np.array([_MIN_FZ], dtype=float)])

    eps = _MIN_FZ
    if (not np.isfinite(f[2])) or abs(f[2]) < eps:
        f[2] = eps

    current = current_control1(
        f,
        magnetic_model,
        angle_scheme=int(angle_scheme),
        norm_mode=int(norm_mode),
    )
    current = np.asarray(current, dtype=float).reshape(-1)

    max_i = _max_current_from_cfg(cfg)
    if max_i is not None and max_i > 0:
        # 使用 clip 保留电流正负方向，避免 abs 超限后符号翻转
        current = np.clip(current, -max_i, max_i)

    if not np.all(np.isfinite(current)):
        return zero_current(current.shape[0] if current.size else 6)

    return current
