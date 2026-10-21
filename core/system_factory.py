"""
文件功能：
    创建运行时上下文 RuntimeContext 及其中所需的全部公共对象。

设计原则：
    将原 motion.py 中的全局对象初始化迁移到这里。
    该文件只负责创建对象，不直接执行控制循环。
    不允许 import motion.py。

与其他模块关系：
    被 main.py 在启动时调用 create_runtime_context()。
    创建磁模型，保留配置与设备等共享资源的注入入口。
    创建的 RuntimeContext 被 mode_manager 与各 motion_modes 通过 self.ctx 访问。

[3D] 状态更新：
    磁珠模型本身负责根据视觉测量直接更新 position 和差分 velocity；
    不再创建或注入 Kalman / state_estimator。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from core import magnetic_model as model

from core.runtime_context import RuntimeContext


# ---------------------------------------------------------------------------
# 默认运行配置（与 motion.py 模块级常量对齐，算法参数不在此修改）
# ---------------------------------------------------------------------------

DEFAULT_RUNTIME_CFG: Dict[str, Any] = {
    # 磁模型离散时间步长（motion.py 中 dt=0.15）
    "model_dt": 0.15,
    # 控制循环周期（main.py 中 self.dt=0.03，由调用方覆盖）
    "dt": 0.03,
    # 磁珠初始位置与速度（cm, cm/s）
    "initial_position": [[8.0], [0.0]],
    "initial_vel": [[0.0001], [0.0001]],
    # 磁矩方向初值（弧度）
    "initial_theta": np.pi / 2,
    # 线圈电流限幅（A）
    "current_limit": 10.0,
    # ----- [3D] 直接差分速度参数 -----
    # velocity_dt_min_s：防止时间戳过近导致除零/尖峰
    # velocity_gap_reset_s：测量中断过长时不跨长间隔算速度，当前帧速度置 0
    "velocity_dt_min_s": 1e-4,
    "velocity_gap_reset_s": 0.5,
}


def merge_runtime_cfg(cfg: Optional[Any] = None) -> Dict[str, Any]:
    """
    将用户传入配置与 DEFAULT_RUNTIME_CFG 合并。

    参数：
        cfg: 字典或带 __dict__ 的对象；None 时使用全部默认值。

    返回：
        合并后的配置字典（新对象，不修改 DEFAULT_RUNTIME_CFG）。
    """
    merged = deepcopy(DEFAULT_RUNTIME_CFG)
    if cfg is None:
        return merged
    if isinstance(cfg, dict):
        merged.update(cfg)
        return merged
    if hasattr(cfg, "__dict__"):
        merged.update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})
        return merged
    merged["dt"] = cfg
    return merged


def create_magnetic_model(cfg: Optional[Dict[str, Any]] = None) -> Any:
    """
    创建磁珠运动学/磁学模型（原 motion.magnetic_model 初始化逻辑）。

    参数：
        cfg: 运行配置字典，见 DEFAULT_RUNTIME_CFG。

    返回：
        model.magnetic_motion_model 实例。
    """
    c = merge_runtime_cfg(cfg)
    model_dt = float(c.get("model_dt", c.get("dt", 0.15)))

    magnetic_model = model.magnetic_motion_model(
        dt=model_dt,
        position=deepcopy(c["initial_position"]),
        vel=deepcopy(c["initial_vel"]),
        theta=float(c["initial_theta"]),
        current_limit=float(c["current_limit"]),
    )

    # 缓存 MVm，供后续电流求解使用（等价于 motion.MVm = magnetic_model.return_MVm()）
    magnetic_model.MVm = magnetic_model.return_MVm()

    v_dt_min = float(c.get("velocity_dt_min_s", 1e-4))
    v_gap = float(c.get("velocity_gap_reset_s", 0.5))
    if v_dt_min <= 0.0:
        raise ValueError(f"velocity_dt_min_s must be > 0, got {v_dt_min}")
    if v_gap <= v_dt_min:
        raise ValueError(
            f"velocity_gap_reset_s must be > velocity_dt_min_s, got {v_gap} <= {v_dt_min}"
        )
    magnetic_model.velocity_dt_min_s = v_dt_min
    magnetic_model.velocity_gap_reset_s = v_gap

    return magnetic_model


def create_runtime_context(
    cfg: Optional[Any] = None,
    devices: Optional[Any] = None,
    recorder: Optional[Any] = None,
    *,
    magnetic_model: Optional[Any] = None,
) -> RuntimeContext:
    """创建运行上下文；磁模型可由调用方注入，其余公共资源保留扩展入口。"""
    merged_cfg = merge_runtime_cfg(cfg)

    if magnetic_model is None:
        magnetic_model = create_magnetic_model(merged_cfg)

    return RuntimeContext(
        cfg=merged_cfg,
        magnetic_model=magnetic_model,
        recorder=recorder,
        devices=devices,
    )
