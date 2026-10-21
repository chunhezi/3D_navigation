"""[3D_PID] Minimal model-free 3D position P/PI controller.

All position values are in centimetres, time in seconds, and output forces in
newtons.  The module constants are intentionally kept here (rather than in the
global runtime configuration) so that an experiment can be tuned by editing a
single controller file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# [3D_PID] BEGIN: 3D P/PI controller parameters -- tune on real apparatus.
#
# Position unit: cm
# Time unit: s
# Force unit: N
# KP unit: N/cm
# KI unit: N/(cm*s)
# ============================================================================

# Safe commissioning default: gravity compensation only.  Enable one axis at
# a time after verifying the corresponding coordinate/force direction.
# KP_X = 1.6e-5
# KP_Y = 1.5e-5
# KP_Z = 1e-5

KP_X = 8e-6
KP_Y = 8e-6
KP_Z = 2e-5

ENABLE_INTEGRAL = True

KI_X = 3e-6
KI_Y = 4e-6
KI_Z = 3e-6

INTEGRAL_THRESHOLD_X_CM = 0.20
INTEGRAL_THRESHOLD_Y_CM = 0.20
INTEGRAL_THRESHOLD_Z_CM = 0.18
# [3D_PID] END: 3D P/PI tuning parameters.


@dataclass(frozen=True)
class PID3DResult:
    """One controller-step snapshot; every vector has shape ``(3,)``."""

    error_xyz: np.ndarray
    p_force_xyz: np.ndarray
    i_force_xyz: np.ndarray
    pid_force_xyz: np.ndarray
    integral_xyz: np.ndarray


# [3D_PID] BEGIN: 三个位置轴相互独立的最小 P/PI 实现。
# 这里有意不加入 D、速度反馈、输出限幅、slew-rate 或复杂 anti-windup；
# 电流安全限幅继续由 RuntimeLoop 的既有下发边界统一负责。
class PID3DController:
    """Three independent P/PI axes with per-axis integral separation."""

    def __init__(self) -> None:
        self.integral = np.zeros(3, dtype=float)

    @staticmethod
    def _vector3(value, *, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"{name} must contain exactly 3 values, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains a non-finite value")
        return arr.copy()

    def reset(self) -> None:
        """Clear the integral only at the start of a new experiment."""
        self.integral[:] = 0.0

    def step(
        self,
        current_position_cm,
        target_position_cm,
        dt_s: float,
        allow_integral_update: bool = True,
    ) -> PID3DResult:
        current = self._vector3(current_position_cm, name="current_position_cm")
        target = self._vector3(target_position_cm, name="target_position_cm")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt_s must be finite and positive, got {dt_s}")

        error = target - current
        kp = np.array([KP_X, KP_Y, KP_Z], dtype=float)
        ki = np.array([KI_X, KI_Y, KI_Z], dtype=float)
        thresholds = np.array(
            [
                INTEGRAL_THRESHOLD_X_CM,
                INTEGRAL_THRESHOLD_Y_CM,
                INTEGRAL_THRESHOLD_Z_CM,
            ],
            dtype=float,
        )

        if ENABLE_INTEGRAL and bool(allow_integral_update):
            # Integral separation is independent for each axis.  Outside the
            # threshold the existing integral is frozen, never cleared.
            update_mask = np.abs(error) <= thresholds
            self.integral[update_mask] += error[update_mask] * dt

        p_force = kp * error
        if ENABLE_INTEGRAL:
            i_force = ki * self.integral
        else:
            i_force = np.zeros(3, dtype=float)
        pid_force = p_force + i_force

        return PID3DResult(
            error_xyz=error.copy(),
            p_force_xyz=p_force.copy(),
            i_force_xyz=i_force.copy(),
            pid_force_xyz=pid_force.copy(),
            integral_xyz=self.integral.copy(),
        )
# [3D_PID] END: 最小三轴 P/PI 计算。


# [3D_PID] 为实验 PKL 提供精确的运行时参数快照，避免调参后无法复现实验。
def controller_parameter_snapshot() -> dict:
    """Return the exact module-level tuning used for the current experiment."""
    return {
        "KP_X": float(KP_X),
        "KP_Y": float(KP_Y),
        "KP_Z": float(KP_Z),
        "KI_X": float(KI_X),
        "KI_Y": float(KI_Y),
        "KI_Z": float(KI_Z),
        "ENABLE_INTEGRAL": bool(ENABLE_INTEGRAL),
        "INTEGRAL_THRESHOLD_X_CM": float(INTEGRAL_THRESHOLD_X_CM),
        "INTEGRAL_THRESHOLD_Y_CM": float(INTEGRAL_THRESHOLD_Y_CM),
        "INTEGRAL_THRESHOLD_Z_CM": float(INTEGRAL_THRESHOLD_Z_CM),
    }
