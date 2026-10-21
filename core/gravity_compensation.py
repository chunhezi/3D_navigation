"""[3D_PID] Formula-based gravity compensation for ``stabilize_3d``.

The +z feed-forward force is the bead's gravity minus buoyancy, multiplied by
an empirical compensation factor::

    F_comp = V * (rho_object - rho_fluid) * g * compensation_factor
    V = (4/3) * pi * (diameter/2)^3

All quantities below use SI units, so the resulting force is in newtons.
This module is intentionally independent from ``magnetic_model``: experiments
can tune the 3D stabilizer without changing the gravity behavior of old modes.
"""

from __future__ import annotations

import math

import numpy as np


# [3D_PID] BEGIN: 可独立调整的重力-浮力等效力参数。
# 默认值逐项复现 magnetic_model.gravity_compensation() 的现有公式：
#   直径 1 mm、物体密度 7.5e3 kg/m^3、流体密度 0.978e3 kg/m^3、
#   重力加速度 9.8 m/s^2、经验上浮补偿系数 1.46。
# 若只想调重力补偿强弱，优先只修改 GRAVITY_COMPENSATION_FACTOR；
# 若磁珠或流体发生变化，再修改直径或密度参数。所有参数均使用 SI 单位。
BEAD_DIAMETER_M = 1.0e-3
OBJECT_DENSITY_KG_M3 = 7.5e3
FLUID_DENSITY_KG_M3 = 0.978e3
GRAVITY_ACCELERATION_M_S2 = 9.8
GRAVITY_COMPENSATION_FACTOR = 1.5
# [3D_PID] END: 重力补偿物理参数和经验系数。


# ============================================================================
# 三维常值前馈补偿
# ============================================================================

# X轴常值偏置补偿力，单位 N。
#
# 该值根据之前纯P定点实验的稳态数据估计：
#
#   X目标位置：x_ref = 0 cm
#   X稳态位置：x_ss ≈ -0.1787 cm
#   因此稳态误差：
#
#       e_x = x_ref - x_ss
#           = 0 - (-0.1787)
#           ≈ 0.1787 cm
#
# 当时X轴比例增益为：
#
#       Kp_x = 9.0e-6 N/cm
#
# 稳态时纯P控制器持续输出：
#
#       F_x = Kp_x * e_x
#           ≈ 9.0e-6 * 0.1787
#           ≈ 1.61e-6 N
#
# 因此可近似认为系统存在约 -1.61e-6 N 的X向等效恒定偏置力，
# 理论完全补偿值约为：
#
#       F_bias_x ≈ +1.61e-6 N
#
# 当前为避免一次补偿过大而影响瞬态响应，先采用约50%的补偿量：
#
#       X_FORCE_BIAS_N = +0.8e-6 N
#
X_FORCE_BIAS_N = 0e-6

# Y轴目前没有加入常值偏置补偿。
Y_FORCE_BIAS_N = 0.0


def calculate_bead_volume_m3(diameter_m=None) -> float:
    """Calculate spherical bead volume ``(4/3)*pi*(diameter/2)^3`` in m^3."""
    diameter = float(BEAD_DIAMETER_M if diameter_m is None else diameter_m)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(f"diameter_m must be finite and positive, got {diameter_m}")
    return (4.0 / 3.0) * math.pi * (diameter / 2.0) ** 3


def calculate_gravity_compensation_n(
    *,
    bead_diameter_m=None,
    object_density_kg_m3=None,
    fluid_density_kg_m3=None,
    gravity_acceleration_m_s2=None,
    compensation_factor=None,
) -> float:
    """Calculate the upward compensation force from explicit SI parameters.

    Any omitted argument is read from the current module-level tuning value,
    which also makes live parameter experiments and tests reflect immediately.
    """
    diameter = BEAD_DIAMETER_M if bead_diameter_m is None else bead_diameter_m
    rho_object = float(
        OBJECT_DENSITY_KG_M3
        if object_density_kg_m3 is None
        else object_density_kg_m3
    )
    rho_fluid = float(
        FLUID_DENSITY_KG_M3
        if fluid_density_kg_m3 is None
        else fluid_density_kg_m3
    )
    gravity = float(
        GRAVITY_ACCELERATION_M_S2
        if gravity_acceleration_m_s2 is None
        else gravity_acceleration_m_s2
    )
    factor = float(
        GRAVITY_COMPENSATION_FACTOR
        if compensation_factor is None
        else compensation_factor
    )
    values = (rho_object, rho_fluid, gravity, factor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("density, gravity and compensation factor must be finite")
    if rho_object <= 0.0 or rho_fluid < 0.0:
        raise ValueError("object density must be positive and fluid density non-negative")
    if rho_object <= rho_fluid:
        raise ValueError("object density must exceed fluid density for +z compensation")
    if gravity <= 0.0 or factor < 0.0:
        raise ValueError("gravity must be positive and compensation factor non-negative")

    volume_m3 = calculate_bead_volume_m3(diameter)
    effective_downward_force_n = volume_m3 * (rho_object - rho_fluid) * gravity
    return float(effective_downward_force_n * factor)


# [3D_PID] 兼容/展示用派生值，不再手写 4.89e-5。默认参数下约为
# 4.8861e-5 N。实际 get_gravity_compensation_force() 每次都会重新按上方
# 参数计算，因此修改 GRAVITY_COMPENSATION_FACTOR 后重启程序即可生效。
GRAVITY_COMPENSATION_N = calculate_gravity_compensation_n()


def get_gravity_compensation_force() -> np.ndarray:
    """返回三轴常值前馈补偿力：[X偏置, Y偏置, Z重力补偿]。"""
    gravity_force_z_n = calculate_gravity_compensation_n(
        bead_diameter_m=BEAD_DIAMETER_M,
        object_density_kg_m3=OBJECT_DENSITY_KG_M3,
        fluid_density_kg_m3=FLUID_DENSITY_KG_M3,
        gravity_acceleration_m_s2=GRAVITY_ACCELERATION_M_S2,
        compensation_factor=GRAVITY_COMPENSATION_FACTOR,
    )
    return np.array(
        [
            X_FORCE_BIAS_N,
            Y_FORCE_BIAS_N,
            gravity_force_z_n,
        ],
        dtype=float,
    )


def gravity_parameter_snapshot() -> dict:
    """Return formula inputs, intermediate values, and the resulting force."""
    volume_m3 = calculate_bead_volume_m3(BEAD_DIAMETER_M)
    effective_force_n = (
        volume_m3
        * (OBJECT_DENSITY_KG_M3 - FLUID_DENSITY_KG_M3)
        * GRAVITY_ACCELERATION_M_S2
    )
    compensation_n = calculate_gravity_compensation_n()
    return {
        "BEAD_DIAMETER_M": float(BEAD_DIAMETER_M),
        "OBJECT_DENSITY_KG_M3": float(OBJECT_DENSITY_KG_M3),
        "FLUID_DENSITY_KG_M3": float(FLUID_DENSITY_KG_M3),
        "GRAVITY_ACCELERATION_M_S2": float(GRAVITY_ACCELERATION_M_S2),
        "GRAVITY_COMPENSATION_FACTOR": float(GRAVITY_COMPENSATION_FACTOR),
        "BEAD_VOLUME_M3": float(volume_m3),
        "EFFECTIVE_GRAVITY_MINUS_BUOYANCY_N": float(effective_force_n),
        "GRAVITY_COMPENSATION_N": float(compensation_n),
        "X_FORCE_BIAS_N": float(X_FORCE_BIAS_N),
        "Y_FORCE_BIAS_N": float(Y_FORCE_BIAS_N),
    }


# [3D_PID] END: stabilize_3d 专用公式化重力补偿。
