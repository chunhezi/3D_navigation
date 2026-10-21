"""
文件功能：
    核心公共模块包入口，存放运动模式共享的运行时对象与底层工具。

设计原则：
    该包只提供公共对象引用与底层函数，不包含具体运动模式逻辑。

与其他模块关系：
    被 motion_modes 下的模式文件和 main.py 引用。
    各 motion 文件之间仍不允许相互 import。

[3D] 说明：
    三维位置由视觉测量直接写入 magnetic_model；
    三维速度由相邻有效测量的位移差和时间差计算。
"""

from core.runtime_context import RuntimeContext
from core.system_factory import (
    DEFAULT_RUNTIME_CFG,
    create_magnetic_model,
    create_runtime_context,
    merge_runtime_cfg,
)
from core.current_allocator import force_3d_to_current, zero_current
from core.current_solver import current_control1

__all__ = [
    "RuntimeContext",
    "DEFAULT_RUNTIME_CFG",
    "create_runtime_context",
    "create_magnetic_model",
    "merge_runtime_cfg",
    "force_3d_to_current",
    "current_control1",
    "zero_current",
]
