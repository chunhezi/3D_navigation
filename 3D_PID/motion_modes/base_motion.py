"""
文件功能：
    定义运动模式的统一输入输出接口与基类。

设计原则：
    1. 只定义数据结构（MotionObservation、MotionCommand）和抽象基类（BaseMotionMode）；
    2. 不包含任何具体控制算法或硬件操作；
    3. 作为当前三维模式和后续新增模式的公共契约。

与其他模块关系：
    被具体运动模式继承。
    被 mode_manager.py 和 main.py 引用。
    不依赖任何 motion 具体实现文件。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class MotionObservation:
    """
    单帧控制观测量。

    该类用于把 main.py 中采集到的所有实时信息统一打包，
    再传递给当前运动模式。

    注意：
    1. 不同模式可以只使用其中一部分字段；
    2. frame 字段仅用于需要图像辅助显示或调试的模式，不建议在控制器内部直接处理复杂图像。
    """

    # 当前系统时间，单位通常为秒
    t: float

    # 控制周期，单位为秒
    dt: float

    # 从端磁珠当前位置，二维坐标，例如 [x, y]
    slave_pos: np.ndarray

    # 从端磁珠速度，二维坐标，例如 [vx, vy]
    slave_vel: np.ndarray

    # ==================== 泵相关：公共观测状态开始 ====================
    # 功能：保留 weak、strong、recover 等泵状态，供后续模式读取。
    pump_state: Optional[str] = None
    # ==================== 泵相关：公共观测状态结束 ====================

    # 原始图像帧，通常不建议控制器直接依赖该字段
    frame: Optional[Any] = None

    # 其他临时信息，例如 ROI、检测置信度、当前实验编号等
    extra: Optional[Dict[str, Any]] = None


@dataclass
class MotionCommand:
    """
    单帧控制输出量。

    每个运动模式的 step() 都必须返回该对象。
    main.py 只读取该对象，不关心内部控制算法。
    """

    # 最终需要下发给线圈驱动的电流命令
    current: np.ndarray

    # 控制器输出的力向量或等效控制量，可用于记录和调试
    force_cmd: Optional[np.ndarray] = None

    # 当前模式生成的目标位置
    target_pos: Optional[np.ndarray] = None

    # 当前模式生成的目标速度
    target_vel: Optional[np.ndarray] = None

    # 用于 UI 或图像 overlay 的显示信息
    display_info: Optional[Dict[str, Any]] = None

    # 用于实验日志保存的信息
    log_info: Optional[Dict[str, Any]] = None


class BaseMotionMode:
    """
    所有运动模式的基类。

    具体模式继承该类并实现 step()，通过模式管理器统一调用。

    该类负责：
        - 保存 RuntimeContext 引用；
        - 定义 reset / step / stop 生命周期接口。

    该类不负责：
        - 具体控制算法；
        - 硬件读写；
        - 模式之间的切换逻辑。

    主要输入：
        MotionObservation（由 main.py 构造后传入 step）。

    主要输出：
        MotionCommand（由 step 返回，main.py 读取 current 下发）。
    """

    # 模式名称标识，子类覆盖为注册表中对应的名称。
    name = "base"

    def __init__(self, ctx):
        """
        参数：
            ctx: RuntimeContext，全局运行上下文，包含 cfg、模型、控制器、记录器等公共对象。

        是否会改变内部状态：
            是，保存 ctx 引用供子类使用。

        是否涉及硬件输出：
            否。
        """
        self.ctx = ctx

    def reset(self):
        """
        模式启动或切换进来时调用。

        用途：
        1. 清空上一轮实验状态；
        2. 重置滤波器；
        3. 重置目标点；
        4. 重置内部缓存变量。

        是否会改变内部状态：
            是，子类应重写以清空模式私有状态。

        是否涉及硬件输出：
            否。
        """
        pass

    def step(self, obs: MotionObservation) -> MotionCommand:
        """
        单帧控制入口。

        参数：
            obs: 当前帧观测量。

        返回：
            MotionCommand: 当前帧控制输出。

        注意：
            子类必须重写该函数。

        是否会改变内部状态：
            是，通常更新滤波器、目标缓存等。

        是否涉及硬件输出：
            间接，返回的 MotionCommand.current 将由 main.py 下发。
        """
        raise NotImplementedError("Each motion mode must implement step().")

    def stop(self):
        """
        模式退出时调用。

        用途：
        1. 保存日志；
        2. 清理临时状态；
        3. 必要时输出零电流；
        4. 停止该模式专用的辅助线程。

        是否会改变内部状态：
            是，清理模式私有资源。

        是否涉及硬件输出：
            间接，子类可在 stop 中请求零电流（由 main.py 执行）。
        """
        pass

    def on_motion_start(self, **kwargs):
        """
        motion 按钮从停止切换到启动时调用。

        该接口用于让具体 motion 模式执行自己的启动初始化逻辑。
        main.py 不需要判断当前模式类型。
        """
        pass

    def on_motion_stop(self, **kwargs):
        """
        motion 按钮从启动切换到停止时调用。

        默认不做任何事，具体模式可按需重写。
        """
        pass

    # [3D_PID] BEGIN: 模式到视觉层的只读显示接口。
    # 该接口加在基类而不是让视觉层直接识别 Stabilize3DMotionMode，目的是
    # 保持控制与绘图解耦。没有叠图数据的模式继承空字典行为；
    # 3D 模式覆盖后只发布 target_xyz/ref_path_3d，不发布或修改控制器状态。
    def get_overlay_state(self) -> Dict[str, Any]:
        """Return read-only display data for the vision thread.

        Modes without overlays expose an empty mapping.
        A mode may override this to publish a target or reference path without
        coupling the renderer to the controller implementation.
        """
        return {}
    # [3D_PID] END: 模式显示状态接口。
