"""
文件功能：
    定义全局运行上下文 RuntimeContext，集中保存运动模式共用的对象引用。

设计原则：
    1. 只保存对象引用，不写具体控制逻辑；
    2. 避免每个 motion 文件重复初始化模型、设备、日志器等；
    3. 不同模式可以只使用其中一部分字段。

与其他模块关系：
    由 main.py 在启动时构造并注入各公共对象。
    被 BaseMotionMode 及其子类通过 self.ctx 访问。
    不依赖任何具体模式文件。
"""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class RuntimeContext:
    """
    全局运行上下文。

    该对象用于保存运动模式可能用到的公共资源。
    这样可以避免每个 motion 文件重复初始化模型、设备、日志器等对象。

    注意：
    1. RuntimeContext 只保存对象引用，不写具体控制逻辑；
    2. 运动模式通过 self.ctx.xxx 访问公共对象；
    3. 不同模式可以只使用其中一部分对象；
    4. 不建议在 motion 文件中重新创建硬件设备对象；
    5. 磁珠位置和速度由 magnetic_model 根据三维视觉测量直接更新。

    该类负责：
        - 持有 cfg、模型、设备、记录器等共享引用。

    该类不负责：
        - 对象的创建与销毁（由 main.py 负责）；
        - 单帧控制计算。
    """

    # 全局参数配置，例如 dt、电流限幅、比例尺、控制参数等
    cfg: Any

    # 磁场/磁力/运动学模型，用于力到电流的映射及磁学计算
    magnetic_model: Any

    # 日志记录器，用于实验数据保存
    recorder: Optional[Any] = None

    # ==================== 泵相关：调度器预留开始 ====================
    # 功能：保留弱/强扰动调度器的共享引用，供现有泵时序扩展使用。
    scheduler: Optional[Any] = None
    # ==================== 泵相关：调度器预留结束 ====================

    # 其他设备管理器或临时对象，例如 Drive、泵和串口等
    devices: Optional[Any] = None

