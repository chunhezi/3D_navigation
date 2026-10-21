"""
文件功能：
    运动模式管理器，负责模式切换与统一调用入口。

设计原则：
    1. 只负责模式切换与 step 转发，不包含具体控制算法；
    2. 保证同一时刻只有一个运动模式处于激活状态；
    3. main.py 通过本模块访问运动控制，不直接调用具体 motion 文件。

与其他模块关系：
    被 main.py 调用。
    当前注册 Stabilize3DMotionMode，保留注册表供后续扩展。
"""

from motion_modes.motion_stabilize_3d import Stabilize3DMotionMode


class ModeManager:
    """
    运动模式管理器。

    功能：
    1. 根据 UI 选择切换当前运动模式；
    2. 保证同一时刻只有一个模式处于激活状态；
    3. 对外提供统一 step() 接口；
    4. main.py 不再关心具体模式内部如何计算电流。

    该类负责：
        - 模式注册、切换、停止；
        - 将 MotionObservation 转发给当前激活模式。

    该类不负责：
        - 构造观测量（由 main.py 负责）；
        - 下发电流（由 main.py 读取 MotionCommand 后执行）；
        - 具体控制算法（由各 MotionMode 子类负责）。

    内部状态：
        current_mode: 当前激活的运动模式实例，未选择时为 None。
        mode_table: 模式名称到模式类的映射表。
    """

    def __init__(self, ctx):
        """
        初始化模式管理器。

        参数：
            ctx: RuntimeContext，全局运行上下文，将传递给各运动模式实例。

        是否会改变内部状态：
            是，初始化 mode_table，current_mode 置为 None。

        是否涉及硬件输出：
            否。
        """
        self.ctx = ctx

        # 当前激活的运动模式对象，切换前为 None
        self.current_mode = None

        # 模式名称到模式类的映射
        self.mode_table = {
            # [3D_PID] UI 名称由 main.py 映射为该内部模式键。
            "stabilize_3d": Stabilize3DMotionMode,
        }

    def switch_mode(self, mode_name: str):
        """
        切换运动模式。

        参数：
            mode_name: 注册表中的模式名称，当前为 stabilize_3d。

        执行流程：
        1. 检查模式名称是否合法；
        2. 停止旧模式内部状态；
        3. 创建新模式对象；
        4. 调用新模式 reset()。

        说明：
            日志保存不由 switch_mode() 负责；
            实验日志由 main.py 在 stop motion 时集中保存到实验会话目录。

        是否会改变内部状态：
            是，替换 current_mode 并触发 reset。

        是否涉及硬件输出：
            否（旧模式 stop() 不再写 PKL / 不下发电流）。
        """
        if mode_name not in self.mode_table:
            raise ValueError(f"Unknown motion mode: {mode_name}")

        # 如果当前已有模式，先停止旧模式内部状态（不保存默认路径 PKL）
        if self.current_mode is not None:
            self.current_mode.stop()

        # 创建新模式实例
        mode_cls = self.mode_table[mode_name]
        self.current_mode = mode_cls(self.ctx)

        # 重置新模式内部状态
        self.current_mode.reset()

    def step(self, obs):
        """
        执行当前模式的一帧控制。

        参数：
            obs: MotionObservation，由 main.py 构造的当前帧观测量。

        返回：
            MotionCommand，当前模式的控制输出。

        是否会改变内部状态：
            间接，当前模式 step() 会更新其内部状态。

        是否涉及硬件输出：
            间接，返回的 MotionCommand.current 由 main.py 下发。
        """
        if self.current_mode is None:
            raise RuntimeError("No motion mode selected. Please call switch_mode() first.")

        return self.current_mode.step(obs)

    def stop(self):
        """
        停止当前模式。

        通常在程序退出、实验结束或紧急停止时调用。

        是否会改变内部状态：
            是，调用当前模式 stop()，但不自动清空 current_mode 引用。

        是否涉及硬件输出：
            间接，当前模式 stop() 可能配合 main.py 输出零电流。
        """
        if self.current_mode is not None:
            self.current_mode.stop()

    def on_motion_start(self, **kwargs):
        """
        转发 motion 启动事件给当前模式。
        main.py 不需要知道当前模式具体类型。
        """
        if self.current_mode is not None and hasattr(self.current_mode, "on_motion_start"):
            return self.current_mode.on_motion_start(**kwargs)

    def on_motion_stop(self, **kwargs):
        """
        转发 motion 停止事件给当前模式。
        """
        if self.current_mode is not None and hasattr(self.current_mode, "on_motion_stop"):
            return self.current_mode.on_motion_stop(**kwargs)

    # [3D_PID] BEGIN: 向视觉线程转发当前模式的只读叠图状态。
    # VisionControlLoop 只能通过管理器读取目标/参考路径，因此不会 import
    # 具体控制模式，也不会误改 PID 积分或目标。dict() 只复制外层映射；
    # 3D 参考数组由模式设为只读并按内容变化整体替换，以支持静态层缓存。
    def get_overlay_state(self):
        """Return a defensive copy-friendly display snapshot from the mode."""
        if self.current_mode is None:
            return {}
        getter = getattr(self.current_mode, "get_overlay_state", None)
        if not callable(getter):
            return {}
        state = getter()
        return dict(state or {})
    # [3D_PID] END: 叠图状态转发。
