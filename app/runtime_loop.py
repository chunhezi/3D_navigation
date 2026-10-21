"""
文件功能：
    单帧运行循环：泵扰动调度、构造 MotionObservation、调用 ModeManager、下发硬件。

设计原则：
    1. 集中原 main.py detect() 中的控制步逻辑，不创建 UI，不读相机；
    2. 运动模式通过公共观察和 ModeManager 接口分派；
    3. 泵时序与硬件电流下发边界保持原逻辑。

与其他模块关系：
    被 main.py 在 detect() 循环中每帧调用 step()。
    依赖 core RuntimeContext、mode_manager、devices，不 import motion.py。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from motion_modes.base_motion import MotionObservation


@dataclass
class RuntimeLoopConfig:
    """
    运行循环配置（与 main.py 模块级/实例级常量对齐）。
    """

    # 控制周期（秒）
    dt: float = 0.03
    # ==================== 泵相关：相位与恢复时序开始 ====================
    # 功能：配置弱/强扰动周期，以及强转弱后的回流保持步数。
    n_weak: int = 60
    n_period: int = 100
    # 强→弱下降沿后的回流保持步数
    n_recover: int = 0
    # ==================== 泵相关：相位与恢复时序结束 ====================
    # 线圈电流限幅（A）
    current_limit: float = 8.0


@dataclass
class RuntimeStepResult:
    """
    单帧运行结果。

    供视觉循环更新控制步编号并报告性能。
    """

    # 本帧线圈电流命令（6 通道）
    current: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # 下一控制步索引（motion 未激活时与输入相同）
    i_motion: int = 0
    # 本帧是否执行了 motion 控制步
    motion_active: bool = False
    # ==================== 泵相关：本帧调度结果开始 ====================
    # 功能：向调用方保留泵相位和恢复控制状态；未运行时为 None。
    pump_flag: Optional[int] = None
    ctrl_flag: Optional[int] = None
    # ==================== 泵相关：本帧调度结果结束 ====================
    # 分段耗时（秒）
    control_time: float = 0.0
    drive_time: float = 0.0


class RuntimeLoop:
    """
    单帧运行循环。

    功能：
        1. 按 i_motion 计算泵扰动相位与 ctrl_flag；
        2. 边沿触发泵命令；
        3. 读取从端位置/速度，构造 MotionObservation；
        4. 调用 ModeManager.step；
        5. 下发线圈电流。

    不负责：
        UI 控件、相机读帧、机器人检测、overlay 绘制。
    """

    def __init__(
        self,
        ctx: Any,
        mode_manager: Any,
        devices: Optional[Any] = None,
        cfg: Optional[RuntimeLoopConfig] = None,
    ):
        """
        参数：
            ctx: RuntimeContext；
            mode_manager: ModeManager 实例；
            devices: DeviceManager，可选（为 None 时不发硬件）；
            cfg: 运行循环配置。
        """
        self.ctx = ctx
        self.mode_manager = mode_manager
        self.devices = devices
        self.cfg = cfg if cfg is not None else RuntimeLoopConfig()

        # ==================== 泵相关：内部调度状态开始 ====================
        # 功能：保留恢复计数、上帧相位和上次已下发命令，供边沿触发使用。
        self.recover_cnt: int = 0
        self.prev_pump_flag: int = 0
        self.prev_pump_cmd: Optional[int] = None
        # ==================== 泵相关：内部调度状态结束 ====================

    # ==================== 泵相关：复位、相位计算和命令下发开始 ====================
    # 功能：管理扰动周期及回流恢复，并仅在相位边沿向设备下发泵命令。
    def reset_pump_scheduler(self) -> None:
        """重置泵调度状态（可选，例如新实验开始时调用）。"""
        self.recover_cnt = 0
        self.prev_pump_flag = 0
        self.prev_pump_cmd = None

    def _compute_pump_and_ctrl_flags(self, i_motion: int) -> tuple[int, int]:
        """
        计算本步 pump_flag 与 ctrl_flag。

        逻辑迁移自 main.py detect()，保持不变。
        """
        cfg = self.cfg
        phase_idx = int(i_motion) % int(cfg.n_period)
        pump_flag = 0 if phase_idx < cfg.n_weak else 1

        if (self.prev_pump_flag == 1) and (pump_flag == 0):
            self.recover_cnt = int(cfg.n_recover)
        self.prev_pump_flag = pump_flag

        if self.recover_cnt > 0:
            ctrl_flag = 1
            self.recover_cnt -= 1
        else:
            ctrl_flag = pump_flag

        return pump_flag, ctrl_flag

    def _apply_pump_command(self, pump_flag: int) -> None:
        """泵命令边沿下发（与原 prev_pump_cmd 逻辑一致）。"""
        if (self.prev_pump_cmd is None) or (pump_flag != self.prev_pump_cmd):
            self.prev_pump_cmd = pump_flag
            if self.devices is not None:
                self.devices.set_pump_disturbance(pump_flag)
    # ==================== 泵相关：复位、相位计算和命令下发结束 ====================

    def build_motion_observation(
        self,
        *,
        slave_pos,
        slave_vel,
        pump_flag: int,
        ctrl_flag: int,
        frame,
        i_motion: int,
        control_mode: str,
    ) -> MotionObservation:
        """
        将实时信息打包为 MotionObservation（原 main.build_motion_observation 逻辑）。
        """
        # ==================== 泵相关：观测状态打包开始 ====================
        # 功能：将恢复、强扰动或弱扰动状态传入公共观察接口。
        if self.recover_cnt > 0:
            pump_state = "recover"
        elif int(pump_flag) == 1:
            pump_state = "strong"
        else:
            pump_state = "weak"
        # ==================== 泵相关：观测状态打包结束 ====================

        return MotionObservation(
            t=time.time(),
            dt=self.cfg.dt,
            slave_pos=np.asarray(slave_pos, dtype=float).reshape(-1)[:2],
            slave_vel=np.asarray(slave_vel, dtype=float).reshape(-1)[:2],
            pump_state=pump_state,
            frame=frame,
            extra={
                "control_mode": control_mode,
                "control_flag": ctrl_flag,
                "pump_flag": pump_flag,
                "i_motion": i_motion,
            },
        )

    def step(
        self,
        *,
        motion_active: bool,
        i_motion: int,
        frame: Any,
        control_mode: str,
        drive_enabled: bool = False,
    ) -> RuntimeStepResult:
        """
        执行一帧控制流程。

        参数：
            motion_active: 是否处于 motion 运行状态；
            i_motion: 当前控制步索引；
            frame: 原始 BGR 图像，传入 MotionObservation；
            control_mode: UI 控制模式字符串；
            drive_enabled: 是否允许下发线圈电流（对应原 port_state）。

        返回：
            RuntimeStepResult。
        """
        result = RuntimeStepResult(
            current=np.zeros(6, dtype=float),
            i_motion=int(i_motion),
            motion_active=bool(motion_active),
        )

        if not motion_active:
            return result

        # ==================== 泵相关：执行本帧扰动调度开始 ====================
        # 功能：保留相位与恢复状态结果，并按既有边沿逻辑下发泵命令。
        pump_flag, ctrl_flag = self._compute_pump_and_ctrl_flags(i_motion)
        result.pump_flag = pump_flag
        result.ctrl_flag = ctrl_flag
        self._apply_pump_command(pump_flag)
        # ==================== 泵相关：执行本帧扰动调度结束 ====================

        mm = self.ctx.magnetic_model
        # 保留公共观察接口的平面位置/速度；三维模式自行读取完整模型状态。
        position_3d = np.asarray(mm.return_position(), dtype=float).reshape(3,)
        try:
            velocity_3d = np.asarray(mm.return_vel(), dtype=float).reshape(3,)
        except Exception:
            velocity_3d = np.zeros(3, dtype=float)
        slave_pos = position_3d[:2]
        slave_vel = velocity_3d[:2]

        obs = self.build_motion_observation(
            slave_pos=slave_pos,
            slave_vel=slave_vel,
            pump_flag=pump_flag,
            ctrl_flag=ctrl_flag,
            frame=frame,
            i_motion=i_motion,
            control_mode=control_mode,
        )

        t_ctrl_start = time.perf_counter()

        # 这边进入mode_manager
        cmd = self.mode_manager.step(obs)
        t_ctrl_end = time.perf_counter()
        result.control_time = t_ctrl_end - t_ctrl_start

        current = np.asarray(cmd.current, dtype=float).reshape(-1)
        if current.size < 6:
            current = np.pad(current, (0, 6 - current.size))
        result.current = current[:6]

        # 三维模式通过 MotionCommand.log_info["error_xyz"] 显式给出误差；
        # 重复/无效 pair 给出三维零误差，防止日志误用上一帧控制误差。
        if hasattr(mm, "append_state_log"):
            err_for_log = None
            cmd_log_info = cmd.log_info or {}
            explicit_error_xyz = cmd_log_info.get("error_xyz")
            if explicit_error_xyz is not None:
                err_for_log = np.asarray(
                    explicit_error_xyz, dtype=float
                ).reshape(3,)
            mm.append_state_log(
                event_type="control",
                control_timestamp=time.perf_counter(),
                target_position=cmd.target_pos,
                error=err_for_log,
                force_cmd=cmd.force_cmd,
                coil_current=result.current,
                control_mode=control_mode,
            )

        t_drive_start = time.perf_counter()
        # [3D_PID] 安全边界保持在既有下发层：新模式同样经过逐路 +/-8 A 限幅，
        # 未在 PID 内重复增加输出限幅或改变串口访问边界。
        if drive_enabled and self.devices is not None:
            current_c = np.clip(result.current, -self.cfg.current_limit, self.cfg.current_limit)
            self.devices.set_currents(current_c)
        t_drive_end = time.perf_counter()
        result.drive_time = t_drive_end - t_drive_start

        result.i_motion = int(i_motion) + 1
        return result
