"""[3D_PID] Minimal fixed-position 3D P/PI stabilization motion mode."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.controllers import pid_3d_controller as pid_cfg
from core.controllers.pid_3d_controller import PID3DController, PID3DResult
from core.current_allocator import force_3d_to_current
# [pid3D_debug] BEGIN: import centralized PID debug printer
from core.debug.pid_3d_debug import PID3DDebugPrinter
# [pid3D_debug] END: import centralized PID debug printer
from core.gravity_compensation import (
    get_gravity_compensation_force,
    gravity_parameter_snapshot,
)
# [REFERENCE 3D] BEGIN: 接入公共三维参考轨迹生成器
# 只导入接口：运动层不关心航点列表、130 计数、循环规则等细节。
from core.reference_generator_3d import (
    get_reference_3d,
    reference_parameter_snapshot_3d,
    reset_reference_3d,
)
# [REFERENCE 3D] END: 接入公共三维参考轨迹生成器
from motion_modes.base_motion import BaseMotionMode, MotionCommand, MotionObservation


# [3D_PID] BEGIN: 目标、起点保持和未来三维参考路径入口。
# Position unit: cm.  x=y=0 is RobotDetector.circle_position: the geometric
# centre of the top-view image after first-frame scene calibration.
TARGET_X_CM = 0.0
TARGET_Y_CM = 0.0
TARGET_Z_CM = 0.5

USE_START_POSITION_AS_TARGET = False

# Future trajectory interface.  When supplied it must be finite, shape (N, 3),
# and use centimetres.  V1 fixed-point control does not consume this path.
REF_PATH_3D = None
# [3D_PID] END: stabilize_3d user-facing mode parameters.


# [3D_PID] 严格限定参考路径为有限 (N,3) cm；空路径等价于定点镇定，
# 只供双平面叠图使用，V1 控制器不会沿路径自动改变目标。
def _validated_ref_path_3d(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"REF_PATH_3D must have shape (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("REF_PATH_3D contains a non-finite value")
    out = arr.copy()
    out.setflags(write=False)
    return out


# [3D_PID] BEGIN: 独立三维 P/PI 运动模式。
class Stabilize3DMotionMode(BaseMotionMode):
    """3D position feedback gated by a new, fully valid stereo pair."""

    name = "stabilize_3d"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.pid = PID3DController()
        # [pid3D_debug] BEGIN: create centralized debug helper
        self.pid_debug = PID3DDebugPrinter()
        # [pid3D_debug] END: create centralized debug helper
        self.target_xyz: Optional[np.ndarray] = None
        self.ref_path_3d = _validated_ref_path_3d(REF_PATH_3D)
        self.last_pair_id = None
        self._start_target_pending = False
        self._first_log_time_s: Optional[float] = None
        self._records: list[Dict[str, Any]] = []

    def _clear_shared_logs(self) -> None:
        """Clear only experiment logs written by this mode/shared runtime."""
        mm = self.ctx.magnetic_model
        for name in (
            "f_des_log",
            "current_log",
            "real_current_log",
            "input_log",
            "dead_input_log",
            "state_log",
        ):
            if hasattr(mm, name):
                setattr(mm, name, [])

    @staticmethod
    def _fixed_target() -> np.ndarray:
        return np.array([TARGET_X_CM, TARGET_Y_CM, TARGET_Z_CM], dtype=float)

    def _current_measurement_is_valid(self) -> bool:
        mm = self.ctx.magnetic_model
        mask = np.asarray(
            getattr(mm, "measurement_valid_mask", np.zeros(3, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        return mask.size == 3 and bool(np.all(mask))

    def reset(self) -> None:
        # [3D_PID] reset 时消费当前 pair，保证运动开始前的旧测量不能触发控制。
        self.pid.reset()
        # [pid3D_debug] BEGIN: reset print counter only
        self.pid_debug.reset()
        # [pid3D_debug] END: reset print counter only
        self._records.clear()
        self._clear_shared_logs()
        self._first_log_time_s = None
        mm = self.ctx.magnetic_model
        self.last_pair_id = getattr(mm, "measurement_pair_id", None)
        self.ref_path_3d = _validated_ref_path_3d(REF_PATH_3D)

        # ============================================================================
        # [REFERENCE 3D] BEGIN: 运动复位时同步复位三维参考轨迹
        # reset_reference_3d()：index=0，有效pair计数=0，从第一个航点重新开始。
        # get_reference_3d(advance=False)：只读取当前参考，不推进计数，
        # 避免 Start Motion 时就把计数从 0 偷偷变成 1。
        # 返回值：
        #   self.target_xyz  -> PID 当前目标点 (3,) cm
        #   self.ref_path_3d -> 整条航点路径 (N,3) cm，供上下视叠图
        # ============================================================================
        reset_reference_3d()
        (
            self.target_xyz,
            self.ref_path_3d,
            _reference_info,
        ) = get_reference_3d(
            advance=False
        )
        # ============================================================================
        # [REFERENCE 3D] END: 运动复位时同步复位三维参考轨迹
        # ============================================================================

        if USE_START_POSITION_AS_TARGET:
            if self._current_measurement_is_valid():
                self.target_xyz = np.asarray(
                    mm.return_position(), dtype=float
                ).reshape(3,).copy()
                self._start_target_pending = False
            else:
                self.target_xyz = None
                self._start_target_pending = True
        else:
            # 原先这里会用 self._fixed_target() 覆盖目标。
            # 现已改为上面 get_reference_3d() 提供目标；本分支只清 pending。
            self._start_target_pending = False

    def on_motion_start(self, **kwargs) -> None:
        # ModeManager creates and resets a fresh instance immediately before
        # this callback.  Do not reset again or consume an intervening pair.
        _ = kwargs

    def stop(self) -> None:
        return None

    def _zero_command(
        self,
        *,
        obs: MotionObservation,
        position_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        pair_id,
        valid_mask: np.ndarray,
        reason: str,
    ) -> MotionCommand:
        # [3D_PID] 重复/无效 pair 的共同安全出口：不调用 PID 和电流求解器，
        # 因此积分自然冻结，并返回严格零力、六路零电流，同时保留周期日志。
        target = None if self.target_xyz is None else self.target_xyz.copy()
        self._append_record(
            obs=obs,
            position_xyz=position_xyz,
            velocity_xyz=velocity_xyz,
            target_xyz=target,
            pid_result=None,
            gravity_force=np.zeros(3, dtype=float),
            force_xyz=np.zeros(3, dtype=float),
            current=np.zeros(6, dtype=float),
            pair_id=pair_id,
            valid_mask=valid_mask,
            valid_control=False,
            reason=reason,
        )
        return MotionCommand(
            current=np.zeros(6, dtype=float),
            force_cmd=np.zeros(3, dtype=float),
            target_pos=target,
            target_vel=np.zeros(3, dtype=float),
            display_info=self.get_overlay_state(),
            log_info={
                "mode": self.name,
                "error_xyz": np.zeros(3, dtype=float),
                "valid_control": False,
                "note": reason,
            },
        )

    def step(self, obs: MotionObservation) -> MotionCommand:
        mm = self.ctx.magnetic_model
        position_xyz = np.asarray(mm.return_position(), dtype=float).reshape(3,)
        velocity_xyz = np.asarray(mm.return_vel(), dtype=float).reshape(3,)
        pair_id = getattr(mm, "measurement_pair_id", None)
        valid_mask = np.asarray(
            getattr(mm, "measurement_valid_mask", np.zeros(3, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if valid_mask.size != 3:
            valid_mask = np.zeros(3, dtype=bool)

        # [3D_PID] 控制门：只允许“新的 pair 且三轴全部有效”进入 PID。
        new_pair = pair_id is not None and pair_id != self.last_pair_id
        if new_pair:
            # Consume invalid pairs too: the same invalid pair must never be
            # reconsidered on later high-rate control iterations.
            self.last_pair_id = pair_id
        valid_control = bool(new_pair and np.all(valid_mask))

        if not valid_control:
            reason = "stale_pair" if not new_pair else "invalid_measurement"
            return self._zero_command(
                obs=obs,
                position_xyz=position_xyz,
                velocity_xyz=velocity_xyz,
                pair_id=pair_id,
                valid_mask=valid_mask,
                reason=reason,
            )

        # [3D_PID] 可选起点保持：reset 时无有效三轴测量则等待首个新有效 pair；
        # 捕获前一直走零命令，捕获当帧误差为零但仍可应用重力前馈。
        if self._start_target_pending:
            self.target_xyz = position_xyz.copy()
            self._start_target_pending = False
        if self.target_xyz is None:
            return self._zero_command(
                obs=obs,
                position_xyz=position_xyz,
                velocity_xyz=velocity_xyz,
                pair_id=pair_id,
                valid_mask=valid_mask,
                reason="target_unavailable",
            )

        # ============================================================================
        # [REFERENCE 3D] BEGIN: 每个新的、有效的 3D measurement pair 推进一次参考
        #
        # 只有通过上面的 valid_control 门控后才会执行到这里，因此：
        #   1 个新有效三维测量对  ==  参考推进一次
        #
        # advance=True：由 reference_generator_3d 内部推进
        #   - step：pair 计数
        #   - helix：累计 obs.dt（时间参数化连续螺旋）
        # 运动层不写螺旋公式、不做 carrot / 到达判断。
        # ============================================================================
        (
            self.target_xyz,
            self.ref_path_3d,
            reference_info,
        ) = get_reference_3d(
            advance=True,
            dt_s=float(obs.dt),
        )
        _ = reference_info  # 仅调试/日志用，当前不参与控制
        # ============================================================================
        # [REFERENCE 3D] END: 每个新的、有效的 3D measurement pair 推进一次参考
        # ============================================================================

        # [3D_PID] 有效控制链：位置误差 -> P/PI 力 -> +z 重力前馈 -> 六路电流。
        pid_result = self.pid.step(
            position_xyz,
            self.target_xyz,
            float(obs.dt),
            allow_integral_update=True,
        )
        mm.set_error_3d(pid_result.error_xyz)

        gravity_force = get_gravity_compensation_force()
        force_xyz = pid_result.pid_force_xyz + gravity_force
        current = np.asarray(
            force_3d_to_current(
                force_xyz,
                mm,
                cfg=getattr(self.ctx, "cfg", None),
            ),
            dtype=float,
        ).reshape(-1)
        current_out = np.zeros(6, dtype=float)
        current_out[: min(6, current.size)] = current[:6]

        # [pid3D_debug] BEGIN: centralized PID debug output
        self.pid_debug.print_step(
            pair_id=pair_id,
            dt_s=float(obs.dt),
            position_xyz=position_xyz,
            target_xyz=self.target_xyz,
            pid_result=pid_result,
            gravity_force_xyz=gravity_force,
            final_force_xyz=force_xyz,
            current_a=current_out,
        )
        # [pid3D_debug] END: centralized PID debug output

        if hasattr(mm, "f_des_log"):
            mm.f_des_log.append(force_xyz.copy())
        if hasattr(mm, "current_log"):
            mm.current_log.append(current_out.copy())

        self._append_record(
            obs=obs,
            position_xyz=position_xyz,
            velocity_xyz=velocity_xyz,
            target_xyz=self.target_xyz,
            pid_result=pid_result,
            gravity_force=gravity_force,
            force_xyz=force_xyz,
            current=current_out,
            pair_id=pair_id,
            valid_mask=valid_mask,
            valid_control=True,
            reason="valid",
        )

        return MotionCommand(
            current=current_out,
            force_cmd=force_xyz.copy(),
            target_pos=self.target_xyz.copy(),
            target_vel=np.zeros(3, dtype=float),
            display_info=self.get_overlay_state(),
            log_info={
                "mode": self.name,
                "error_xyz": pid_result.error_xyz.copy(),
                "valid_control": True,
                "measurement_pair_id": pair_id,
            },
        )

    def _append_record(
        self,
        *,
        obs: MotionObservation,
        position_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        target_xyz: Optional[np.ndarray],
        pid_result: Optional[PID3DResult],
        gravity_force: np.ndarray,
        force_xyz: np.ndarray,
        current: np.ndarray,
        pair_id,
        valid_mask: np.ndarray,
        valid_control: bool,
        reason: str,
    ) -> None:
        # [3D_PID] 私有日志覆盖每个控制周期，包括 stale/invalid 周期；这与
        # magnetic_model 的共享 state_log 并存，便于独立复盘 pair 门控与积分。
        if self._first_log_time_s is None:
            self._first_log_time_s = float(obs.t)
        elapsed = float(obs.t) - self._first_log_time_s

        if pid_result is None:
            error = np.zeros(3, dtype=float)
            p_force = np.zeros(3, dtype=float)
            i_force = np.zeros(3, dtype=float)
            pid_force = np.zeros(3, dtype=float)
            integral = self.pid.integral.copy()
        else:
            error = pid_result.error_xyz.copy()
            p_force = pid_result.p_force_xyz.copy()
            i_force = pid_result.i_force_xyz.copy()
            pid_force = pid_result.pid_force_xyz.copy()
            integral = pid_result.integral_xyz.copy()

        self._records.append(
            {
                "time_s": elapsed,
                "position_cm": position_xyz.copy(),
                "velocity_cm_s": velocity_xyz.copy(),
                "target_position_cm": (
                    np.full(3, np.nan, dtype=float)
                    if target_xyz is None
                    else np.asarray(target_xyz, dtype=float).reshape(3,).copy()
                ),
                "error_cm": error,
                "p_force_n": p_force,
                "integral_state_cm_s": integral,
                "i_force_n": i_force,
                "pid_force_n": pid_force,
                "gravity_compensation_n": float(np.asarray(gravity_force)[2]),
                "force_cmd_n": np.asarray(force_xyz, dtype=float).reshape(3,).copy(),
                "current_a": np.asarray(current, dtype=float).reshape(6,).copy(),
                "measurement_pair_id": pair_id,
                "measurement_valid_mask": valid_mask.copy(),
                "integral_enabled": bool(pid_cfg.ENABLE_INTEGRAL),
                "valid_control": bool(valid_control),
                "reason": str(reason),
            }
        )

    def get_overlay_state(self) -> Dict[str, Any]:
        # [3D_PID] 仅发布绘图所需的防御性目标副本和只读路径引用。
        return {
            "target_xyz": (
                None if self.target_xyz is None else self.target_xyz.copy()
            ),
            "ref_path_3d": (
                None if self.ref_path_3d is None else self.ref_path_3d
            ),
        }

    def save_motion_log(self, pkl_path: str = "stabilize_3d_data.pkl"):
        # [3D_PID] 按列保存逐周期数据，并同时保存 grouped/top-level 参数快照，
        # 兼容简单历史分析脚本，也能精确追溯当次实验的控制配置。
        def series(key: str, *, dtype=float):
            return np.asarray([record[key] for record in self._records], dtype=dtype)

        parameters = pid_cfg.controller_parameter_snapshot()
        # [3D_PID] 记录公式的所有输入、中间量和结果，而不是只保存最终常量；
        # 之后可明确追溯磁珠/流体参数和经验系数分别取了什么值。
        parameters.update(gravity_parameter_snapshot())
        # [REFERENCE 3D] BEGIN: 保存当次三维参考轨迹参数快照
        parameters.update(reference_parameter_snapshot_3d())
        # [REFERENCE 3D] END: 保存当次三维参考轨迹参数快照
        parameters.update(
            {
                "TARGET_X_CM": float(TARGET_X_CM),
                "TARGET_Y_CM": float(TARGET_Y_CM),
                "TARGET_Z_CM": float(TARGET_Z_CM),
                "USE_START_POSITION_AS_TARGET": bool(USE_START_POSITION_AS_TARGET),
            }
        )
        # Import here to keep the control module independent from vision startup.
        from vision.side_detector import SIDE_Y_SIGN

        parameters["SIDE_Y_SIGN"] = int(SIDE_Y_SIGN)

        data: Dict[str, Any] = {
            "time_s": series("time_s"),
            "position_cm": series("position_cm"),
            "velocity_cm_s": series("velocity_cm_s"),
            "target_position_cm": series("target_position_cm"),
            "error_cm": series("error_cm"),
            "p_force_n": series("p_force_n"),
            "integral_state_cm_s": series("integral_state_cm_s"),
            "i_force_n": series("i_force_n"),
            "pid_force_n": series("pid_force_n"),
            "gravity_compensation_n": series("gravity_compensation_n"),
            "force_cmd_n": series("force_cmd_n"),
            "current_a": series("current_a"),
            "measurement_pair_id": np.asarray(
                [r["measurement_pair_id"] for r in self._records], dtype=object
            ),
            "measurement_valid_mask": series(
                "measurement_valid_mask", dtype=bool
            ),
            "integral_enabled": series("integral_enabled", dtype=bool),
            "valid_control": series("valid_control", dtype=bool),
            "reason": np.asarray([r["reason"] for r in self._records], dtype=object),
            "parameters": parameters,
            "ref_path_3d_cm": (
                None if self.ref_path_3d is None else self.ref_path_3d.copy()
            ),
            "stabilize_3d_version": "STABILIZE_3D_V1",
        }
        # Also expose the tuning keys at the PKL top level for simple legacy
        # analysis scripts, while retaining the grouped snapshot above.
        data.update(parameters)

        output_path = Path(pkl_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(data, file_obj)
        return str(output_path.resolve())
# [3D_PID] END: 独立三维 P/PI 运动模式。
