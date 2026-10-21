"""
文件功能：
    相机-检测-overlay-控制单帧循环（视觉线程主入口）。

设计原则：
    从 main.py 的 detect() 拆出视觉与控制步进逻辑，
    使 main.py 只负责 Qt 显示与线程调度，不直接处理检测与 overlay 细节。
    该类不依赖 PyQt 控件，通过 on_frame_ready 回调与 main.py 解耦。

模块职责分区（阅读代码时先看标签）：
    【上视】       仅涉及 OpenCVCamera、RobotDetector、OverlayRenderer、top_* frame_bundle
    【侧视】       仅涉及 HikSideCamera、SideDetector、side_* frame_bundle、pair_id/z 显示
    【双视·共用】  同一 while 循环取帧、合并 frame_bundle、DualViewCalibrator 15 对标定
    【状态更新】   top xy + side z → 三维直接测量状态与差分速度
    【控制链】     RuntimeLoop.step()；读取测量更新后的三维状态

[3D] 双视状态更新（重要）：
    侧视 z 与上视 xy 在同一 pair 中直接写入 magnetic_model.position；
    速度由相邻有效测量的位移差 / 各轴测量时间差计算；
    RobotDetector 只输出原始 xy，不再直接写模型；
    无效 z 仅在日志中记为 -10，position / return_matrix 绝不使用 -10。

单帧数据流（第 k 次循环）：
    ┌─【上视】OpenCVCamera.read() → top_timestamp
    ├─【侧视】HikSideCamera.grab_gray() → side_frame（禁止复用上一轮缓存）
    ├─【上视】_process_vision_frame() → top_main + 四个上视调试窗
    ├─【侧视】_build_side_frame_bundle() → side_main + 四个侧视调试窗
    ├─【双视·共用】_update_dual_calibration() → 15 对静止标定（不接控制）
    ├─【状态更新】_update_3d_measured_state_from_pair() → 直接测量写回模型
    ├─【双视·共用】frame_bundle.update() → on_frame_ready() → main.py Qt 显示
    └─【控制链】runtime_loop.step(frame=cv_img) → 线圈/模式

软件配对规则（无硬件触发）：
    1. 同一循环内：先 read 上视，再 grab_gray 侧视；
    2. 仅当本轮 side_frame 成功时：pair_id += 1，得到 new_pair_id；
       无侧视帧则 new_pair_id=None，不更新测量状态、不收集标定；
    3. 同一 new_pair_id 最多触发一次 _update_3d_measured_state_from_pair（一次 measurement）；
    4. pair_dt_ms = |side_timestamp - top_timestamp| * 1000；
    5. 侧视 grab 失败时显示 NO SIDE FRAME，不用上一帧顶替；
    6. 侧视相机离线时显示 SIDE CAMERA OFFLINE，上视照常运行。

配置读取优先级：
    runtime_ctx.cfg（main.py 中的 dict）> VisionControlLoopConfig 字段默认值。
    侧视检测参数详见 vision/side_detector.py 顶部「参数调试速查」。

frame_bundle key 一览：
    【上视】 top_main, top_binary, top_eroded_roi, top_candidates, top_filter_result
    【侧视】 side_main, side_binary, side_eroded_roi, side_candidates, side_filtered

与其他模块关系：
    被 main.py detect() 在线程中调用 run()。
    依赖 devices.hik_side_camera、vision.side_detector、vision.robot_detector。
    不修改 app/runtime_loop.py 内部逻辑；不 import motion_modes 控制器实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from devices.hik_side_camera import HikSideCamera
from vision.camera_source import CameraConfig, OpenCVCamera
from vision.dual_view_calibrator import DualViewCalibrationConfig, DualViewCalibrator
# [3D_PID] BEGIN: 双视 y 标定方向参数与侧视 yz 叠图接入。
# 视觉循环只使用公开坐标/绘图接口，不 import 三维 PID 控制器。
from vision.side_detector import SIDE_Y_SIGN, SideDetector
from vision.side_overlay_renderer import SideOverlayRenderer
# [3D_PID] END: 双视 y/侧视叠图依赖。

@dataclass
class VisionControlLoopConfig:
    """
    视觉循环配置。

    字段按【上视】【侧视】【双视·共用】分组；运行时优先被 runtime_ctx.cfg 覆盖。
    """

    # ----- 【上视】OpenCV 彩色相机采集参数 -----
    camera_index: int = 0
    fps: float = 20.0
    width: int = 1920
    height: int = 1080
    flip_horizontal: bool = True
    video_output_path: str = "output.avi"

    # ----- 【双视·共用】UI 占位图尺寸（侧视离线/缺帧时用）-----
    display_width: int = 640
    display_height: int = 360

    # ----- 【侧视】海康 Mono8 相机与检测参数 -----
    side_camera_enabled: bool = True
    side_camera_index: int = 0
    side_camera_timeout_ms: int = 50
    side_roi_half_height: int = 100
    side_roi_half_width: int = 350
    side_roi_center_offset_x: float = 0.0
    side_roi_center_offset_y: float = 0.0
    side_threshold: int = 80
    side_area_min: int = 50
    side_area_max: int = 150 # 这边可以把max给打一下，磁珠前后的移动导致的放大
    # 超过该值时在 side_main 叠加 PAIR DELAY WARNING（仍允许显示）
    side_pair_max_dt_ms: float = 80.0
    side_mvs_runtime_dir: Optional[str] = None

    # ----- 【双视·共用】15 对静止标定参数（磁珠放底面，不接控制）-----
    bead_diameter_mm: float = 1.0
    dual_calibration_required_pairs: int = 15
    dual_calibration_minimum_inliers: int = 10
    dual_calibration_relative_diameter_tolerance: float = 0.25
    dual_calibration_pair_max_dt_ms: float = 80.0

    top_calibration_diameter_min_px: float = 6.0
    top_calibration_diameter_max_px: float = 120.0
    top_calibration_circularity_min: float = 0.50

    side_calibration_diameter_min_px: float = 4.0
    side_calibration_diameter_max_px: float = 100.0
    side_calibration_circularity_min: float = 0.30

    top_calibration_center_spread_max_px: float = 3.0
    side_calibration_center_spread_max_px: float = 2.0


class VisionControlLoop:
    """
    相机-检测-overlay-控制单帧循环。

    main.py 通过 on_frame_ready 回调接收 frame_bundle（dict），
    在主线程分发到上视/侧视共 10 个 QLabel。

    成员分区：
        【上视】 robot_detector, overlay_renderer, pt1/pt2, circle_position/circle_state
        【侧视】 side_camera, side_detector, pair_id, latest_side_result, latest_top_result, _side_camera_online
        【双视·标定】 dual_calibrator, dual_calibration_result
        【状态更新】 _update_3d_measured_state_from_pair（top xy + side z → 直接测量/差分速度）
        【控制链】 runtime_loop（每帧 step；状态已由视觉测量写回模型）
    """

    def __init__(
        self,
        *,
        runtime_ctx: Any,
        robot_detector: Any,
        overlay_renderer: Any,
        runtime_loop: Any,
        cfg: Optional[VisionControlLoopConfig] = None,
        is_camera_active: Callable[[], bool],
        is_motion_active: Callable[[], bool],
        is_video_enabled: Callable[[], bool],
        is_drive_enabled: Callable[[], bool],
        get_i_motion: Callable[[], int],
        set_i_motion: Callable[[int], None],
        get_control_mode: Callable[[], str],
        on_frame_ready: Callable[[np.ndarray], None],
        on_scene_initialized: Optional[Callable[[], None]] = None,
        get_experiment_session_info: Optional[Callable[[], Dict[str, str]]] = None,
        on_video_recording_started: Optional[Callable[[str], None]] = None,
        on_video_recording_failed: Optional[Callable[[str], None]] = None,
        on_video_recording_stopped: Optional[Callable[[str, dict], None]] = None,
        is_drive_close_requested: Optional[Callable[[], bool]] = None,
        on_drive_close_finished: Optional[Callable[[bool, str], None]] = None,
    ):
        # ----- 【双视·共用】运行时上下文与回调 -----
        self.runtime_ctx = runtime_ctx
        self.cfg = cfg if cfg is not None else VisionControlLoopConfig()
        self.on_frame_ready = on_frame_ready
        self.on_scene_initialized = on_scene_initialized
        self.get_experiment_session_info = get_experiment_session_info or (lambda: {})
        self.on_video_recording_started = on_video_recording_started
        self.on_video_recording_failed = on_video_recording_failed
        self.on_video_recording_stopped = on_video_recording_stopped
        self.is_drive_close_requested = is_drive_close_requested or (lambda: False)
        self.on_drive_close_finished = on_drive_close_finished
        self._drive_close_already_handled = False

        # ----- 【上视】检测与 overlay 渲染器（由 main.py 注入）-----
        self.robot_detector = robot_detector
        self.overlay_renderer = overlay_renderer

        # ----- 【控制链】单帧运动/电流步进（不读侧视）-----
        self.runtime_loop = runtime_loop

        # ----- 【双视·共用】由 main.py 注入的状态查询回调 -----
        self.is_camera_active = is_camera_active
        self.is_motion_active = is_motion_active
        self.is_video_enabled = is_video_enabled
        self.is_drive_enabled = is_drive_enabled
        self.get_i_motion = get_i_motion
        self.set_i_motion = set_i_motion
        self.get_control_mode = get_control_mode

        # ----- 【上视】跨帧场景标定与 ROI 缓存 -----
        self.pt1 = None
        self.pt2 = None
        self.circle_position = None
        self.circle_state = False

        # ----- 【侧视】双相机配对与最新识别结果（仅供视觉调试）-----
        self.pair_id = 0
        self.latest_side_result = None
        self.latest_top_result = None
        self._side_camera_online = False
        self.side_camera: Optional[HikSideCamera] = None
        self.side_detector: Optional[SideDetector] = None
        # [3D_PID] 侧视 yz 叠图器与上视 XY 投影各自管理缓存。
        # 它仅在 stabilize_3d_mode 且 y 标定完成时工作。
        self.side_overlay_renderer = SideOverlayRenderer()

        if self._get_loop_cfg("side_camera_enabled", True):
            self.side_camera = HikSideCamera(
                index=int(self._get_loop_cfg("side_camera_index", 0)),
                runtime_dir=self._get_loop_cfg("side_mvs_runtime_dir", None),
            )
            self.side_detector = SideDetector(
                roi_half_height=int(self._get_loop_cfg("side_roi_half_height", 103)),
                roi_half_width=int(self._get_loop_cfg("side_roi_half_width", 350)),
                roi_center_offset_x=float(
                    self._get_loop_cfg("side_roi_center_offset_x", 0.0)
                ),
                roi_center_offset_y=float(
                    self._get_loop_cfg("side_roi_center_offset_y", 0.0)
                ),
                threshold=int(self._get_loop_cfg("side_threshold", 80)),
                area_min=int(self._get_loop_cfg("side_area_min", 10)),
                area_max=int(self._get_loop_cfg("side_area_max", 1000)),
                # ============== 本次视觉修改【侧视接入】：公用参数传入开始 ==============
                # 作用：复用上视已读取的公用配置，防止两路腐蚀核或次数不一致。
                kernel_size=self.robot_detector.cfg.erode_kernel_size,
                erode_iterations=self.robot_detector.cfg.erode_iterations,
                # ============== 本次视觉修改【侧视接入】：公用参数传入结束 ==============
            )

        self.dual_calibrator = DualViewCalibrator(
            DualViewCalibrationConfig(
                bead_diameter_mm=float(self._get_loop_cfg("bead_diameter_mm", 1.0)),
                required_pair_count=int(
                    self._get_loop_cfg("dual_calibration_required_pairs", 15)
                ),
                minimum_inlier_count=int(
                    self._get_loop_cfg("dual_calibration_minimum_inliers", 10)
                ),
                relative_diameter_tolerance=float(
                    self._get_loop_cfg("dual_calibration_relative_diameter_tolerance", 0.25)
                ),
                pair_max_dt_ms=float(
                    self._get_loop_cfg("dual_calibration_pair_max_dt_ms", 80.0)
                ),
                top_diameter_min_px=float(
                    self._get_loop_cfg("top_calibration_diameter_min_px", 6.0)
                ),
                top_diameter_max_px=float(
                    self._get_loop_cfg("top_calibration_diameter_max_px", 120.0)
                ),
                top_circularity_min=float(
                    self._get_loop_cfg("top_calibration_circularity_min", 0.50)
                ),
                side_diameter_min_px=float(
                    self._get_loop_cfg("side_calibration_diameter_min_px", 4.0)
                ),
                side_diameter_max_px=float(
                    self._get_loop_cfg("side_calibration_diameter_max_px", 100.0)
                ),
                side_circularity_min=float(
                    self._get_loop_cfg("side_calibration_circularity_min", 0.30)
                ),
                top_center_spread_max_px=float(
                    self._get_loop_cfg("top_calibration_center_spread_max_px", 3.0)
                ),
                side_center_spread_max_px=float(
                    self._get_loop_cfg("side_calibration_center_spread_max_px", 2.0)
                ),
            )
        )
        self.dual_calibration_result = None
        self._last_calibration_error: Optional[str] = None

    def _get_loop_cfg(self, key: str, default: Any) -> Any:
        """
        【双视·共用】读取循环配置。

        优先 runtime_ctx.cfg（main.py），其次 VisionControlLoopConfig 默认值。
        上视/侧视参数均可通过同一 dict 覆盖，例如 side_threshold、camera_index。
        """
        runtime_cfg = getattr(self.runtime_ctx, "cfg", None)
        if isinstance(runtime_cfg, dict) and key in runtime_cfg:
            return runtime_cfg[key]
        return getattr(self.cfg, key, default)

    def _placeholder_size(self) -> Tuple[int, int]:
        """【侧视】离线/缺帧占位图尺寸，与 main.py 显示格 16:9 对齐。"""
        return (
            int(self._get_loop_cfg("display_width", 640)),
            int(self._get_loop_cfg("display_height", 360)),
        )

    def _make_side_placeholder(self, text: str) -> Dict[str, np.ndarray]:
        """
        【侧视】生成五窗统一的占位图（相机离线或本轮无侧视帧）。

        返回 dict 仅含 side_* key；由 run() 合并进完整 frame_bundle。
        常见 text：
            "SIDE CAMERA OFFLINE" — 打开失败或设备不可用
            "NO SIDE FRAME"        — 本轮 grab_gray 超时/失败
        """
        width, height = self._placeholder_size()
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            image,
            text,
            (24, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return {
            "side_main": image,
            "side_binary": image.copy(),
            "side_eroded_roi": image.copy(),
            "side_candidates": image.copy(),
            "side_filtered": image.copy(),
        }

    def _draw_side_pair_overlay(self, main_bgr: np.ndarray, pair_dt_ms: float) -> None:
        """
        【侧视】在 side_main 底部叠加软件配对时差信息。

        pair_dt_ms 由本循环内 top_timestamp 与 side_timestamp 计算；
        超过 side_pair_max_dt_ms 时追加 PAIR DELAY WARNING（仍显示本帧结果）。
        """
        max_dt = float(self._get_loop_cfg("side_pair_max_dt_ms", 80.0))
        lines = [f"Pair dt: {pair_dt_ms:.1f} ms"]
        if pair_dt_ms > max_dt:
            lines.append("PAIR DELAY WARNING")
        y0 = main_bgr.shape[0] - 16 - 28 * (len(lines) - 1)
        for idx, line in enumerate(lines):
            color = (0, 0, 255) if "WARNING" in line else (0, 255, 255)
            cv2.putText(
                main_bgr,
                line,
                (12, y0 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    def _build_side_frame_bundle(
        self,
        *,
        side_frame: Optional[Any],
        top_timestamp: float,
        pair_id: int,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Any], Optional[float]]:
        """
        【侧视】构建本帧 side_* 图像字典（不处理上视）。

        参数：
            side_frame: 本轮 grab_gray 结果；None 表示本轮未取到侧视帧；
            top_timestamp: 本轮上视 read 成功后的时间戳，用于 pair_dt_ms；
            pair_id: 显示用配对序号（display_pair_id）；
                有侧视帧时为本轮新 pair，无侧视帧时为最近一次已生成的 pair。
                不得用该值判断是否应触发状态更新（以 run() 的 new_pair_id 为准）。

        返回：
            (side_bundle, side_result, pair_dt_ms)
            - 相机禁用：({}, None, None)
            - 相机离线 / 无侧视帧：(placeholder, None, None)
            - 有侧视帧：(side_bundle, SideDetectionResult, pair_dt_ms)

        注意：
            - 禁止用上一轮 side_frame 调用本函数；
            - 不在此修改 self.pair_id；
            - 不写入 magnetic_model / RuntimeLoop。
        """
        if not self._get_loop_cfg("side_camera_enabled", True):
            return {}, None, None

        if not self._side_camera_online or self.side_detector is None:
            return self._make_side_placeholder("SIDE CAMERA OFFLINE"), None, None

        if side_frame is None:
            return self._make_side_placeholder("NO SIDE FRAME"), None, None

        side_timestamp = float(side_frame.timestamp)
        pair_dt_ms = abs(side_timestamp - top_timestamp) * 1000.0

        side_result = self.side_detector.process(
            side_frame.gray,
            pair_id=pair_id,
            timestamp=side_timestamp,
        )

        main_bgr = side_result.main_bgr.copy()
        self._draw_side_pair_overlay(main_bgr, pair_dt_ms)

        side_bundle = {
            "side_main": main_bgr,
            "side_binary": side_result.binary_bgr,
            "side_eroded_roi": side_result.eroded_roi_bgr,
            "side_candidates": side_result.candidates_bgr,
            "side_filtered": side_result.filtered_bgr,
        }
        return side_bundle, side_result, pair_dt_ms

    def reset_dual_calibration(self) -> None:
        """重置双视标定样本与检测器尺度/零位（不重置上视场景 ROI）。"""
        self.dual_calibrator.reset()
        self.dual_calibration_result = None
        self._last_calibration_error = None

        self.robot_detector.reset_scale_calibration()

        if self.side_detector is not None:
            self.side_detector.reset_calibration()
        # [3D_PID] y/z 标定变化会改变像素投影，必须同时清除侧视静态层和轨迹层。
        self.side_overlay_renderer.reset()

        # 尺度/零位变化后重建差分基准，避免旧尺度坐标与新尺度坐标混差分
        mm = getattr(self.runtime_ctx, "magnetic_model", None)
        if mm is not None and hasattr(mm, "reset_measurement_history"):
            mm.reset_measurement_history()

        print("[DUAL CAL] reset; keep the bead still on the bottom")

    def _update_3d_measured_state_from_pair(
        self,
        *,
        pair_id: int,
        top_timestamp: float,
        pair_dt_ms,
        top_result,
        side_result,
    ) -> bool:
        """
        [3D] 在 top/side 汇合后，按 pair 直接更新三维测量状态与差分速度。

        作用：
            1. 上视有效 xy 直接写入模型 x、y；
            2. 侧视有效 z 直接写入模型 z；
            3. 三轴速度由各自相邻有效测量的位移差/时间差计算；
            4. 无效轴位置保持最后有效值，速度设为 0；
            5. 每个新 pair 最多更新一次。
        入口约束（由 run() 保证，此处再校验）：
            1. pair_id 必须是正整数（本轮 new_pair_id）；
            2. side_result 不得为 None（必须有本轮侧视检测结果对象）；
            3. 不满足则直接返回 False，不写测量；
            4. 本函数不自行生成或递增 pair_id。
        返回：
            True 表示本 pair 至少有一个有效测量轴；
            False 表示入口拒绝或模型不可用。
        安全边界：
            无效 z 不把 -10 写入 position；日志占位由 magnetic_model 内部处理。
        """
        try:
            pair_id_int = int(pair_id)
        except (TypeError, ValueError):
            return False
        if pair_id_int <= 0:
            return False
        if side_result is None:
            return False
        pair_id = pair_id_int

        magnetic_model = getattr(self.runtime_ctx, "magnetic_model", None)
        if magnetic_model is None:
            return False
        if not hasattr(magnetic_model, "update_state_from_measurement"):
            return False

        # ----- xy 有效条件（上视唯一候选 + 非刚标定首帧）-----
        xy_valid = False
        xy_cm = None
        if top_result is not None:
            model_xy = getattr(top_result, "model_position_xy_cm", None)
            xy_valid = (
                (not bool(getattr(top_result, "scene_just_initialized", False)))
                and int(getattr(top_result, "filtered_contour_count", 0)) == 1
                and model_xy is not None
            )
            if xy_valid:
                xy_arr = np.asarray(model_xy, dtype=float).reshape(-1)
                if xy_arr.size >= 2 and np.all(np.isfinite(xy_arr[:2])):
                    xy_cm = xy_arr[:2].copy()
                else:
                    xy_valid = False
                    xy_cm = None

        # ----- z 有效条件（侧视标定完成 + 配对时差可接受）-----
        z_valid = False
        z_cm = None
        max_dt = float(self._get_loop_cfg("side_pair_max_dt_ms", 80.0))
        z_mm = getattr(side_result, "z_mm", None)
        z_valid = (
            bool(getattr(side_result, "valid", False))
            and z_mm is not None
            and np.isfinite(float(z_mm))
            and bool(getattr(side_result, "scale_calibrated", False))
            and bool(getattr(side_result, "zero_calibrated", False))
            and int(getattr(side_result, "pair_id", -1)) == int(pair_id)
            and pair_dt_ms is not None
            and float(pair_dt_ms) <= max_dt
        )
        if z_valid:
            z_cm = float(z_mm) / 10.0
            if (not np.isfinite(z_cm)) or abs(z_cm - (-10.0)) < 1e-12:
                z_valid = False
                z_cm = None

        # x/y 用上视时间；z 用侧视时间；pair 时间仅作元数据
        xy_timestamp_s = float(top_timestamp)
        side_ts = getattr(side_result, "timestamp", None)
        if side_ts is not None and np.isfinite(float(side_ts)):
            z_timestamp_s = float(side_ts)
        else:
            z_timestamp_s = float(top_timestamp)
        pair_timestamp_s = max(xy_timestamp_s, z_timestamp_s)

        position_3d, velocity_3d = magnetic_model.update_state_from_measurement(
            xy_cm=xy_cm if xy_valid else None,
            z_cm=z_cm if z_valid else None,
            xy_valid=xy_valid,
            z_valid=z_valid,
            pair_id=pair_id,
            pair_timestamp_s=pair_timestamp_s,
            xy_timestamp_s=xy_timestamp_s,
            z_timestamp_s=z_timestamp_s,
        )

        ### 调整：两视图 Y 零点各自独立，取消跨视图 Y 差值诊断；控制仍使用上视 XY、侧视 Z。

        # 限频显示：每 30 个 pair 最多输出一次
        if int(pair_id) > 0 and (int(pair_id) % 30) == 0:
            meas = np.asarray(
                magnetic_model.measurement_position_cm, dtype=float
            ).reshape(3)
            mask = np.asarray(
                magnetic_model.measurement_valid_mask, dtype=bool
            ).reshape(3)
            dt_s = np.asarray(magnetic_model.measurement_dt_s, dtype=float).reshape(3)
            delta = np.asarray(
                magnetic_model.measurement_delta_cm, dtype=float
            ).reshape(3)
            print(
                "[3D MEASURED] "
                f"pair={pair_id} "
                f"timestamp={pair_timestamp_s:.6f} "
                f"measurement_valid_mask=("
                f"{bool(mask[0])},{bool(mask[1])},{bool(mask[2])}) "
                f"measurement_xyz_cm=({meas[0]:.3f},{meas[1]:.3f},{meas[2]:.3f}) "
                f"state_xyz_cm=("
                f"{float(position_3d[0]):.3f},"
                f"{float(position_3d[1]):.3f},"
                f"{float(position_3d[2]):.3f}) "
                f"velocity_xyz_cm_s=("
                f"{float(velocity_3d[0]):.3f},"
                f"{float(velocity_3d[1]):.3f},"
                f"{float(velocity_3d[2]):.3f}) "
                f"measurement_dt_s=("
                f"{dt_s[0]:.4f},{dt_s[1]:.4f},{dt_s[2]:.4f}) "
                f"measurement_delta_cm=("
                f"{delta[0]:.4f},{delta[1]:.4f},{delta[2]:.4f}) "
                f"xy_valid={bool(xy_valid)} "
                f"z_valid={bool(z_valid)}"
            )

        return bool(xy_valid or z_valid)

    def _update_dual_calibration(
        self,
        *,
        pair_id: int,
        pair_dt_ms: Optional[float],
        top_result,
        side_result,
    ) -> None:
        """
        【双视·标定】静止且双路唯一候选时收集配对观测，满 15 对后 finalize。

        标定结果仅注入 RobotDetector / SideDetector 比例尺与 side z 零位；
        不写入 magnetic_model、RuntimeContext 或 MotionObservation。
        """
        top_obs = getattr(top_result, "calibration_observation", None)
        side_obs = getattr(side_result, "calibration_observation", None)

        if pair_id % 30 == 0:
            print(
                "[DUAL CAL DEBUG] "
                f"pair={pair_id} "
                f"motion={bool(self.is_motion_active())} "
                f"pair_dt_ms={pair_dt_ms} "
                f"top_result={top_result is not None} "
                f"side_result={side_result is not None} "
                f"top_obs={top_obs is not None} "
                f"side_obs={side_obs is not None} "
                f"side_valid={getattr(side_result, 'valid', None)} "
                f"progress={self.dual_calibrator.progress}"
            )

        if self.dual_calibrator.completed:
            return
        if bool(self.is_motion_active()):
            return
        if pair_dt_ms is None:
            return
        if top_result is None or side_result is None:
            return
        if getattr(top_result, "scene_just_initialized", False):
            return
        if top_result.calibration_observation is None:
            return
        if side_result.calibration_observation is None:
            return

        # ============== 本次视觉修改：上视原点就绪检查开始 ==============
        # 作用：上视原点尚未建立时等待下一帧，避免提前完成标定。
        circle_position = self.robot_detector.circle_position
        if circle_position is None:
            return
        # ============== 本次视觉修改：上视原点就绪检查结束 ==============

        self.dual_calibrator.add_pair(
            pair_id=pair_id,
            pair_dt_ms=float(pair_dt_ms),
            top=top_result.calibration_observation,
            side=side_result.calibration_observation,
        )

        calibration_result = self.dual_calibrator.try_finalize()
        if calibration_result is None:
            current_error = self.dual_calibrator.last_error
            if current_error != self._last_calibration_error:
                if current_error is not None:
                    print(f"[DUAL CAL] error: {current_error}")
                self._last_calibration_error = current_error
            return

        self.robot_detector.set_scale_calibration(calibration_result.top_px_per_mm)

        # ============== 本次视觉修改：两路物理 Y 零点对齐开始 ==============
        # 作用：复用两路标定中心中位数，按上视 Y 坐标反算侧视 Y=0 的像素列。
        side_y_px_per_mm = float(calibration_result.side_y_px_per_mm)
        top_xy_cm = self.robot_detector.pixel_to_model_position(
            calibration_result.top_center_x_px,
            calibration_result.top_center_y_px,
            circle_position,
            calibration_result.top_px_per_cm,
        )
        y_cal_cm = float(top_xy_cm[1, 0])
        side_y_zero_x_px = float(
            calibration_result.side_center_x_px
            - SIDE_Y_SIGN * 10.0
            * calibration_result.side_y_px_per_mm
            * y_cal_cm
        )
        calibration_result = replace(
            calibration_result,
            side_y_zero_x_px=side_y_zero_x_px,
            side_y_sign=int(SIDE_Y_SIGN),
        )
        # ============== 本次视觉修改：两路物理 Y 零点对齐结束 ==============

        if self.side_detector is not None:
            # [3D_PID] 原 z=0 行和新增 y=0 列一起注入；SideDetector 内部仍将
            # 两种标定状态分开保存，旧 z/mm 公式不变。
            self.side_detector.set_calibration(
                # Side Z keeps using the measured side-view scale.
                px_per_mm=calibration_result.side_px_per_mm,
                ### 调整：向检测器传入独立 Y 比例尺，测量和叠图使用同一坐标映射。
                y_px_per_mm=side_y_px_per_mm,
                ### 独立 Y 比例尺注入结束。
                zero_center_y_px=calibration_result.side_zero_y_px,
                y_zero_center_x_px=calibration_result.side_y_zero_x_px,
                y_sign=calibration_result.side_y_sign,
            )

        self.dual_calibration_result = calibration_result
        self._last_calibration_error = None
        ### 侧视对齐零点注入结束；轨迹 (Y,Z)=(0,0) 投影到 (Y0-x,Z0-y)，Z 零点沿用落底标定。

        print("[DUAL CAL] completed")
        print(f"top diameter = {calibration_result.top_diameter_px:.2f} px")
        print(f"top scale = {calibration_result.top_px_per_mm:.2f} px/mm")
        # print(f"top scale = {calibration_result.top_px_per_cm:.2f} px/cm")
        print(f"side diameter = {calibration_result.side_diameter_px:.2f} px")
        ### 调整：完成日志分别显示侧视 Y/Z 比例尺，便于核对两个方向的倍率。
        print(f"side Y scale = {side_y_px_per_mm:.2f} px/mm")
        print(f"side Z scale = {calibration_result.side_px_per_mm:.2f} px/mm")
        ### Y/Z 比例尺日志结束。
        print(f"side y=0 x (aligned with top Y) = {calibration_result.side_y_zero_x_px:.2f} px")
        print(f"side physical z=0 y = {calibration_result.side_zero_y_px:.2f} px")
        print(f"accepted pairs = {calibration_result.accepted_pair_count}")
        print(f"inlier pairs = {calibration_result.inlier_pair_count}")

    def _draw_bottom_left_text(
        self,
        image: np.ndarray,
        lines: Sequence[str],
        *,
        color: Tuple[int, int, int],
    ) -> None:
        """在图像左下角自下而上绘制多行文字。"""
        height = image.shape[0]
        line_height = 28
        margin = 16
        for idx, line in enumerate(reversed(lines)):
            y = height - margin - idx * line_height
            cv2.putText(
                image,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_dual_calibration_overlay(
        self,
        *,
        top_main: Optional[np.ndarray],
        side_main: Optional[np.ndarray],
    ) -> None:
        """【双视·标定】在 top_main / side_main 左下角叠加标定状态（仅 UI）。"""
        cal = self.dual_calibrator
        required_count = cal.required_count
        color_calibrating = (0, 255, 255)
        color_done = (0, 255, 0)
        color_error = (0, 0, 255)

        if cal.completed or self.dual_calibration_result is not None:
            result = self.dual_calibration_result
            if result is None:
                return

            if top_main is not None:
                self._draw_bottom_left_text(
                    top_main,
                    [
                        "TOP CAL: DONE",
                        f"Scale: {result.top_px_per_mm:.2f} px/mm",
                        f"Scale: {result.top_px_per_cm:.2f} px/cm",
                    ],
                    color=color_done,
                )

            if side_main is not None:
                # [3D_PID] 明确区分 y=0 对应图像 x 列和 z=0 对应图像 y 行，
                # 防止调试时把两个不同方向的零点混用。
                side_lines = [
                    "SIDE CAL: DONE",
                    ### 调整：画面分别显示 Y/Z 比例尺，避免混用横向和竖向倍率。
                    f"Y Scale: {result.side_y_px_per_mm:.2f} px/mm",
                    f"Z Scale: {result.side_px_per_mm:.2f} px/mm",
                    ### Y/Z 比例尺画面显示结束。
                    f"Y0-x: {result.side_y_zero_x_px:.2f} px",
                    f"Z0-y: {result.side_zero_y_px:.2f} px",
                ]
                z_mm = None
                if self.latest_side_result is not None:
                    z_mm = getattr(self.latest_side_result, "z_mm", None)
                if z_mm is None:
                    side_lines.append("Z: N/A")
                else:
                    side_lines.append(f"Z: {z_mm:+.2f} mm")
                self._draw_bottom_left_text(
                    side_main,
                    side_lines,
                    color=color_done,
                )
            return

        last_error = cal.last_error
        if last_error:
            progress_lines = [
                f"DUAL CAL: {cal.progress} / {required_count}",
                str(last_error),
            ]
            overlay_color = color_error
        else:
            progress_lines = [
                f"DUAL CAL: {cal.progress} / {required_count}",
                "KEEP BEAD STILL ON BOTTOM",
            ]
            overlay_color = color_calibrating

        if top_main is not None:
            self._draw_bottom_left_text(
                top_main,
                progress_lines,
                color=overlay_color,
            )
        if side_main is not None:
            self._draw_bottom_left_text(
                side_main,
                progress_lines,
                color=overlay_color,
            )

    @staticmethod
    def _resize_bgr(frame: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
        """将 BGR 帧缩放到固定显示尺寸；空帧则返回黑图。"""
        if frame is None:
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[0] == int(height) and arr.shape[1] == int(width):
            return arr
        return cv2.resize(arr, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    def _compose_dual_view_frame(
        self,
        top_main: Optional[np.ndarray],
        side_main: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        合成上下双视角录像帧：上=上视主画面，下=侧视主画面。
        输出尺寸 = (display_width, 2 * display_height)。
        """
        w = int(self.cfg.display_width)
        h = int(self.cfg.display_height)
        top = self._resize_bgr(top_main, w, h)
        side = self._resize_bgr(side_main, w, h)
        return np.vstack([top, side])

    def _finalize_video_recording(
        self,
        video_output,
        video_path: str,
        video_info_path: str,
        record_start_time: float,
        written_frames: int,
        session_info: Dict[str, str],
    ) -> dict:
        """
        释放 VideoWriter，并按方案 A 写入 _video_info.txt。
        """
        declared_fps = float(self.cfg.fps)
        try:
            if video_output is not None:
                video_output.release()
        except Exception as e:
            print(f"[WARN] VideoWriter.release failed: {e}")

        real_duration_s = max(0.0, float(time.perf_counter()) - float(record_start_time))
        actual_write_fps = (
            float(written_frames) / real_duration_s if real_duration_s > 1e-9 else 0.0
        )
        estimated_video_duration_s = (
            float(written_frames) / declared_fps if declared_fps > 1e-9 else 0.0
        )
        playback_speed_ratio = (
            actual_write_fps / declared_fps if declared_fps > 1e-9 else 0.0
        )
        vlc_real_time_rate = playback_speed_ratio
        speed_at_vlc_half_rate = 0.5 * playback_speed_ratio

        stats = {
            "declared_fps": declared_fps,
            "written_frames": int(written_frames),
            "real_duration_s": real_duration_s,
            "actual_write_fps": actual_write_fps,
            "estimated_video_duration_s": estimated_video_duration_s,
            "playback_speed_ratio": playback_speed_ratio,
            "vlc_real_time_rate": vlc_real_time_rate,
            "speed_at_vlc_half_rate": speed_at_vlc_half_rate,
        }

        info_path = Path(video_info_path)
        try:
            info_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session_dir={session_info.get('session_dir', '')}",
                f"mode_name={session_info.get('mode_name', '')}",
                f"timestamp={session_info.get('timestamp', '')}",
                f"video_path={session_info.get('video_path', video_path)}",
                f"video_info_path={session_info.get('video_info_path', video_info_path)}",
                f"pkl_path={session_info.get('pkl_path', '')}",
                f"declared_fps={declared_fps:.6f}",
                f"written_frames={int(written_frames)}",
                f"real_duration_s={real_duration_s:.6f}",
                f"actual_write_fps={actual_write_fps:.6f}",
                f"estimated_video_duration_s={estimated_video_duration_s:.6f}",
                f"playback_speed_ratio={playback_speed_ratio:.6f}",
                f"vlc_real_time_rate={vlc_real_time_rate:.6f}",
                f"speed_at_vlc_half_rate={speed_at_vlc_half_rate:.6f}",
            ]
            info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[WARN] write video_info failed: {e}")

        return stats

    def _handle_drive_close_request(self) -> None:
        """
        在视觉线程内归零并关闭线圈串口。
        必须在 RuntimeLoop.step() 之外调用，避免与控制步并发访问串口。
        """
        success = False
        message = ""
        devices = getattr(self.runtime_ctx, "devices", None)
        if devices is None:
            devices = getattr(self.runtime_loop, "devices", None)

        try:
            if devices is None:
                raise RuntimeError("devices is None; cannot close drive")
            devices.close_drive()
            success = True
        except Exception as exc:
            message = str(exc)
            try:
                drive = getattr(devices, "drive", None) if devices is not None else None
                if drive is not None:
                    close_fn = getattr(drive, "close_serial", None)
                    if callable(close_fn):
                        close_fn()
            except Exception as close_exc:
                if message:
                    message += " | "
                message += f"fallback serial close failed: {close_exc}"
        finally:
            if self.on_drive_close_finished is not None:
                self.on_drive_close_finished(success, message)

    def run(self):
        """
        【双视·共用】视觉线程主循环（阻塞至 camera_state 为 False）。

        启动顺序：
            1.【上视】打开 OpenCVCamera（失败则整个 run 退出）
            2.【侧视】尝试打开 HikSideCamera（失败则上视继续，侧视显示 OFFLINE）

        录像：
            仅当 is_video_enabled() 且主线程已准备实验会话路径时，
            创建上下双视角 AVI；路径由 get_experiment_session_info() 提供，
            视觉线程不再自行生成时间戳。

        每帧顺序：
            取上视 → 取侧视 → 处理上视 → 构建侧视 bundle → 合并显示
            →（可选）写录像 → 控制步进

        退出时 finally 释放上视相机、侧视相机与活动录像句柄。
        """
        mm = self.runtime_ctx.magnetic_model

        # ----- 【上视】打开 OpenCV 相机 -----
        camera = OpenCVCamera(
            camera_index=self.cfg.camera_index,
            cfg=CameraConfig(
                fps=self.cfg.fps,
                width=self.cfg.width,
                height=self.cfg.height,
                flip_horizontal=self.cfg.flip_horizontal,
            ),
        )

        if not camera.open():
            print("错误：相机未打开！请检查 Camera_index 或设备连接。")
            return

        camera.log_properties()

        # ----- 【侧视】打开海康相机（失败不阻断上视）-----
        if self.side_camera is not None:
            if self.side_camera.open():
                self._side_camera_online = True
                print("[VisionControlLoop] 侧视相机已就绪")
            else:
                self._side_camera_online = False
                last_error = getattr(self.side_camera, "last_error", "unknown")
                print(
                    f"[VisionControlLoop] 侧视相机打开失败: {last_error}；"
                    "上视功能继续，侧视区域显示 SIDE CAMERA OFFLINE"
                )

        self.reset_dual_calibration()

        video_output = None
        video_path = ""
        video_info_path = ""
        active_session_info: Dict[str, str] = {}
        record_start_time = 0.0
        written_frames = 0

        try:
            while self.is_camera_active():
                # ----- 【上视】取彩色帧 -----
                ret, cv_img = camera.read()
                if not ret or cv_img is None:
                    continue
                top_timestamp = time.perf_counter()

                # ----- 【侧视】紧接上视取 Mono8 帧（禁止异步缓存配对）-----
                side_frame = None
                if self._side_camera_online and self.side_camera is not None:
                    timeout_ms = int(self._get_loop_cfg("side_camera_timeout_ms", 50))
                    side_frame = self.side_camera.grab_gray(timeout_ms=timeout_ms)

                # ----- 【双视·共用】仅有侧视帧时生成新 pair；显示可用最近 pair -----
                new_pair_id = None
                display_pair_id = self.pair_id
                if side_frame is not None:
                    self.pair_id += 1
                    new_pair_id = self.pair_id
                    display_pair_id = new_pair_id

                # ----- 【上视】检测 + overlay → top_* bundle -----
                frame_show, frame_bundle, detect_time, top_result = self._process_vision_frame(
                    cv_img, mm
                )

                # ----- 【侧视】检测或占位 → side_* bundle，合并进同一 dict -----
                side_bundle, side_result, pair_dt_ms = self._build_side_frame_bundle(
                    side_frame=side_frame,
                    top_timestamp=top_timestamp,
                    pair_id=display_pair_id,
                )
                frame_bundle.update(side_bundle)

                self.latest_side_result = side_result
                self.latest_top_result = top_result

                if new_pair_id is not None:
                    self._update_dual_calibration(
                        pair_id=new_pair_id,
                        pair_dt_ms=pair_dt_ms,
                        top_result=top_result,
                        side_result=side_result,
                    )
                    self._update_3d_measured_state_from_pair(
                        pair_id=new_pair_id,
                        top_timestamp=top_timestamp,
                        pair_dt_ms=pair_dt_ms,
                        top_result=top_result,
                        side_result=side_result,
                    )

                # [3D_PID] BEGIN: 侧视 yz 叠图接入录像前处理链。
                # 当前模式通过 ModeManager 仅发布 target_xyz/ref_path_3d；这里在
                # side_main 合成录像之前绘制，所以既有 VideoWriter 会自动记录。
                # side_result 必须是本轮真实侧视帧，缺帧占位图不追加旧 pair 轨迹。
                overlay_state = self.runtime_loop.mode_manager.get_overlay_state()
                if (
                    self.get_control_mode() == "stabilize_3d_mode"
                    and self.side_detector is not None
                    and side_result is not None
                    and frame_bundle.get("side_main") is not None
                ):
                    frame_bundle["side_main"] = self.side_overlay_renderer.render(
                        frame_bundle["side_main"],
                        side_detector=self.side_detector,
                        magnetic_model=mm,
                        motion_active=bool(self.is_motion_active()),
                        target_xyz=overlay_state.get("target_xyz"),
                        ref_path_3d=overlay_state.get("ref_path_3d"),
                    )
                # [3D_PID] END: 侧视 yz 叠图。

                if "top_main" in frame_bundle and frame_bundle["top_main"] is not None:
                    frame_bundle["top_main"] = frame_bundle["top_main"].copy()
                if "side_main" in frame_bundle and frame_bundle["side_main"] is not None:
                    frame_bundle["side_main"] = frame_bundle["side_main"].copy()
                self._draw_dual_calibration_overlay(
                    top_main=frame_bundle.get("top_main"),
                    side_main=frame_bundle.get("side_main"),
                )

                # ----- 【录像】按 video_state 启停；路径由主线程实验会话提供 -----
                video_requested = bool(self.is_video_enabled())
                if video_requested and video_output is None:
                    session_info = dict(self.get_experiment_session_info() or {})
                    required_keys = (
                        "mode_name",
                        "timestamp",
                        "session_dir",
                        "video_path",
                        "video_info_path",
                        "pkl_path",
                    )
                    missing = [k for k in required_keys if not session_info.get(k)]
                    if missing:
                        reason = f"missing session fields: {missing}"
                        print(f"[Video] refuse to start recording: {reason}")
                        if self.on_video_recording_failed is not None:
                            self.on_video_recording_failed(reason)
                    else:
                        try:
                            session_dir_relative = Path(session_info["session_dir"])
                            video_path_relative = Path(session_info["video_path"])
                            video_info_path_relative = Path(
                                session_info["video_info_path"]
                            )

                            # 绝对路径仅用于安全检查与诊断打印，不传给 VideoWriter
                            session_dir_absolute = session_dir_relative.resolve()
                            video_path_absolute = video_path_relative.resolve()
                            video_info_path_absolute = (
                                video_info_path_relative.resolve()
                            )

                            if video_path_absolute.parent != session_dir_absolute:
                                raise ValueError(
                                    "Video path is outside session directory"
                                )
                            if (
                                video_info_path_absolute.parent
                                != session_dir_absolute
                            ):
                                raise ValueError(
                                    "Video info path is outside session directory"
                                )

                            video_path_relative.parent.mkdir(
                                parents=True,
                                exist_ok=True,
                            )
                            if not video_path_relative.parent.is_dir():
                                raise RuntimeError(
                                    "Experiment video directory does not exist: "
                                    f"{video_path_relative.parent}"
                                )

                            frame_w = int(self.cfg.display_width)
                            frame_h = int(self.cfg.display_height) * 2
                            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
                            # 必须传纯 ASCII 相对路径，禁止中文绝对路径
                            candidate_path = str(video_path_relative)

                            print(
                                "[Video] opening writer:\n"
                                f"  writer_path={candidate_path}\n"
                                f"  absolute_path="
                                f"{video_path_absolute}\n"
                                f"  directory_exists="
                                f"{video_path_relative.parent.is_dir()}\n"
                                f"  fps={float(self.cfg.fps)}\n"
                                f"  size=("
                                f"{frame_w}, "
                                f"{frame_h})"
                            )

                            writer = cv2.VideoWriter(
                                candidate_path,
                                fourcc,
                                float(self.cfg.fps),
                                (frame_w, frame_h),
                            )
                            if not writer.isOpened():
                                diagnostic_path = (
                                    Path("recordings")
                                    / "_video_writer_diagnostic.avi"
                                )
                                diagnostic_path.parent.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )
                                diagnostic_writer = cv2.VideoWriter(
                                    str(diagnostic_path),
                                    fourcc,
                                    float(self.cfg.fps),
                                    (frame_w, frame_h),
                                )
                                diagnostic_ok = diagnostic_writer.isOpened()
                                diagnostic_writer.release()
                                try:
                                    if diagnostic_path.exists():
                                        diagnostic_path.unlink()
                                except OSError:
                                    pass
                                raise RuntimeError(
                                    "VideoWriter open failed: "
                                    f"{candidate_path}; "
                                    f"ASCII diagnostic path opened="
                                    f"{diagnostic_ok}"
                                )

                            dual0 = self._compose_dual_view_frame(
                                frame_bundle.get("top_main"),
                                frame_bundle.get("side_main"),
                            )
                            writer.write(dual0)
                            video_output = writer
                            video_path = candidate_path
                            video_info_path = str(video_info_path_relative)
                            # 说明文件可记录绝对路径便于查看；不用于 VideoWriter
                            active_session_info = dict(session_info)
                            active_session_info["session_dir"] = str(
                                session_dir_absolute
                            )
                            active_session_info["video_path"] = str(
                                video_path_absolute
                            )
                            active_session_info["video_info_path"] = str(
                                video_info_path_absolute
                            )
                            if session_info.get("pkl_path"):
                                active_session_info["pkl_path"] = str(
                                    Path(session_info["pkl_path"]).resolve()
                                )
                            record_start_time = time.perf_counter()
                            written_frames = 1
                            print(f"[Video] recording started: {video_path}")
                            if self.on_video_recording_started is not None:
                                self.on_video_recording_started(video_path)
                        except Exception as e:
                            reason = str(e)
                            print(f"[Video] start failed: {reason}")
                            video_output = None
                            video_path = ""
                            video_info_path = ""
                            active_session_info = {}
                            record_start_time = 0.0
                            written_frames = 0
                            if self.on_video_recording_failed is not None:
                                self.on_video_recording_failed(reason)

                elif (not video_requested) and video_output is not None:
                    stats = self._finalize_video_recording(
                        video_output=video_output,
                        video_path=video_path,
                        video_info_path=video_info_path,
                        record_start_time=record_start_time,
                        written_frames=written_frames,
                        session_info=active_session_info,
                    )
                    finished_path = video_path
                    video_output = None
                    video_path = ""
                    video_info_path = ""
                    active_session_info = {}
                    record_start_time = 0.0
                    written_frames = 0
                    print(f"[Video] recording stopped: {finished_path}")
                    if self.on_video_recording_stopped is not None:
                        self.on_video_recording_stopped(finished_path, stats)

                elif video_requested and video_output is not None:
                    dual = self._compose_dual_view_frame(
                        frame_bundle.get("top_main"),
                        frame_bundle.get("side_main"),
                    )
                    video_output.write(dual)
                    written_frames += 1

                # ----- 【双视·共用】一次回调刷新全部 QLabel -----
                self.on_frame_ready(frame_bundle)

                # ----- 【线圈串口】在下一控制步之前独占处理关闭请求 -----
                if self.is_drive_close_requested():
                    if not self._drive_close_already_handled:
                        self._handle_drive_close_request()
                        self._drive_close_already_handled = True
                    # 关闭串口这一轮不得再执行 RuntimeLoop.step()
                    continue
                self._drive_close_already_handled = False

                # ----- 【控制链】直接步进 -----
                step_result = self.runtime_loop.step(
                    motion_active=bool(self.is_motion_active()),
                    i_motion=int(self.get_i_motion()),
                    frame=cv_img,
                    control_mode=self.get_control_mode(),
                    drive_enabled=bool(self.is_drive_enabled()),
                )

                self.set_i_motion(step_result.i_motion)

                if step_result.motion_active:
                    print(
                        f"detect_time:{detect_time}, "
                        f"control_time:{step_result.control_time}, "
                        f"drive_time:{step_result.drive_time}"
                    )
                else:
                    print(f"detect_time:{detect_time}")
        finally:
            camera.release()
            if self.side_camera is not None:
                self.side_camera.close()
            if video_output is not None:
                stats = self._finalize_video_recording(
                    video_output=video_output,
                    video_path=video_path,
                    video_info_path=video_info_path,
                    record_start_time=record_start_time,
                    written_frames=written_frames,
                    session_info=active_session_info,
                )
                if self.on_video_recording_stopped is not None:
                    self.on_video_recording_stopped(video_path, stats)
            cv2.destroyAllWindows()

    def _process_vision_frame(self, cv_img, mm):
        """
        【上视】单帧上视检测、overlay 渲染与 top_* frame_bundle 组装。

        本函数不涉及侧视相机与 SideDetector。
        [3D] RobotDetector 只返回原始 xy（model_position_xy_cm），不再写模型；
        模型更新在 run() 中经 _update_3d_measured_state_from_pair 统一完成。

        返回：
            frame_show — 叠加后的上视主画面（供录像）；
            frame_bundle — 仅含 top_* 五个 key；
            detect_time — 上视视觉处理耗时（秒）；
            det_result — RobotDetector.process_frame 的原始检测结果。
        """
        time_1 = time.perf_counter()

        # ----- 【上视】RobotDetector：二值化 / 轮廓 / 原始 xy 测量 -----
        det_result = self.robot_detector.process_frame(cv_img)

        cv_img = det_result.annotated_bgr if det_result.annotated_bgr is not None else cv_img
        frame_show = cv_img.copy()

        if det_result.circle_position is not None:
            self.circle_position = det_result.circle_position

        self.circle_state = self.robot_detector.calibrated

        if det_result.scene_just_initialized:
            print("img center: ", self.circle_position)
            self.overlay_renderer.reset()
            if self.on_scene_initialized is not None:
                self.on_scene_initialized()

        if det_result.roi_rect is not None:
            self.pt1, self.pt2 = det_result.roi_rect

        px_per_cm = det_result.px_per_cm
        roi_rect = (self.pt1, self.pt2) if (self.pt1 is not None and self.pt2 is not None) else None
        # [3D_PID] BEGIN: 向原上视 OverlayRenderer 注入三维显示状态。
        # 模式通过管理器发布目标与 (N,3) 参考路径；实际轨迹按有效 pair 增量生成。
        overlay_state = self.runtime_loop.mode_manager.get_overlay_state()

        # ----- 【上视】OverlayRenderer：三维参考投影 / 实际轨迹 / 目标点 -----
        frame_show = self.overlay_renderer.render(
            frame_show,
            circle_position=self.circle_position if self.circle_state else det_result.circle_position,
            circle_state=self.circle_state,
            px_per_cm=px_per_cm,
            roi_rect=roi_rect,
            frame_shape=cv_img.shape,
            magnetic_model=mm,
            robot_centers_px=det_result.robot_centers_px,
            motion_state=bool(self.is_motion_active()),
            control_mode=self.get_control_mode(),
            target_xyz=overlay_state.get("target_xyz"),
            ref_path_3d=overlay_state.get("ref_path_3d"),
        )
        # [3D_PID] END: 上视 xy 叠图状态注入。

        time_2 = time.perf_counter()

        # ----- 【上视】调试图可视化素材 -----
        eroded_roi_vis = det_result.eroded_roi_mask
        if eroded_roi_vis is not None:
            if eroded_roi_vis.ndim == 2:
                eroded_roi_vis = cv2.cvtColor(eroded_roi_vis, cv2.COLOR_GRAY2BGR)
            if det_result.roi_rect is not None:
                eroded_roi_vis = eroded_roi_vis.copy()
                pt1, pt2 = det_result.roi_rect
                cv2.rectangle(eroded_roi_vis, pt1, pt2, (0, 255, 255), 2)

        frame_bundle = {
            "top_main": frame_show,
            "top_binary": det_result.binary_mask,
            "top_eroded_roi": eroded_roi_vis,
            "top_candidates": det_result.candidate_overlay_bgr,
            "top_filter_result": det_result.filter_result_overlay_bgr,
        }

        return frame_show, frame_bundle, time_2 - time_1, det_result
