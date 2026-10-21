"""
文件功能：
    上视三维参考路径的 XY 投影、有效测量轨迹、机器人位置与目标点。

设计原则：
    1. 只负责图像绘制与 blend，不执行控制算法、不下发电流；
    2. 保留现有颜色、线宽与透明度；
    3. magnetic_model 仅读取三维测量位置、有效性及 pair 编号。

与其他模块关系：
    被 app/vision_control_loop.py 调用。
    内部依赖 trajectory_overlay 模块，不 import motion.py、motion_modes 或 devices。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision import trajectory_overlay
from vision.trajectory_overlay import OverlayStyle


@dataclass
class OverlayRendererConfig:
    """
    叠加渲染配置。

    注意：
        参数与 main.py 原 detect() 中硬编码值对齐。
    """

    # 三维参考路径 blend 透明度
    static_alpha: float = 0.78


class OverlayRenderer:
    """
    图像叠加渲染器（有状态：静态层缓存与轨迹层）。

    功能：
        1. 绘制并缓存三维参考路径的 XY 投影；
        2. 追加并混合有效测量 pair 的运动轨迹；
        3. 绘制机器人检测中心、ROI 与当前三维目标。
    """

    def __init__(
        self,
        style: Optional[OverlayStyle] = None,
        cfg: Optional[OverlayRendererConfig] = None,
    ):
        """
        参数：
            style: trajectory_overlay 样式；None 时使用 OverlayStyle 默认；
            cfg: 渲染配置；None 时使用 OverlayRendererConfig 默认。
        """
        self.style = style if style is not None else OverlayStyle()
        self.cfg = cfg if cfg is not None else OverlayRendererConfig()

        self.static_overlay_bgr: Optional[np.ndarray] = None
        self.static_overlay_mask: Optional[np.ndarray] = None
        self.trace_overlay_bgr: Optional[np.ndarray] = None
        self.trace_overlay_mask: Optional[np.ndarray] = None
        # stabilize_3d 专用 XY 静态层与增量轨迹；按有效 pair 消费测量。
        self.stabilize_overlay_cache_key: Optional[tuple] = None
        self.stabilize_last_pair_id = None
        self.stabilize_last_trace_pixel: Optional[Tuple[int, int]] = None
        self.stabilize_was_motion_active = False
        # [3D_PID] END: 三维模式上视叠图状态。

    def reset(self) -> None:
        """清除静态层与轨迹层缓存（场景标定变化或重新开始 motion 时调用）。"""
        self.static_overlay_bgr = None
        self.static_overlay_mask = None
        self.trace_overlay_bgr = None
        self.trace_overlay_mask = None
        # [3D_PID] reset 必须同步清除新模式缓存、pair 和上一轨迹点，避免新实验
        # 连接到上一次 stabilize_3d 实验的末点。
        self.stabilize_overlay_cache_key = None
        self.stabilize_last_pair_id = None
        self.stabilize_last_trace_pixel = None
        self.stabilize_was_motion_active = False

    # [3D_PID] BEGIN: 将严格 (N,3) cm 参考路径投影到上视 xy 平面。
    # 使用与 RobotDetector 相反方向一致的映射：图像列由模型 y 决定，
    # 图像行由模型 x 决定；z 在上视投影中不参与绘制。
    @staticmethod
    def project_path_3d_to_top_pixels(
        ref_path_3d: Any,
        circle_position: Sequence[int],
        px_per_cm: float,
    ) -> np.ndarray:
        if ref_path_3d is None:
            return np.empty((0, 2), dtype=np.int32)
        path = np.asarray(ref_path_3d, dtype=float)
        if path.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"ref_path_3d must have shape (N, 3), got {path.shape}")
        if not np.all(np.isfinite(path)):
            raise ValueError("ref_path_3d contains a non-finite value")
        pixels = [
            OverlayRenderer._cm_point_to_px(
                circle_position, px_per_cm, point[:2]
            )
            for point in path
        ]
        return np.asarray(pixels, dtype=np.int32)

    def _ensure_stabilize_layers(
        self,
        frame_shape: tuple,
        circle_position: Sequence[int],
        px_per_cm: float,
        ref_path_3d: Any,
    ) -> None:
        path = None if ref_path_3d is None else np.asarray(ref_path_3d, dtype=float)
        # [3D_PID] 路径由模式验证为只读，并在内容变化时整体替换，因此可用对象
        # 身份做 O(1) 缓存键；只有路径、标定、画面尺寸变化时才重建静态层。
        path_key = None if path is None or path.size == 0 else (id(ref_path_3d), path.shape)
        key = (
            int(frame_shape[0]),
            int(frame_shape[1]),
            int(circle_position[0]),
            int(circle_position[1]),
            float(px_per_cm),
            path_key,
        )
        if self.static_overlay_bgr is not None and key == self.stabilize_overlay_cache_key:
            return
        height, width = frame_shape[:2]
        self.static_overlay_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        self.static_overlay_mask = np.zeros((height, width), dtype=np.uint8)
        self.trace_overlay_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        self.trace_overlay_mask = np.zeros((height, width), dtype=np.uint8)
        pixels = self.project_path_3d_to_top_pixels(
            ref_path_3d, circle_position, px_per_cm
        )
        if len(pixels) >= 2:
            cv2.polylines(
                self.static_overlay_bgr, [pixels], False, self.style.ref_color,
                self.style.ref_thickness, cv2.LINE_AA,
            )
            cv2.polylines(
                self.static_overlay_mask, [pixels], False, 255,
                max(1, self.style.ref_thickness), cv2.LINE_AA,
            )
        self.stabilize_overlay_cache_key = key
        self.stabilize_last_pair_id = None
        self.stabilize_last_trace_pixel = None

    def _append_stabilize_trace(
        self,
        magnetic_model: Any,
        circle_position: Sequence[int],
        px_per_cm: float,
    ) -> None:
        pair_id = getattr(magnetic_model, "measurement_pair_id", None)
        mask = np.asarray(
            getattr(magnetic_model, "measurement_valid_mask", np.zeros(3, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        # [3D_PID] 实际轨迹只消费新 pair，并要求 x/y/z 全部有效；重复或任一轴
        # 无效时不画线，避免把高频渲染循环误当成新的视觉测量。
        if pair_id is None or pair_id == self.stabilize_last_pair_id:
            return
        self.stabilize_last_pair_id = pair_id
        if mask.size != 3 or not bool(np.all(mask)):
            return
        position = np.asarray(magnetic_model.return_position(), dtype=float).reshape(3,)
        pixel = self._cm_point_to_px(circle_position, px_per_cm, position[:2])
        if (
            self.stabilize_last_trace_pixel is not None
            and self.trace_overlay_bgr is not None
            and self.trace_overlay_mask is not None
        ):
            trajectory_overlay.append_trace_segment(
                self.trace_overlay_bgr,
                self.trace_overlay_mask,
                self.stabilize_last_trace_pixel,
                pixel,
                color=self.style.trace_color,
                thickness=self.style.trace_thickness,
            )
        self.stabilize_last_trace_pixel = pixel
    # [3D_PID] END: stabilize_3d 上视投影、缓存与增量轨迹实现。

    @staticmethod
    def _cm_point_to_px(
        circle_position: Sequence[int],
        px_per_cm: float,
        point_cm: Any,
    ) -> Tuple[int, int]:
        """
        物理平面坐标 (cm) → 像素坐标。

        与原 detect() 一致：x/y 对调后乘以 px_per_cm 再加原点偏移。
        """
        pt = np.asarray(point_cm, dtype=float).reshape(2,)
        px = int(circle_position[0] + px_per_cm * pt[1])
        py = int(circle_position[1] + px_per_cm * pt[0])
        return px, py

    def _blend_overlays(self, frame_show: np.ndarray) -> np.ndarray:
        """将静态层与轨迹层混合到显示帧。"""
        if self.static_overlay_bgr is not None and self.static_overlay_mask is not None:
            frame_show = trajectory_overlay.blend_masked_overlay(
                frame_show,
                self.static_overlay_bgr,
                self.static_overlay_mask,
                alpha=self.cfg.static_alpha,
            )
        if self.trace_overlay_bgr is not None and self.trace_overlay_mask is not None:
            frame_show = trajectory_overlay.blend_masked_overlay(
                frame_show,
                self.trace_overlay_bgr,
                self.trace_overlay_mask,
                alpha=1.0,
            )
        return frame_show

    @staticmethod
    def _draw_robot_centers(
        frame_show: np.ndarray,
        robot_centers_px: Sequence[Tuple[int, int]],
    ) -> None:
        """绘制机器人检测中心（外圈蓝、内点黄）。"""
        for cx, cy in robot_centers_px:
            cv2.circle(frame_show, (cx, cy), 6, (255, 0, 0), 2)
            cv2.circle(frame_show, (cx, cy), 2, (255, 255, 0), -1)

    def render(
        self,
        frame_show: np.ndarray,
        *,
        circle_position: Optional[Sequence[int]],
        circle_state: bool,
        px_per_cm: float,
        roi_rect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        frame_shape: tuple,
        magnetic_model: Any,
        robot_centers_px: Sequence[Tuple[int, int]],
        motion_state: bool,
        control_mode: str,
        # [3D_PID] 由 ModeManager.get_overlay_state() 提供，仅用于绘图。
        target_xyz: Any = None,
        ref_path_3d: Any = None,
    ) -> np.ndarray:
        """绘制当前模式的参考投影、有效测量轨迹、中心、ROI 与目标。"""
        # 三维模式绘制 XY 参考投影和有效 pair 轨迹。
        if control_mode == "stabilize_3d_mode":
            if circle_position is None or not circle_state:
                return frame_show
            self._ensure_stabilize_layers(
                frame_shape,
                circle_position,
                px_per_cm,
                ref_path_3d,
            )
            if motion_state:
                if not self.stabilize_was_motion_active:
                    # Top rendering precedes this loop's stereo state update.
                    # Mark the pre-motion pair as consumed so it cannot seed
                    # an experiment trace; the next updated pair starts it.
                    self.stabilize_last_pair_id = getattr(
                        magnetic_model, "measurement_pair_id", None
                    )
                self._append_stabilize_trace(
                    magnetic_model, circle_position, px_per_cm
                )
            self.stabilize_was_motion_active = bool(motion_state)
            frame_show = self._blend_overlays(frame_show)
            self._draw_robot_centers(frame_show, robot_centers_px)
            # [3D_PID] stabilize 上视主画面也画出检测 ROI（与调试图同色：黄框）。
            # roi_rect 已由 vision_control_loop 传入，此处直接复用，不改检测逻辑。
            if roi_rect is not None:
                pt1, pt2 = roi_rect
                cv2.rectangle(frame_show, tuple(pt1), tuple(pt2), (0, 255, 255), 2)
            if motion_state and target_xyz is not None:
                target = np.asarray(target_xyz, dtype=float).reshape(3,)
                if np.all(np.isfinite(target)):
                    px, py = self._cm_point_to_px(
                        circle_position, px_per_cm, target[:2]
                    )
                    cv2.circle(frame_show, (px, py), 6, (0, 0, 255), -1)
            return frame_show

        # 保留模式分派边界，未来新增模式不会复用三维叠图。
        self.stabilize_was_motion_active = False
        return frame_show
