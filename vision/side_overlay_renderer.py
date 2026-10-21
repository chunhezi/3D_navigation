"""[3D_PID] Cached y-z trajectory/target overlay for the side camera."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from vision import trajectory_overlay


# [3D_PID] BEGIN: stabilize_3d 侧视 yz 投影、缓存和增量轨迹。
# 本类只绘图，不读取 PID 参数、不产生控制力，也不写 magnetic_model。
class SideOverlayRenderer:
    """Render optional ``(N,3)`` references and incremental measured y-z trace."""

    def __init__(self) -> None:
        self.static_overlay_bgr: Optional[np.ndarray] = None
        self.static_overlay_mask: Optional[np.ndarray] = None
        self.trace_overlay_bgr: Optional[np.ndarray] = None
        self.trace_overlay_mask: Optional[np.ndarray] = None
        self.cache_key: Optional[tuple] = None
        self.last_pair_id = None
        self.last_trace_pixel: Optional[tuple[int, int]] = None
        self._was_motion_active = False

    def reset(self) -> None:
        self.static_overlay_bgr = None
        self.static_overlay_mask = None
        self.trace_overlay_bgr = None
        self.trace_overlay_mask = None
        self.cache_key = None
        self.last_pair_id = None
        self.last_trace_pixel = None

    @staticmethod
    def _validated_path(ref_path_3d) -> Optional[np.ndarray]:
        if ref_path_3d is None:
            return None
        path = np.asarray(ref_path_3d, dtype=float)
        if path.size == 0:
            return None
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"ref_path_3d must have shape (N, 3), got {path.shape}")
        if not np.all(np.isfinite(path)):
            raise ValueError("ref_path_3d contains a non-finite value")
        return path

    @staticmethod
    def _calibration_key(side_detector: Any) -> tuple:
        return (
            float(side_detector.y_zero_center_x_px),
            float(side_detector.zero_center_y_px),
            float(side_detector.side_px_per_cm),
            ### 新增：Y 比例尺也参与缓存判定，改变后重建参考路径和历史轨迹叠图。
            float(side_detector.side_y_px_per_cm),
            ### Y 比例尺缓存判定结束。
            int(side_detector.y_sign),
        )

    @staticmethod
    def project_path_to_pixels(ref_path_3d, side_detector: Any) -> np.ndarray:
        path = SideOverlayRenderer._validated_path(ref_path_3d)
        if path is None:
            return np.empty((0, 2), dtype=np.int32)
        ### 侧视叠图：直接按轨迹 Y/Z 坐标投影，(0,0) 对应侧视 Y/Z 零点交点。
        pixels = [
            side_detector.yz_cm_to_pixel(float(point[1]), float(point[2]))
            for point in path
        ]
        ### 侧视轨迹投影结束；非零起点保留轨迹自身的坐标偏移。
        return np.rint(np.asarray(pixels, dtype=float)).astype(np.int32)

    def _ensure_layers(
        self,
        frame_shape: tuple,
        side_detector: Any,
        ref_path_3d,
    ) -> None:
        if ref_path_3d is None:
            path = None
        else:
            path = np.asarray(ref_path_3d, dtype=float)
            if path.size == 0:
                path = None
            elif path.ndim != 2 or path.shape[1] != 3:
                raise ValueError(
                    f"ref_path_3d must have shape (N, 3), got {path.shape}"
                )
        # [3D_PID] The mode exposes a read-only array and replaces it when the path
        # changes, so identity is a constant-time cache key on normal frames.
        path_key = None if path is None else (id(path), path.shape)
        key = (
            int(frame_shape[0]),
            int(frame_shape[1]),
            self._calibration_key(side_detector),
            path_key,
        )
        if self.static_overlay_bgr is not None and key == self.cache_key:
            return
        if path is not None and not np.all(np.isfinite(path)):
            raise ValueError("ref_path_3d contains a non-finite value")

        height, width = frame_shape[:2]
        self.static_overlay_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        self.static_overlay_mask = np.zeros((height, width), dtype=np.uint8)
        self.trace_overlay_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        self.trace_overlay_mask = np.zeros((height, width), dtype=np.uint8)
        self.last_pair_id = None
        self.last_trace_pixel = None

        pixels = self.project_path_to_pixels(path, side_detector)
        if len(pixels) >= 2:
            cv2.polylines(
                self.static_overlay_bgr,
                [pixels],
                False,
                (20, 109, 91),
                2,
                cv2.LINE_AA,
            )
            cv2.polylines(
                self.static_overlay_mask,
                [pixels],
                False,
                255,
                2,
                cv2.LINE_AA,
            )
        self.cache_key = key

    def _append_current_pair(self, magnetic_model: Any, side_detector: Any) -> None:
        pair_id = getattr(magnetic_model, "measurement_pair_id", None)
        mask = np.asarray(
            getattr(
                magnetic_model,
                "measurement_valid_mask",
                np.zeros(3, dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
        # [3D_PID] 只对新且三轴有效的 pair 追加线段；无效 pair 也会被消费，
        # 避免同一个坏 pair 在后续高频渲染中被重复检查或错误补画。
        if pair_id is None or pair_id == self.last_pair_id:
            return
        self.last_pair_id = pair_id
        if mask.size != 3 or not bool(np.all(mask)):
            return

        position = np.asarray(
            magnetic_model.return_position(), dtype=float
        ).reshape(3,)
        u, v = side_detector.yz_cm_to_pixel(position[1], position[2])
        current_pixel = (int(round(u)), int(round(v)))
        if (
            self.last_trace_pixel is not None
            and self.trace_overlay_bgr is not None
            and self.trace_overlay_mask is not None
        ):
            trajectory_overlay.append_trace_segment(
                self.trace_overlay_bgr,
                self.trace_overlay_mask,
                self.last_trace_pixel,
                current_pixel,
                color=(0, 0, 255),
                thickness=1,
            )
        self.last_trace_pixel = current_pixel

    def render(
        self,
        frame_bgr: np.ndarray,
        *,
        side_detector: Any,
        magnetic_model: Any,
        motion_active: bool,
        target_xyz=None,
        ref_path_3d=None,
    ) -> np.ndarray:
        if not bool(getattr(side_detector, "y_calibrated", False)):
            self._was_motion_active = bool(motion_active)
            return frame_bgr

        if bool(motion_active) and not self._was_motion_active:
            self.reset()
        self._was_motion_active = bool(motion_active)

        self._ensure_layers(frame_bgr.shape, side_detector, ref_path_3d)
        if motion_active:
            self._append_current_pair(magnetic_model, side_detector)

        out = trajectory_overlay.blend_masked_overlay(
            frame_bgr,
            self.static_overlay_bgr,
            self.static_overlay_mask,
            alpha=0.78,
        )
        out = trajectory_overlay.blend_masked_overlay(
            out,
            self.trace_overlay_bgr,
            self.trace_overlay_mask,
            alpha=1.0,
        )

        if motion_active and target_xyz is not None:
            target = np.asarray(target_xyz, dtype=float).reshape(3,)
            if np.all(np.isfinite(target)):
                u, v = side_detector.yz_cm_to_pixel(target[1], target[2])
                cv2.circle(
                    out,
                    (int(round(u)), int(round(v))),
                    6,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )
        return out
# [3D_PID] END: 侧视 yz 叠图实现。
