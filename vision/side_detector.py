"""
文件功能：
    侧视磁珠轻量检测：二值化、腐蚀、ROI、轮廓筛选与 z_px 计算。

设计原则：
    1. 只负责图像处理与 z_px 换算，不写入 magnetic_model 或控制循环；
    2. 输出五幅与原图同尺寸的 BGR 调试图，语义与上视四阶段小窗对齐；
    3. 第一版仅按面积筛选；尺度与落底零位由 set_calibration() 注入，不在此收集 15 帧。

检测流水线（与 UI 调试小窗对应关系）：
    Mono8 灰度 gray
      → GaussianBlur(3×3)               → 抑制孤立噪点，稳定阈值分割
      → THRESH_BINARY_INV               → 调试窗「侧视二值化」(binary_bgr)
      → 形态学腐蚀 eroded               → 去掉小噪点、断开细连接
      → 与 ROI 掩膜相与                 → 调试窗「侧视腐蚀 + ROI」(eroded_roi_bgr)
      → findContours 得 contours
      → 面积筛选后的全部候选可视化       → 调试窗「侧视候选轮廓」(candidates_bgr)
      → 取最大面积候选 + moments 圆心   → 调试窗「侧视筛选结果」(filtered_bgr)
      → z_px / z_mm（标定后）           → 主视角叠加显示 (main_bgr)

z_px / z_mm 定义（标定后）：
    zero_center_y_px — 15 帧落底标定得到的中位 center_y（由 DualViewCalibrator 写入）
    z_px = zero_center_y_px - center_y_px
    z_mm = z_px / calibrated_px_per_mm

    标定前 z_px、z_mm 均为 None；不使用 ROI 下底作为正式 z=0。

参数调试速查（建议每次只改一个量，对照 UI 五个侧视小窗观察）：
    ------------------------------------------------------------------
    roi_half_height / roi_half_width   默认 103 / 350（像素）
    roi_center_offset_x / roi_center_offset_y   默认 0 / 0（像素）
    ------------------------------------------------------------------
    以图像中心为原点，向上下/左右各延伸半高、半宽，得到 ROI 矩形 (y1,y2,x1,x2)。
    center_offset_y > 0 下移 ROI，< 0 上移；center_offset_x > 0 右移，< 0 左移。

    调试建议：
      - 主视角/候选窗里磁珠在黄色 ROI 外
          → 增大 roi_half_height 或 roi_half_width（对称扩展）
      - ROI 相对画面中心偏移（相机安装/光轴）
          → 调整 roi_center_offset_x / roi_center_offset_y（如 -80 表示 ROI 中心上移 80 px）
      - 腐蚀+ROI 窗外仍有白色块
          → 略减小半宽/半高，或提高 threshold
      - 改分辨率后面积筛选失效
          → 按面积比例调整 area_min / area_max（ROI 半尺寸可保持不变）

    ------------------------------------------------------------------
    threshold = 80
    ------------------------------------------------------------------
    对 GaussianBlur 后的灰度做固定阈值二值化。
    使用 THRESH_BINARY_INV：灰度 <= threshold 的像素变为 255（白），即暗磁珠为白。

    调试建议：
      - 「侧视二值化」窗全黑、看不到磁珠
          → 适当提高 threshold；或先固定相机曝光/增益
      - 二值化窗满屏白点、背景噪点多
          → 适当降低 threshold
      - 磁珠被碎成多块白斑
          → 略提高 threshold，或减少 erode_iterations

    ------------------------------------------------------------------
    kernel_size = 2,  erode_iterations = 0（主程序与上视读取同一组配置）
    ------------------------------------------------------------------
    结构元素为正方形 MORPH_RECT，边长 kernel_size。
    对二值图中的白色前景（暗磁珠）做腐蚀，每次腐蚀白色区域收缩一圈。

    调试建议：
      - 腐蚀+ROI 窗目标碎成很多小块
          → erode_iterations 减 1，或 kernel_size 从 3 改回 2
      - 候选窗噪点轮廓过多
          → 略增 erode_iterations（谨慎，易吃掉小目标）
      - 磁珠面积刚好卡在 area_min 下方
          → 减少腐蚀强度，或降低 area_min

    ------------------------------------------------------------------
    area_min = 10,  area_max = 1000
    ------------------------------------------------------------------
    候选筛选第一版只使用轮廓面积（像素²）：
        area_min <= contourArea <= area_max

    多个合格候选时，选择面积最大的一个作为最终磁珠（无评分、无平滑）。

    调试建议：
      - 候选窗有轮廓、但主视角显示 NO TARGET
          → 所有候选面积都 < area_min：降低 area_min 或减轻腐蚀
      - 选中了错误的大块背景
          → 降低 area_max，或缩小 roi 避开干扰区域
      - 两个相近候选交替跳变
          → 先通过 threshold/腐蚀减少假轮廓；第一版不做多帧平滑
      - 更换相机分辨率后面积极变
          → 按面积比例重新估计 area_min / area_max

    ------------------------------------------------------------------
    五幅调试图与 frame_bundle key 对应
    ------------------------------------------------------------------
    main_bgr         → "side_main"         侧视主视角（ROI + 中心 + Pair/Z 状态）
    binary_bgr       → "side_binary"       侧视二值化
    eroded_roi_bgr   → "side_eroded_roi"   侧视腐蚀 + ROI
    candidates_bgr   → "side_candidates"   侧视候选轮廓（编号:面积）
    filtered_bgr     → "side_filtered"     侧视筛选结果（选中轮廓 + 基准线 + z_px）

与其他模块关系：
    被 app/vision_control_loop.py 调用；调试图经 frame_bundle 送到 main.py 显示。
    不 import motion_modes、controllers、RuntimeLoop、magnetic_model。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# ============== 本次视觉修改【侧视接入】：公用函数导入开始 ==============
# 作用：与上视复用同一份预处理、原始轮廓关联代码。
from vision.bead_processing import get_calibration_contour, preprocess_bead
# ============== 本次视觉修改【侧视接入】：公用函数导入结束 ==============
from vision.dual_view_calibrator import ViewCalibrationObservation


# [3D_PID] BEGIN: 侧视图像 x 轴与模型 y 轴之间唯一的方向参数。
# 单个静止标定点只能确定零点和尺度，不能确定 image-right 是物理 +y 还是 -y；
# 硬件阶段沿模型 +y 移动一次磁珠后，仅修改该常量，不要在其它公式重复加符号。
SIDE_Y_SIGN = 1
# [3D_PID] END: 侧视 y 方向参数。


def roi_rect_from_image_center(
    shape: tuple,
    *,
    half_height: int,
    half_width: int,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> Tuple[int, int, int, int]:
    """
    以图像几何中心为基准，生成 ROI 矩形并裁剪到图像范围内。

    参数：
        shape: 灰度图 (height, width, ...)；
        half_height: 从中心向上下各延伸的像素半高；
        half_width: 从中心向左右各延伸的像素半宽；
        center_offset_x: ROI 中心相对图像中心的水平偏移（px，正=右移）；
        center_offset_y: ROI 中心相对图像中心的垂直偏移（px，正=下移）。

    返回：
        (y1, y2, x1, x2)，与 numpy 切片 gray[y1:y2, x1:x2] 一致。
    """
    height, width = shape[:2]
    cx = width / 2.0 + float(center_offset_x)
    cy = height / 2.0 + float(center_offset_y)
    y1 = int(round(cy - half_height))
    y2 = int(round(cy + half_height))
    x1 = int(round(cx - half_width))
    x2 = int(round(cx + half_width))
    return clamp_roi((y1, y2, x1, x2), shape)


def clamp_roi(roi: Sequence[int], shape: tuple) -> Tuple[int, int, int, int]:
    """
    将 ROI 限制在图像范围内，避免越界切片或 findContours 异常。

    参数：
        roi: (y1, y2, x1, x2)，行优先，与 numpy 切片 gray[y1:y2, x1:x2] 一致；
        shape: 灰度图 shape，至少包含 (height, width)。

    返回：
        裁剪后的 (y1, y2, x1, x2)，保证 y2 > y1 且 x2 > x1。
    """
    height, width = shape[:2]
    y1, y2, x1, x2 = map(int, roi)
    y1 = max(0, min(y1, height - 2))
    y2 = max(y1 + 1, min(y2, height))
    x1 = max(0, min(x1, width - 2))
    x2 = max(x1 + 1, min(x2, width))
    return y1, y2, x1, x2


@dataclass
class SideDetectionResult:
    """
    单帧侧视检测结果（数值输出 + 五路调试图）。

    控制相关：
        第一版 valid / center_* / z_px 仅用于显示与调试，不写入 magnetic_model。

    调试相关（送入 main.py frame_bundle）：
        main_bgr / binary_bgr / eroded_roi_bgr / candidates_bgr / filtered_bgr

    配对相关：
        pair_id / timestamp 由 VisionControlLoop 注入，便于与上视帧对齐排查。
    """

    # --- 检测数值 ---
    valid: bool
    center_x_px: Optional[float]
    center_y_px: Optional[float]
    z_px: Optional[float]
    z_mm: Optional[float]
    selected_area: Optional[float]
    diameter_px: Optional[float]
    circularity: Optional[float]
    candidate_count: int
    contour_areas_px2: tuple[float, ...]
    calibration_observation: Optional[ViewCalibrationObservation]
    scale_calibrated: bool
    zero_calibrated: bool
    px_per_mm: Optional[float]
    zero_center_y_px: Optional[float]
    pair_id: int
    timestamp: float

    # --- 五路 BGR 调试图（与原图同尺寸）---
    main_bgr: np.ndarray
    binary_bgr: np.ndarray
    eroded_roi_bgr: np.ndarray
    candidates_bgr: np.ndarray
    filtered_bgr: np.ndarray

    # [3D_PID] BEGIN: 追加侧视物理 y 诊断字段。
    # 放在 dataclass 末尾并提供默认值，可保持旧 SideDetectionResult 的位置参数
    # 构造兼容。y_cm 只用于叠图和 top/side 一致性诊断，不写入控制状态。
    y_cm: Optional[float] = None
    y_calibrated: bool = False
    side_px_per_cm: Optional[float] = None
    y_zero_center_x_px: Optional[float] = None
    y_sign: int = SIDE_Y_SIGN
    # [3D_PID] END: 侧视 y 检测结果字段。


class SideDetector:
    """
    侧视磁珠检测器（带尺度/零位标定状态）。

    构造参数可由 VisionControlLoop 从 cfg 覆盖。调参见模块顶部「参数调试速查」。

    ROI：以图像中心对称扩展 roi_half_height × roi_half_width，
    可用 roi_center_offset_x/y 修正安装/光轴偏移。
    标定后 z 约定：
        z_px = zero_center_y_px - center_y_px
        z_mm = z_px / calibrated_px_per_mm
    """

    def __init__(
        self,
        *,
        roi_half_height: int = 150,
        roi_half_width: int = 350,
        roi_center_offset_x: float = 0.0,
        roi_center_offset_y: float = 0.0,
        threshold: int = 70,
        area_min: int = 50,
        area_max: int = 150, # 这边只是初值，真正的设定在vision_control_loop中的105L
        kernel_size: int = 2,
        erode_iterations: int = 0,
    ):
        # ROI 半尺寸（相对图像中心，单位像素）
        self.roi_half_height = max(1, int(roi_half_height))
        self.roi_half_width = max(1, int(roi_half_width))
        self.roi_center_offset_x = float(roi_center_offset_x)
        self.roi_center_offset_y = float(roi_center_offset_y)
        # 反向二值化阈值：灰度 < threshold → 白（暗磁珠）
        self.threshold = int(threshold)
        # 轮廓面积合格区间（像素²），闭区间
        self.area_min = int(area_min)
        self.area_max = int(area_max)
        # 腐蚀结构元边长（像素）
        self.kernel_size = max(1, int(kernel_size))
        # 腐蚀重复次数；0 表示跳过腐蚀
        self.erode_iterations = max(0, int(erode_iterations))

        self.scale_calibrated: bool = False
        self.zero_calibrated: bool = False
        self.calibrated_px_per_mm: Optional[float] = None
        self.zero_center_y_px: Optional[float] = None
        # [3D_PID] BEGIN: y 标定状态与原有 z 标定状态分离保存。
        # scale_calibrated/zero_calibrated/zero_center_y_px 仍服务原 z/mm 计算；
        # y_calibrated 只有在提供 y_zero_center_x_px 后才成立。
        self.y_calibrated: bool = False
        self.side_px_per_cm: Optional[float] = None
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        # 侧视标尺补偿：Y 与 Z 使用独立比例尺。
        self.side_y_px_per_cm: Optional[float] = None
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================
        self.y_zero_center_x_px: Optional[float] = None
        self.y_sign: int = int(SIDE_Y_SIGN)
        # [3D_PID] END: 独立 y 标定状态。

    def set_calibration(
        self,
        *,
        px_per_mm: float,
        zero_center_y_px: float,
        # [3D_PID] 新增参数均有默认值，旧调用只传 z 标定参数仍可正常工作。
        y_zero_center_x_px: Optional[float] = None,
        y_sign: int = SIDE_Y_SIGN,
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        # 侧视标尺补偿：可选 Y 专用 px/mm；不传则与 Z 共用侧视比例尺。
        y_px_per_mm: Optional[float] = None,
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================
    ) -> None:
        px_value = float(px_per_mm)
        zero_value = float(zero_center_y_px)
        if not math.isfinite(px_value) or not math.isfinite(zero_value):
            raise ValueError("px_per_mm 与 zero_center_y_px 必须是有限数")
        if px_value <= 0.0:
            raise ValueError(f"px_per_mm 必须是正数，收到: {px_per_mm}")
        sign_value = int(y_sign)
        if sign_value not in (-1, 1):
            raise ValueError(f"y_sign 必须为 +1 或 -1，收到: {y_sign}")
        y_zero_value = None
        if y_zero_center_x_px is not None:
            y_zero_value = float(y_zero_center_x_px)
            if not math.isfinite(y_zero_value):
                raise ValueError("y_zero_center_x_px 必须是有限数或 None")
        # [3D_PID] BEGIN: 同时注入共用尺度、原 z 零线和新增 y 零线。
        # side_px_per_cm 由既有 px/mm 严格换算；没有 y 零线时只启用旧 z 标定。
        self.calibrated_px_per_mm = px_value
        self.side_px_per_cm = 10.0 * px_value
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        # 侧视标尺补偿：
        # side_px_per_cm 继续用于 Z；
        # side_y_px_per_cm 单独用于 Y。
        if y_px_per_mm is None:
            self.side_y_px_per_cm = self.side_px_per_cm
        else:
            self.side_y_px_per_cm = 10.0 * float(y_px_per_mm)
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================
        self.zero_center_y_px = zero_value
        self.y_zero_center_x_px = y_zero_value
        self.y_sign = sign_value
        self.scale_calibrated = True
        self.zero_calibrated = True
        self.y_calibrated = y_zero_value is not None
        # [3D_PID] END: 双轴标定状态注入。

    def reset_calibration(self) -> None:
        self.calibrated_px_per_mm = None
        self.zero_center_y_px = None
        # [3D_PID] 标定重置必须连同新增 y 尺度、零点、方向和有效状态清空；
        # 否则新一轮 z 标定前可能继续显示旧的侧视 y。
        self.side_px_per_cm = None
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        # 侧视标尺补偿：重置时一并清空 Y 专用比例尺。
        self.side_y_px_per_cm = None
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================
        self.y_zero_center_x_px = None
        self.y_sign = int(SIDE_Y_SIGN)
        self.scale_calibrated = False
        self.zero_calibrated = False
        self.y_calibrated = False

    # [3D_PID] BEGIN: 侧视 y 坐标正反转换公共接口。
    # 正变换：y = sign * (u-u0) / px_per_cm
    # 反变换：u = u0 + sign * px_per_cm * y
    # 检测、叠图和测试共用这组方法，避免三处公式发生符号漂移。
    def pixel_x_to_y_cm(self, center_x_px: float) -> float:
        """Convert a side-view image x pixel to side-view y (cm)."""
        if (
            not self.y_calibrated
            or self.y_zero_center_x_px is None
            # ============================================================================
            # [SIDE VIEW SCALE COMPENSATION] BEGIN
            # 侧视标尺补偿：Y 正反变换均使用 side_y_px_per_cm。
            or self.side_y_px_per_cm is None
            or self.side_y_px_per_cm <= 0.0
            # ============================================================================
            # [SIDE VIEW SCALE COMPENSATION] END
            # ============================================================================
        ):
            raise RuntimeError("side-view y calibration is not available")
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        return float(
            self.y_sign
            * (float(center_x_px) - self.y_zero_center_x_px)
            / self.side_y_px_per_cm
        )
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================

    def y_cm_to_pixel_x(self, y_cm: float) -> float:
        """Convert side-view y (cm) to a side-view image x pixel."""
        if (
            not self.y_calibrated
            or self.y_zero_center_x_px is None
            # ============================================================================
            # [SIDE VIEW SCALE COMPENSATION] BEGIN
            # 侧视标尺补偿：Y 正反变换均使用 side_y_px_per_cm。
            or self.side_y_px_per_cm is None
            or self.side_y_px_per_cm <= 0.0
            # ============================================================================
            # [SIDE VIEW SCALE COMPENSATION] END
            # ============================================================================
        ):
            raise RuntimeError("side-view y calibration is not available")
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] BEGIN
        return float(
            self.y_zero_center_x_px
            + self.y_sign * self.side_y_px_per_cm * float(y_cm)
        )
        # ============================================================================
        # [SIDE VIEW SCALE COMPENSATION] END
        # ============================================================================

    def z_cm_to_pixel_y(self, z_cm: float) -> float:
        """Convert physical z (cm) using the existing bottom z=0 calibration."""
        if (
            not self.is_fully_calibrated
            or self.zero_center_y_px is None
            or self.side_px_per_cm is None
        ):
            raise RuntimeError("side-view z calibration is not available")
        return float(self.zero_center_y_px - self.side_px_per_cm * float(z_cm))

    def yz_cm_to_pixel(self, y_cm: float, z_cm: float) -> Tuple[float, float]:
        return self.y_cm_to_pixel_x(y_cm), self.z_cm_to_pixel_y(z_cm)
    # [3D_PID] END: 侧视 y/z 投影公共接口。

    @property
    def is_fully_calibrated(self) -> bool:
        return (
            self.scale_calibrated
            and self.zero_calibrated
            and self.calibrated_px_per_mm is not None
            and self.zero_center_y_px is not None
        )

    @staticmethod
    def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
        """Mono8 灰度转 BGR，供叠加绘制与 UI 统一显示。"""
        if gray.ndim != 2:
            raise ValueError("侧视输入必须是二维灰度图")
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _draw_roi(bgr: np.ndarray, roi_rect: Tuple[int, int, int, int]) -> None:
        """在调试图上叠加黄色 ROI 矩形，便于确认磁珠是否在搜索区内。"""
        y1, y2, x1, x2 = roi_rect
        cv2.rectangle(bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)

    @staticmethod
    def _draw_origin_line(
        bgr: np.ndarray,
        origin_y: Optional[float],
        roi_rect: Tuple[int, int, int, int],
    ) -> None:
        """在 filtered 窗画出 z=0 基准线（仅标定完成后在 zero_center_y_px 处绘制）。"""
        if origin_y is None:
            return
        _, _, x1, x2 = roi_rect
        y = int(round(origin_y))
        cv2.line(bgr, (x1, y), (x2 - 1, y), (255, 128, 0), 1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            "Z=0",
            (x2 + 4, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 128, 0),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
        """单通道掩膜转 BGR，使 UI 显示函数与上视调试窗处理一致。"""
        if mask.ndim == 2:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask.copy()

    def _draw_main_overlay(
        self,
        bgr: np.ndarray,
        *,
        pair_id: int,
        valid: bool,
        z_mm: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ) -> None:
        """在侧视主视角叠加运行状态文字与中心标记。"""
        if valid and center is not None:
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.drawMarker(
                bgr,
                (cx, cy),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )

        status = "VALID" if valid else "NO TARGET"
        if self.is_fully_calibrated:
            lines = [
                f"Pair: {pair_id}",
                f"Side: {status}",
                f"Scale: {self.calibrated_px_per_mm:.2f} px/mm",
                "Z: N/A" if (not valid or z_mm is None) else f"Z: {z_mm:+.2f} mm",
            ]
        else:
            lines = [
                f"Pair: {pair_id}",
                f"Side: {status}",
                "CAL: WAITING",
                "Z: N/A",
            ]

        y0 = 28
        for idx, line in enumerate(lines):
            if valid and not line.startswith("Z:"):
                color = (0, 255, 0)
            elif valid and line.startswith("Z:") and z_mm is not None:
                color = (0, 255, 0)
            elif line.startswith("Side:") and not valid:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255) if "CAL:" in line or "Scale:" in line else (0, 255, 0)
            cv2.putText(
                bgr,
                line,
                (12, y0 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _contour_circularity(contour, area: float) -> float:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return 0.0
        return float(4.0 * math.pi * area / (perimeter * perimeter))

    ### 新增：椭圆投影直径，用于分别标定侧视 Y/Z；拟合失败或 ROI 截断时跳过标定。
    @staticmethod
    def _fit_directional_diameters(contour, roi_rect):
        """返回椭圆的水平/竖直投影直径；无完整有效轮廓时不参与标定。"""
        if len(contour) < 5:
            return None, None
        y1, y2, x1, x2 = roi_rect
        points = np.asarray(contour).reshape(-1, 2)
        if (np.min(points[:, 0]) <= x1 or np.max(points[:, 0]) >= x2 - 1
                or np.min(points[:, 1]) <= y1 or np.max(points[:, 1]) >= y2 - 1):
            return None, None
        try:
            (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
        except cv2.error:
            return None, None
        if not np.all(np.isfinite([cx, cy, a, b, angle])) or min(a, b) <= 0:
            return None, None
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        return math.hypot(a * c, b * s), math.hypot(a * s, b * c)
    ### 椭圆投影直径结束。

    def _draw_filtered_overlay(
        self,
        filtered_bgr: np.ndarray,
        *,
        center_x_px: float,
        center_y_px: float,
        diameter_px: float,
        z_px: Optional[float],
        z_mm: Optional[float],
    ) -> None:
        lines = [f"center=({center_x_px:.1f}, {center_y_px:.1f})"]
        if self.is_fully_calibrated:
            if z_px is not None:
                lines.append(f"z_px={z_px:+.2f}")
            if z_mm is not None:
                lines.append(f"z_mm={z_mm:+.2f}")
            lines.append(f"scale={self.calibrated_px_per_mm:.2f} px/mm")
        else:
            lines.append(f"diameter={diameter_px:.2f} px")
            lines.append("CALIBRATING")

        for idx, line in enumerate(lines):
            cv2.putText(
                filtered_bgr,
                line,
                (12, 24 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _make_result(
        self,
        *,
        valid: bool,
        center_x_px: Optional[float],
        center_y_px: Optional[float],
        z_px: Optional[float],
        z_mm: Optional[float],
        selected_area: Optional[float],
        diameter_px: Optional[float],
        circularity: Optional[float],
        candidate_count: int,
        contour_areas_px2: tuple[float, ...],
        calibration_observation: Optional[ViewCalibrationObservation],
        pair_id: int,
        timestamp: float,
        main_bgr: np.ndarray,
        binary_bgr: np.ndarray,
        eroded_roi_bgr: np.ndarray,
        candidates_bgr: np.ndarray,
        filtered_bgr: np.ndarray,
    ) -> SideDetectionResult:
        # [3D_PID] 只有“本帧检测有效 + y 已独立标定”时才计算 y_cm。
        # 无效帧返回 None，不沿用历史侧视 y；该值不会覆盖上视控制 y。
        y_cm = None
        if valid and center_x_px is not None and self.y_calibrated:
            y_cm = self.pixel_x_to_y_cm(center_x_px)
        return SideDetectionResult(
            valid=valid,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            z_px=z_px,
            z_mm=z_mm,
            y_cm=y_cm,
            selected_area=selected_area,
            diameter_px=diameter_px,
            circularity=circularity,
            candidate_count=candidate_count,
            contour_areas_px2=contour_areas_px2,
            calibration_observation=calibration_observation,
            scale_calibrated=self.scale_calibrated,
            zero_calibrated=self.zero_calibrated,
            px_per_mm=self.calibrated_px_per_mm,
            zero_center_y_px=self.zero_center_y_px,
            y_calibrated=self.y_calibrated,
            side_px_per_cm=self.side_px_per_cm,
            y_zero_center_x_px=self.y_zero_center_x_px,
            y_sign=self.y_sign,
            pair_id=pair_id,
            timestamp=timestamp,
            main_bgr=main_bgr,
            binary_bgr=binary_bgr,
            eroded_roi_bgr=eroded_roi_bgr,
            candidates_bgr=candidates_bgr,
            filtered_bgr=filtered_bgr,
        )

    def process(
        self,
        gray: np.ndarray,
        *,
        pair_id: int,
        timestamp: float,
    ) -> SideDetectionResult:
        """
        单帧侧视检测主入口。

        参数：
            gray: uint8 二维 Mono8 灰度图，尺寸须与相机输出一致；
            pair_id: 与本轮上视帧共用的配对序号；
            timestamp: 侧视帧采集时刻（秒），用于 pair_dt 统计。

        返回：
            SideDetectionResult；无目标时 valid=False，z_px=None（不保留上一帧旧值）。
        """
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("侧视输入必须是 uint8 二维灰度图")

        # ==================================================================
        # 步骤 0：以图像中心生成 ROI；z=0 基准仅来自标定 zero_center_y_px
        # ==================================================================
        roi_rect = roi_rect_from_image_center(
            gray.shape,
            half_height=self.roi_half_height,
            half_width=self.roi_half_width,
            center_offset_x=self.roi_center_offset_x,
            center_offset_y=self.roi_center_offset_y,
        )
        y1, y2, x1, x2 = roi_rect
        origin_y = self.zero_center_y_px if self.zero_calibrated else None

        # ============== 本次视觉修改【侧视接入】：公用预处理开始 ==============
        # 作用：与上视执行相同的平滑、反向二值化、可选腐蚀，并衔接原有小窗。
        binary, eroded = preprocess_bead(
            gray, self.threshold, self.kernel_size, self.erode_iterations
        )
        # ============== 本次视觉修改【侧视接入】：公用预处理结束 ==============

        # ==================================================================
        # 步骤 3：ROI 掩膜裁剪
        # 对应调试窗 side_eroded_roi（ROI 外全部清零）
        # ==================================================================
        roi_mask = np.zeros_like(eroded, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        eroded_roi = cv2.bitwise_and(eroded, roi_mask)

        # ==================================================================
        # 步骤 4：轮廓提取
        # ==================================================================
        contours, _hierarchy = cv2.findContours(
            eroded_roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_areas_px2 = tuple(float(cv2.contourArea(c)) for c in contours)

        # ==================================================================
        # 步骤 5：准备五路 BGR 底图（全图尺寸，不 crop）
        # ==================================================================
        binary_bgr = self._mask_to_bgr(binary)
        eroded_roi_bgr = self._mask_to_bgr(eroded_roi)
        candidates_bgr = self._gray_to_bgr(gray)
        filtered_bgr = self._gray_to_bgr(gray)
        main_bgr = self._gray_to_bgr(gray)

        self._draw_roi(candidates_bgr, roi_rect)
        self._draw_roi(filtered_bgr, roi_rect)
        self._draw_roi(main_bgr, roi_rect)
        self._draw_origin_line(filtered_bgr, origin_y, roi_rect)

        # ==================================================================
        # 步骤 6：面积筛选 + 候选可视化
        # 对应调试窗 side_candidates（编号:面积）
        # ==================================================================
        area_candidates = []
        for idx, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if self.area_min <= area <= self.area_max:
                area_candidates.append((area, idx, contour))
                cv2.drawContours(candidates_bgr, [contour], -1, (255, 255, 0), 2)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                else:
                    # m00==0 的退化轮廓：用外接矩形中心仅用于候选窗标注
                    x_f, y_f, w_f, h_f = cv2.boundingRect(contour)
                    cx = x_f + w_f / 2.0
                    cy = y_f + h_f / 2.0
                cv2.putText(
                    candidates_bgr,
                    f"{idx}: A={area:.0f}",
                    (int(cx) + 8, max(24, int(cy) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        # ==================================================================
        # 步骤 7a：无合格候选 → 无效帧
        # ==================================================================
        if not area_candidates:
            self._draw_main_overlay(
                main_bgr,
                pair_id=pair_id,
                valid=False,
            )
            return self._make_result(
                valid=False,
                center_x_px=None,
                center_y_px=None,
                z_px=None,
                z_mm=None,
                selected_area=None,
                diameter_px=None,
                circularity=None,
                candidate_count=0,
                contour_areas_px2=contour_areas_px2,
                calibration_observation=None,
                pair_id=pair_id,
                timestamp=timestamp,
                main_bgr=main_bgr,
                binary_bgr=binary_bgr,
                eroded_roi_bgr=eroded_roi_bgr,
                candidates_bgr=candidates_bgr,
                filtered_bgr=filtered_bgr,
            )

        candidate_count = len(area_candidates)

        # ==================================================================
        # 步骤 7b：从合格候选中取面积最大者（第一版无评分）
        # ==================================================================
        selected_area, _selected_idx, selected_contour = max(
            area_candidates, key=lambda item: item[0]
        )
        moments = cv2.moments(selected_contour)
        if moments["m00"] == 0:
            self._draw_main_overlay(
                main_bgr,
                pair_id=pair_id,
                valid=False,
            )
            return self._make_result(
                valid=False,
                center_x_px=None,
                center_y_px=None,
                z_px=None,
                z_mm=None,
                selected_area=None,
                diameter_px=None,
                circularity=None,
                candidate_count=candidate_count,
                contour_areas_px2=contour_areas_px2,
                calibration_observation=None,
                pair_id=pair_id,
                timestamp=timestamp,
                main_bgr=main_bgr,
                binary_bgr=binary_bgr,
                eroded_roi_bgr=eroded_roi_bgr,
                candidates_bgr=candidates_bgr,
                filtered_bgr=filtered_bgr,
            )

        # ==================================================================
        # 步骤 8：moments 求中心；minEnclosingCircle 求直径/圆度
        # ==================================================================
        center_x_px = float(moments["m10"] / moments["m00"])
        center_y_px = float(moments["m01"] / moments["m00"])
        (_cx_circle, _cy_circle), radius = cv2.minEnclosingCircle(selected_contour)
        diameter_px = float(2.0 * radius)
        circularity = self._contour_circularity(selected_contour, float(selected_area))

        if self.zero_calibrated and self.zero_center_y_px is not None:
            z_px = float(self.zero_center_y_px - center_y_px)
        else:
            z_px = None

        if (
            z_px is not None
            and self.scale_calibrated
            and self.calibrated_px_per_mm is not None
            and self.calibrated_px_per_mm > 0.0
        ):
            z_mm = float(z_px / self.calibrated_px_per_mm)
        else:
            z_mm = None

        # ============== 本次视觉修改【侧视接入】：腐蚀前测径开始 ==============
        # 作用：用公用函数关联原始轮廓，计算标定直径、圆度和 Y/Z 方向直径。
        # 检测中心、候选数及外接圆诊断仍来自检测结果；关联失败仅跳过标定。
        calibration_observation = None
        bead_contour = get_calibration_contour(
            binary, selected_contour, roi_mask, self.erode_iterations
        )
        if bead_contour is not None:
            diameter_u_px, diameter_v_px = self._fit_directional_diameters(bead_contour, roi_rect)
            if diameter_u_px is not None and diameter_v_px is not None:
                _, calibration_radius = cv2.minEnclosingCircle(bead_contour)
                calibration_observation = ViewCalibrationObservation(
                    center_x_px=center_x_px,
                    center_y_px=center_y_px,
                    diameter_px=2.0 * float(calibration_radius),
                    circularity=self._contour_circularity(bead_contour, cv2.contourArea(bead_contour)),
                    candidate_count=candidate_count,
                    diameter_u_px=diameter_u_px,
                    diameter_v_px=diameter_v_px,
                )
        # ============== 本次视觉修改【侧视接入】：腐蚀前测径结束 ==============

        x_i, y_i, w_i, h_i = cv2.boundingRect(selected_contour)
        cv2.drawContours(filtered_bgr, [selected_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(filtered_bgr, (x_i, y_i), (x_i + w_i - 1, y_i + h_i - 1), (255, 0, 0), 2)
        cx_i, cy_i = int(round(center_x_px)), int(round(center_y_px))
        cv2.drawMarker(
            filtered_bgr,
            (cx_i, cy_i),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        self._draw_filtered_overlay(
            filtered_bgr,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            diameter_px=diameter_px,
            z_px=z_px,
            z_mm=z_mm,
        )

        self._draw_main_overlay(
            main_bgr,
            pair_id=pair_id,
            valid=True,
            z_mm=z_mm,
            center=(center_x_px, center_y_px),
        )

        return self._make_result(
            valid=True,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            z_px=z_px,
            z_mm=z_mm,
            selected_area=selected_area,
            diameter_px=diameter_px,
            circularity=circularity,
            candidate_count=candidate_count,
            contour_areas_px2=contour_areas_px2,
            calibration_observation=calibration_observation,
            pair_id=pair_id,
            timestamp=timestamp,
            main_bgr=main_bgr,
            binary_bgr=binary_bgr,
            eroded_roi_bgr=eroded_roi_bgr,
            candidates_bgr=candidates_bgr,
            filtered_bgr=filtered_bgr,
        )
