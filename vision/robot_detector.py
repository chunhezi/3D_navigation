"""
文件功能：
    磁控微机器人视觉检测：灰度二值化、形态学腐蚀、ROI 约束、轮廓筛选与坐标换算。

设计原则：
    1. 只负责图像处理与位置换算，不包含 overlay 绘制、控制律或硬件下发；
    2. 检测参数通过 DetectorConfig 注入（main.py 创建 RobotDetector 时传入）；
    3. [3D] 只输出原始上视 xy 测量（model_position_xy_cm）；
       不直接写 magnetic_model。三维测量与差分速度更新
       由 VisionControlLoop 在 top/side 汇合后统一完成。

检测流水线（与 UI 调试小窗对应关系）：
    BGR 原图
      → 灰度 gray
      → 固定阈值二值化 binary          → 调试窗「二值化」(binary_mask)
      → 形态学腐蚀 eroded              → 内部字段 eroded_mask
      → 与 ROI 掩膜相与                → 调试窗「腐蚀+ROI」(eroded_roi_mask)
      → findContours 得 contours
      → 全部轮廓可视化                 → 调试窗「候选轮廓」(candidate_overlay_bgr)
      → 按面积 min_area~max_area 筛选
      → 合格/不合格分色                → 调试窗「筛选结果」(filter_result_overlay_bgr)
      → 合格轮廓外接圆中心
      → [3D] 唯一合格轮廓时写入 model_position_xy_cm（原始 xy 测量）

参数调试速查（配合 UI 四个小窗观察）：
    - 二值化窗全黑/全白     → 调 threshold（反向二值化，或先固定相机曝光）
    - 腐蚀+ROI 窗目标碎成多块 → 略提高 threshold，或减少 erode_iterations
    - 候选窗有轮廓、筛选窗全红 small → 降低 min_area 或减轻腐蚀
    - 候选窗无轮廓、二值化窗有白斑   → 检查 ROI 是否盖住磁珠（图像中心 ROI 是否偏）
    - 筛选窗 large               → 提高 max_area，或检查是否与大块背景连通

与其他模块关系：
    被 app/vision_control_loop.py 调用；调试图经 frame_bundle 送到 main.py 显示。
    不 import motion_modes、controllers、devices。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ============== 本次视觉修改【上视接入】：公用函数导入开始 ==============
# 作用：复用与侧视相同的预处理、腐蚀前轮廓关联，避免分别维护两套实现。
from vision.bead_processing import get_calibration_contour, preprocess_bead
# ============== 本次视觉修改【上视接入】：公用函数导入结束 ==============
from vision.dual_view_calibrator import ViewCalibrationObservation


@dataclass
class DetectorConfig:
    """
    机器人检测参数。

    在 main.py 中创建 RobotDetector 时注入；与模块级 min_area/max_area/roi_l/roi_w 对齐。
    调参时建议每次只改一个量，并对照 UI 四个调试小窗观察变化。
    """

    # ------------------------------------------------------------------
    # 二值化（对应调试窗「二值化」）
    # ------------------------------------------------------------------
    # 灰度 <= threshold 的像素变为 255（白），否则为 0（黑）。
    # 与侧视共用 THRESH_BINARY_INV：暗磁珠为白色前景。
    #
    # 调试建议：
    #   - 二值化窗看不到完整磁珠 → 适当提高 threshold
    #   - 背景噪点过多、与背景粘连 → 适当降低 threshold
    #   - 调焦距后检不到，优先看此窗是否从「整块」变成「碎块」
    threshold: int = 90

    # ------------------------------------------------------------------
    # 形态学腐蚀（影响 eroded_mask / eroded_roi_mask）
    # ------------------------------------------------------------------
    # 结构元素为正方形 MORPH_RECT，边长 erode_kernel_size。
    # erode_iterations 为重复腐蚀次数；每次腐蚀白色区域收缩一圈。
    #
    # 作用：去掉孤立噪点、断开细连接。
    # 副作用：白色磁珠变小，可能导致面积 < min_area；标定测径使用腐蚀前图。
    #
    # 调试建议：
    #   - 碎块过多 → iterations 减 1，或 kernel 从 5 改为 3
    #   - 背景噪点仍多 → 略增 iterations（谨慎，易吃掉小目标）
    # ============== 本次视觉修改【上视参数】：腐蚀默认值开始 ==============
    # 作用：默认不腐蚀；主程序用 main.py 的两路公用配置覆盖这里的构造默认值。
    erode_kernel_size: int = 2
    erode_iterations: int = 0
    # ============== 本次视觉修改【上视参数】：腐蚀默认值结束 ==============

    # ------------------------------------------------------------------
    # 轮廓面积筛选（对应调试窗「筛选结果」绿/红判定）
    # ------------------------------------------------------------------
    # 仅保留 min_area < 轮廓面积 < max_area 的连通域作为机器人。
    # 单位：像素²（与 1920×1080 分辨率相关，改分辨率后需重新估面积）。
    #
    # main.py 传入 min_area=100, max_area=300，作为未腐蚀轮廓的调试初值。
    #
    # 调试建议：
    #   - 筛选窗全红 small → 降低 min_area
    #   - 筛选窗全红 large → 提高 max_area，或检查是否与大轮廓粘连
    #   - 多个候选时位置跳变   → 先减少噪点，再考虑收紧面积范围
    # ============== 本次视觉修改【上视参数】：面积默认值开始 ==============
    # 作用：改用未腐蚀白色磁珠的面积初值，不再使用原来被放大的黑孔面积。
    min_area: int = 100
    max_area: int = 300
    # ============== 本次视觉修改【上视参数】：面积默认值结束 ==============

    # ------------------------------------------------------------------
    # ROI 矩形（以图像几何中心为几何中心）
    # ------------------------------------------------------------------
    # 首帧用图像中心定物理坐标原点后，建立 roi_mask：矩形内 255、外 0。
    # 半长 roi_half_length（像素）、半宽 roi_half_width（像素）。
    # main.py 默认 roi_l=500, roi_w=155。
    #
    # 调试建议：
    #   - 磁珠在腐蚀+ROI 窗外 → 扩大 half_length/half_width，或 reset() 后重新标定
    #   - 窗外干扰仍被检出   → 缩小 ROI
    roi_half_length: int = 500
    roi_half_width: int = 155

    # ------------------------------------------------------------------
    # 像素 → 厘米标定（影响控制精度，不影响轮廓有无）
    # ------------------------------------------------------------------
    # px_per_cm = px_width_ref / cm_width_ref
    # 表示画面中 px_width_ref 像素宽度对应 cm_width_ref 厘米物理距离。
    px_width_ref: float = 380.0
    cm_width_ref: float = 3.0

    @property
    def px_per_cm(self) -> float:
        """像素/厘米 换算比例。"""
        return float(self.px_width_ref) / float(self.cm_width_ref)


@dataclass
class RobotDetectionResult:
    """
    单帧检测结果（检测输出 + 调试可视化素材）。

    控制相关：
        robot_centers_px / annotated_bgr — 合格轮廓的中心与标注图
        [3D] model_position_xy_cm — 唯一合格轮廓时的原始平面位置（cm），
             供 VisionControlLoop 更新三维测量状态；本检测器不写模型
    调试相关（送入 main.py frame_bundle）：
        binary_mask / eroded_roi_mask / candidate_overlay_bgr / filter_result_overlay_bgr
    标定相关：
        circle_position / roi_rect / scene_just_initialized
        calibration_observation — 供 DualViewCalibrator 使用的单帧磁珠几何摘要
    """

    robot_centers_px: List[Tuple[int, int]] = field(default_factory=list)
    annotated_bgr: Optional[np.ndarray] = None
    px_per_cm: float = 380.0 / 3.0
    circle_position: Optional[List[int]] = None
    roi_rect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    scene_just_initialized: bool = False
    has_contours: bool = False
    calibration_observation: Optional[ViewCalibrationObservation] = None
    calibration_rejection_reason: Optional[str] = None
    filtered_contour_count: int = 0
    scale_calibrated: bool = False
    active_px_per_mm: float = 380.0 / 30.0
    # [3D] 原始上视 xy 测量（cm），shape=(2,1)；仅 filtered_contour_count==1 时非 None
    # 作用：把像素中心换算成模型平面坐标，交给 VisionControlLoop，
    #       检测器本身不再调用 magnetic_model.update_status
    model_position_xy_cm: Optional[np.ndarray] = None
    # --- 调试图像（单通道 uint8 或 BGR）---
    binary_mask: Optional[np.ndarray] = None
    eroded_mask: Optional[np.ndarray] = None
    eroded_roi_mask: Optional[np.ndarray] = None
    candidate_overlay_bgr: Optional[np.ndarray] = None
    filter_result_overlay_bgr: Optional[np.ndarray] = None


class RobotDetector:
    """
    磁控微机器人检测器（有状态）。

    [3D] 职责边界：
        只做上视图像识别与 xy 像素→cm 换算，输出 RobotDetectionResult；
        三维 Kalman 与 magnetic_model 状态更新由 VisionControlLoop 统一完成。

    跨帧保持的状态：
        calibrated / circle_position — 首帧场景标定（图像中心为坐标原点）
        roi_mask / roi_rect          — 限制后续 findContours 的搜索区域
        scale_calibrated / calibrated_px_per_mm — 双视尺度标定（与场景标定独立）

    调用 reset() 可清空标定，下次 process_frame 会重新以图像中心定原点。
    """

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg if cfg is not None else DetectorConfig()
        self.calibrated: bool = False
        self.circle_position: Optional[List[int]] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.roi_rect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.scale_calibrated: bool = False
        self.calibrated_px_per_mm: Optional[float] = None

    @property
    def active_px_per_mm(self) -> float:
        if self.scale_calibrated and self.calibrated_px_per_mm is not None:
            value = float(self.calibrated_px_per_mm)
            if math.isfinite(value) and value > 0.0:
                return value
        return self.cfg.px_per_cm / 10.0

    @property
    def active_px_per_cm(self) -> float:
        return 10.0 * self.active_px_per_mm

    def set_scale_calibration(self, px_per_mm: float) -> None:
        value = float(px_per_mm)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"px_per_mm 必须是有限正数，收到: {px_per_mm}")
        self.calibrated_px_per_mm = value
        self.scale_calibrated = True

    def reset_scale_calibration(self) -> None:
        self.calibrated_px_per_mm = None
        self.scale_calibrated = False

    def reset(self) -> None:
        """清除场景标定与 ROI（换相机、换工作区、标定错误时调用）。"""
        self.calibrated = False
        self.circle_position = None
        self.roi_mask = None
        self.roi_rect = None
        self.reset_scale_calibration()

    # [3D_PID] BEGIN: 公开复用既有上视像素 -> 模型 xy 坐标变换。
    # 双视 y 标定必须与日常上视检测使用完全相同的轴交换、符号和 circle_position
    # 原点，否则反算出的侧视 y=0 会与控制坐标不一致。公开方法只暴露原算法；
    # 下方私有别名继续保留，避免改变旧 detect() 路径及潜在外部调用。
    def pixel_to_model_position(
        self,
        x: float,
        y: float,
        circle_position: Sequence[int],
        px_per_cm: float,
    ) -> np.ndarray:
        """
        像素坐标 → 磁模型平面坐标（cm）。

        约定（与 overlay 绘制互逆）：
            图像列 x 增大 → 模型 y 增大
            图像行 y 增大 → 模型 x 增大
        原点为首帧标定 circle_position（上视图像几何中心）。

        [3D] 返回 shape (2, 1)，作为原始 xy 测量写入 result.model_position_xy_cm；
        不再直接供 magnetic_model.update_status 使用。
        """
        cx, cy = float(circle_position[0]), float(circle_position[1])
        return np.array(
            [[(y - cy) / px_per_cm], [(x - cx) / px_per_cm]],
            dtype=float,
        )

    def _pixel_to_model_position(
        self,
        x: float,
        y: float,
        circle_position: Sequence[int],
        px_per_cm: float,
    ) -> np.ndarray:
        """Backward-compatible alias for :meth:`pixel_to_model_position`."""
        return self.pixel_to_model_position(x, y, circle_position, px_per_cm)
    # [3D_PID] END: 上视坐标变换公共接口及旧接口兼容层。

    def _draw_roi_rect(self, image: np.ndarray) -> None:
        """在调试图上叠加 ROI 矩形（黄色），便于确认磁珠是否在搜索区内。"""
        if self.roi_rect is None:
            return
        pt1, pt2 = self.roi_rect
        cv2.rectangle(image, pt1, pt2, (0, 255, 255), 2)

    def _build_debug_overlays(
        self,
        bgr_frame: np.ndarray,
        contours: Sequence,
        cfg: DetectorConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        由 findContours 的原始轮廓列表生成两路调试图。

        candidate_overlay_bgr（UI 左下「候选轮廓」）：
            - 绘制本帧全部轮廓（未做面积筛选）
            - 青色外轮廓 + 黄点圆心 + 文字「索引:面积」
            - 用于判断：提取阶段是否「看到」目标

        filter_result_overlay_bgr（UI 右下「筛选结果」）：
            - 同一批轮廓按 min_area / max_area 判定
            - 绿色 + OK  ：面积在 (min_area, max_area) 内，会进入 filtered_contours
            - 红色 + small：面积 <= min_area（常见：碎块、噪点）
            - 红色 + large：面积 >= max_area（常见：容器边、大块反光）
        """
        candidate_overlay = bgr_frame.copy()
        filter_overlay = bgr_frame.copy()
        self._draw_roi_rect(candidate_overlay)
        self._draw_roi_rect(filter_overlay)

        for idx, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            (x_f, y_f), _radius = cv2.minEnclosingCircle(contour)
            x, y = int(x_f), int(y_f)

            # --- 候选图：只展示，不判对错 ---
            # 字号在 1920×1080 原图上绘制，UI 小窗缩放后仍可读（fontScale/thickness 可在此调）
            _cand_font_scale = 5.0
            _cand_font_thickness = 2
            cv2.drawContours(candidate_overlay, [contour], -1, (255, 255, 0), 2)
            cv2.circle(candidate_overlay, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(
                candidate_overlay,
                f"{idx}:{area:.0f}",
                (x + 10, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                _cand_font_scale,
                (0, 255, 255),
                _cand_font_thickness,
                cv2.LINE_AA,
            )

            # --- 筛选图：与下方 filtered_contours 使用相同判据 ---
            if area <= float(cfg.min_area):
                passed = False
                reason = "small"
            elif area >= float(cfg.max_area):
                passed = False
                reason = "large"
            else:
                passed = True
                reason = "OK"

            color = (0, 255, 0) if passed else (0, 0, 255)
            cv2.drawContours(filter_overlay, [contour], -1, color, 2)
            cv2.circle(filter_overlay, (x, y), 4, color, -1)
            cv2.putText(
                filter_overlay,
                f"{idx}:{reason}",
                (x + 6, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return candidate_overlay, filter_overlay

    def process_frame(
        self,
        bgr_frame: np.ndarray,
    ) -> RobotDetectionResult:
        """
        单帧检测主入口。

        参数：
            bgr_frame: BGR 图像（相机已 flip；分辨率通常 1920×1080）

        [3D] 输出约定：
            仅当 filtered_contour_count == 1 时，
            result.model_position_xy_cm 为唯一候选中心的 (2,1) cm 坐标；
            0 个或多个候选时为 None（避免多目标互相覆盖写模型的旧问题）。
        """
        cfg = self.cfg
        px_per_cm = self.active_px_per_cm
        annotated = bgr_frame.copy()
        result = RobotDetectionResult(
            annotated_bgr=annotated,
            px_per_cm=px_per_cm,
            circle_position=self.circle_position,
            roi_rect=self.roi_rect,
            scale_calibrated=self.scale_calibrated,
            active_px_per_mm=self.active_px_per_mm,
        )

        # ============== 本次视觉修改【上视接入】：公用预处理开始 ==============
        # 作用：与侧视统一暗珠为白色前景，返回图直接接入原有二值化、腐蚀小窗。
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        binary, eroded = preprocess_bead(
            gray, cfg.threshold, cfg.erode_kernel_size, cfg.erode_iterations
        )

        # 保存腐蚀后、ROI 前的结果（便于对比 ROI 裁剪掉了什么）
        result.binary_mask = binary.copy()
        result.eroded_mask = eroded.copy()
        # ============== 本次视觉修改【上视接入】：公用预处理结束 ==============

        # ==================================================================
        # 步骤 3：首帧场景标定（仅执行一次）
        # ==================================================================
        # 物理坐标原点 = 上视图像几何中心；不再使用最大轮廓外接圆。
        # 本帧同时建立 roi_mask，后续 findContours 只在 ROI 内搜索。
        if not self.calibrated:
            height, width = gray.shape[:2]
            center_x = int(width // 2)
            center_y = int(height // 2)

            self.circle_position = [center_x, center_y]
            self.calibrated = True

            self.roi_mask = np.zeros_like(gray, dtype=np.uint8)
            pt1 = (
                int(center_x - cfg.roi_half_length),
                int(center_y - cfg.roi_half_width),
            )
            pt2 = (
                int(center_x + cfg.roi_half_length),
                int(center_y + cfg.roi_half_width),
            )
            cv2.rectangle(self.roi_mask, pt1, pt2, 255, -1)
            self.roi_rect = (pt1, pt2)

            result.scene_just_initialized = True

        result.circle_position = self.circle_position
        result.roi_rect = self.roi_rect

        # ==================================================================
        # 步骤 4：ROI 裁剪
        # ==================================================================
        # 标定后只保留 ROI 矩形内的白色区域，抑制窗外干扰。
        eroded_for_contours = eroded
        if self.roi_mask is not None:
            eroded_for_contours = cv2.bitwise_and(eroded, self.roi_mask)
        result.eroded_roi_mask = eroded_for_contours.copy()

        # ==================================================================
        # 步骤 5：轮廓提取
        # ==================================================================
        # ============== 本次视觉修改【上视接入】：外轮廓提取开始 ==============
        # 作用：暗珠已是白色前景，与侧视统一提取外轮廓，避免继续测白背景中的黑孔。
        # CHAIN_APPROX_SIMPLE：压缩水平/垂直/对角线段，节省点数
        contours, _hierarchy = cv2.findContours(
            eroded_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # ============== 本次视觉修改【上视接入】：外轮廓提取结束 ==============

        # 有轮廓则生成调试图（即使后续面积筛选为空，也能在 UI 看到原因）
        if contours:
            result.candidate_overlay_bgr, result.filter_result_overlay_bgr = (
                self._build_debug_overlays(bgr_frame, contours, cfg)
            )

        if not contours:
            # 无轮廓：位置不更新；对照二值化/腐蚀+ROI 窗排查
            return result

        result.has_contours = True

        if self.circle_position is None:
            return result

        # ==================================================================
        # 步骤 6：面积筛选 → 机器人候选
        # ==================================================================
        # 判据与 filter_result_overlay 中 OK/small/large 完全一致
        filtered_contours = [
            c for c in contours
            if cfg.min_area < cv2.contourArea(c) < cfg.max_area
        ]
        result.filtered_contour_count = len(filtered_contours)

        if not result.scene_just_initialized and len(filtered_contours) == 1:
            # ============== 本次视觉修改【上视接入】：腐蚀前测径开始 ==============
            # 作用：复用公用轮廓关联函数，测原始直径与圆度，检测中心仍沿用原结果。
            (x, y), _ = cv2.minEnclosingCircle(filtered_contours[0])
            bead_contour = get_calibration_contour(
                binary, filtered_contours[0], self.roi_mask, cfg.erode_iterations
            )
            if bead_contour is not None:
                _, radius = cv2.minEnclosingCircle(bead_contour)
                area = float(cv2.contourArea(bead_contour))
                perimeter = float(cv2.arcLength(bead_contour, True))
                circularity = 0.0 if perimeter <= 0.0 else float(
                    4.0 * math.pi * area / (perimeter * perimeter)
                )
                result.calibration_observation = ViewCalibrationObservation(
                    center_x_px=float(x),
                    center_y_px=float(y),
                    diameter_px=2.0 * float(radius),
                    circularity=circularity,
                    candidate_count=1,
                )
            else:
                result.calibration_rejection_reason = "contour_touches_roi_boundary_or_invalid"
        elif not result.scene_just_initialized:
            result.calibration_rejection_reason = (
                f"filtered_contour_count={len(filtered_contours)}"
            )
            # ============== 本次视觉修改【上视接入】：腐蚀前测径结束 ==============

        # ==================================================================
        # 步骤 7：合格轮廓 → 像素中心；唯一候选时输出原始 xy 测量
        # ==================================================================
        for contour in filtered_contours:
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)
            (x_f, y_f), _radius = cv2.minEnclosingCircle(contour)
            x, y = int(x_f), int(y_f)

            result.robot_centers_px.append((x, y))

        # [3D] 仅唯一合格轮廓时给出原始 xy 测量；不写 magnetic_model
        # 作用：把“检测结果”与“状态估计”解耦，避免多轮廓互相覆盖、
        #       也避免上视单独改模型而侧视 z 无法成对更新测量状态
        if result.filtered_contour_count == 1 and self.circle_position is not None:
            x_u, y_u = result.robot_centers_px[0]
            result.model_position_xy_cm = self._pixel_to_model_position(
                x_u, y_u, self.circle_position, px_per_cm
            )
        else:
            result.model_position_xy_cm = None

        result.annotated_bgr = annotated
        result.circle_position = self.circle_position
        result.roi_rect = self.roi_rect
        return result
