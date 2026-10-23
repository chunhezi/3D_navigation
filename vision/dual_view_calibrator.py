"""
文件功能：
    双视标定样本收集与稳健统计。

设计原则：
    1. 只负责配对观测的接收、内点筛选与比例尺/零点计算；
    2. 不负责相机取流、图像检测、UI 绘制或控制下发；
    3. 不 import RobotDetector、SideDetector、RuntimeLoop、devices、motion_modes；
    4. 不保存文件；标定结果由调用方（VisionControlLoop）决定如何持久化或显示。

标定前提（实验操作，由操作者保证）：
    - 磁珠静止放在底面；
    - 上视、侧视各能稳定检测到唯一磁珠候选；
    - 连续收集 required_pair_count（默认 15）对有效配对帧。

数据流（与 VisionControlLoop 的衔接）：
    RobotDetector / SideDetector 输出几何量
        → 调用方构造 ViewCalibrationObservation（上视一条、侧视一条）
        → DualViewCalibrator.add_pair(pair_id, pair_dt_ms, top, side)
        → progress 递增；满 15 对后 try_finalize()
        → DualViewCalibrationResult（比例尺 + side_zero_y_px）
        → 注入 SideDetector / RobotDetector / overlay（由后续轮次实现）

标定输出公式：
    top_px_per_mm  = median(top_diameter_px_inliers) / bead_diameter_mm
    top_px_per_cm  = 10.0 * top_px_per_mm
    side_y_px_per_mm = median(side_diameter_u_px_inliers) / bead_diameter_mm
    side_px_per_mm = median(side_diameter_v_px_inliers) / bead_diameter_mm  # Z
    side_zero_y_px = median(side_center_y_px_inliers)   # 底面静止时作为 z 零点

    后续运行期（SideDetector，尚未在本模块实现）：
        z_px = side_zero_y_px - center_y_px
        z_mm = z_px / side_px_per_mm

try_finalize() 统计流程：
    1. 样本数 < required_pair_count → 返回 None（继续采集）
    2. 上视直径和侧视水平、竖直投影直径分别取中位数
    3. 联合内点：三个方向同时满足相对中位数容差，Y/Z 使用同一批内点
    4. 内点数 < minimum_inlier_count → 失败，清空样本
    5. 内点静止性：top center_x/y、side center_y 的 P90-P10 spread 不得超过阈值
    6. 对内点再取中位数 → 最终结果；completed=True

add_pair() 拒收条件（返回 False，不抛异常）：
    - 已完成标定（completed=True）
    - pair_id 重复
    - pair_dt_ms 非有限或超过 pair_max_dt_ms
    - 上视 candidate_count != 1，或侧视 candidate_count < 1
    - 直径 / 圆度 / 有限性检查不通过

参数调试速查：
    ------------------------------------------------------------------
    bead_diameter_mm = 1.0
    ------------------------------------------------------------------
    物理磁珠直径（mm），用于由像素直径反推 px_per_mm。

    ------------------------------------------------------------------
    required_pair_count = 15,  minimum_inlier_count = 10
    ------------------------------------------------------------------
    至少收集 15 对才尝试 finalize；直径联合内点至少 10 对才接受标定。

    ------------------------------------------------------------------
    relative_diameter_tolerance = 0.25
    ------------------------------------------------------------------
    相对直径容差：离群帧（尺寸异常、误检）会被联合掩膜剔除。

    ------------------------------------------------------------------
    pair_max_dt_ms = 80.0
    ------------------------------------------------------------------
    与 VisionControlLoop 软件配对阈值一致；超时配对不进入标定样本。

    ------------------------------------------------------------------
    top_diameter_* / side_diameter_* , top/side_circularity_min
    ------------------------------------------------------------------
    单帧入样前的粗筛；上视通常更圆（circularity 阈值更高）。

    ------------------------------------------------------------------
    top_center_spread_max_px = 3.0,  side_center_spread_max_px = 2.0
    ------------------------------------------------------------------
    静止性检查：标定期间磁珠若移动，center 的 P90-P10 会超限并重置全部样本。

与其他模块关系：
    由 app/vision_control_loop.py 调用；输入来自检测器，输出供检测器/runtime 之外的显示层使用。
    侧视 z 标定结果不得写入 magnetic_model 或 RuntimeLoop（调用方约束）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewCalibrationObservation:
    """
    单个相机一帧中的标定候选摘要。

    由 VisionControlLoop 从 RobotDetector / SideDetector 的轮廓几何量构造；
    本模块不保存图像、不保存 pair_id（pair_id 由 DualViewCalibrationSample 保存）。

    字段：
        center_x_px / center_y_px: 磁珠中心（像素）；
        diameter_px: 等效直径（通常来自 minEnclosingCircle 或外接圆）；
        circularity: 圆度，4πA/P²，范围 [0, 1]；
        candidate_count: 本帧通过面积等粗筛的候选个数；
            上视标定要求 == 1，侧视标定要求 >= 1（侧视已选面积最大者）。
    """

    center_x_px: float
    center_y_px: float
    diameter_px: float
    circularity: float
    candidate_count: int
    ### 新增：侧视水平 Y、竖直 Z 投影直径；默认 None 保留原有构造方式。
    diameter_u_px: Optional[float] = None
    diameter_v_px: Optional[float] = None
    ### 方向直径字段结束。


@dataclass
class DualViewCalibrationConfig:
    """
    双视标定配置。

    可通过 VisionControlLoopConfig 或 runtime_ctx.cfg 覆盖（后续轮次接入）。
    """

    # --- 物理参照 ---
    bead_diameter_mm: float = 1.0

    # --- 样本数量 ---
    required_pair_count: int = 15
    minimum_inlier_count: int = 10

    # --- 稳健统计 ---
    relative_diameter_tolerance: float = 0.25
    pair_max_dt_ms: float = 80.0

    # --- 上视单帧入样门限 ---
    top_diameter_min_px: float = 6.0
    top_diameter_max_px: float = 120.0
    top_circularity_min: float = 0.50

    # --- 侧视单帧入样门限（侧视圆度阈值通常低于上视）---
    side_diameter_min_px: float = 4.0
    side_diameter_max_px: float = 100.0
    side_circularity_min: float = 0.30

    # --- 静止性检查（联合内点的 P90-P10 spread 上限，单位 px）---
    top_center_spread_max_px: float = 3.0
    side_center_spread_max_px: float = 2.0


@dataclass(frozen=True)
class DualViewCalibrationSample:
    """
    已成功 add_pair 的一对标定观测（内部存储）。

    pair_id / pair_dt_ms 由 VisionControlLoop 在软件配对时生成。
    """

    pair_id: int
    pair_dt_ms: float
    top: ViewCalibrationObservation
    side: ViewCalibrationObservation


@dataclass(frozen=True)
class DualViewCalibrationResult:
    """
    标定完成后的最终结果（frozen，便于只读传递）。

    比例尺：
        top_px_per_mm / top_px_per_cm — 上视独立比例（不与其他模块混用旧 380/3 常数）；
        side_px_per_mm — 侧视 z 方向像素/毫米。

    零点：
        side_zero_y_px — 底面静止时侧视 center_y 中位数，运行期 z_px 相对此值。

    诊断：
        accepted_pair_count — add_pair 成功总数（finalize 前）；
        inlier_pair_count   — 直径联合内点数量。
    """

    top_px_per_mm: float
    top_px_per_cm: float
    side_px_per_mm: float
    side_zero_y_px: float

    top_diameter_px: float
    side_diameter_px: float  # 竖直投影直径的内点中位数（Z）
    accepted_pair_count: int
    inlier_pair_count: int

    # [3D_PID] BEGIN: 联合内点新增统计结果。
    # 新字段全部追加在旧必填字段之后并提供默认值，保证旧代码按位置构造
    # DualViewCalibrationResult 时参数顺序不变。Calibrator 只报告统计量，
    # 不 import RobotDetector，也不决定全局坐标原点。
    side_px_per_cm: float = 0.0
    # Median centres of the final joint inliers, retained for diagnostics;
    # the side-view Z origin uses the median row of the bottom-resting bead.
    top_center_x_px: float = float("nan")
    top_center_y_px: float = float("nan")
    side_center_x_px: float = float("nan")
    side_center_y_px: float = float("nan")
    # Filled by VisionControlLoop using the side image centre column and
    # SIDE_Y_SIGN. Raw calibrator results leave these unset.
    side_y_zero_x_px: Optional[float] = None
    side_y_sign: Optional[int] = None
    ### 新增：独立侧视 Y 比例尺；原 side_px_per_mm 继续表示 Z 比例尺。
    side_y_px_per_mm: Optional[float] = None
    ### 独立 Y 比例尺字段结束。
    # [3D_PID] END: 双视 y 标定所需统计与派生字段。


def _is_finite_observation(obs: ViewCalibrationObservation) -> bool:
    """检查观测数值均为有限值（reject NaN/Inf）。"""
    values = (
        obs.center_x_px,
        obs.center_y_px,
        obs.diameter_px,
        obs.circularity,
    )
    return all(math.isfinite(float(v)) for v in values) and math.isfinite(float(obs.candidate_count))


def _percentile_spread(values: Sequence[float]) -> float:
    """
    计算 P90 - P10  spread，用于稳健估计标定期间是否发生位移。

    比 max-min 更抗个别离群帧；样本为空时返回 inf 以触发静止性失败。
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("inf")
    p90 = float(np.percentile(arr, 90))
    p10 = float(np.percentile(arr, 10))
    return p90 - p10


class DualViewCalibrator:
    """
    双视配对标定管理器（有状态）。

    典型用法：
        calibrator = DualViewCalibrator(cfg)
        calibrator.reset()
        while not calibrator.completed:
            if calibrator.add_pair(pair_id=..., pair_dt_ms=..., top=..., side=...):
                print(calibrator.progress, "/", calibrator.required_count)
            result = calibrator.try_finalize()  # 未满 15 对时返回 None
        # result 为 DualViewCalibrationResult

    失败恢复：
        try_finalize 因内点不足或静止性失败时会清空 _samples，需重新采集 15 对。
        last_error 记录失败原因。
    """

    def __init__(self, cfg: Optional[DualViewCalibrationConfig] = None):
        self.cfg = cfg if cfg is not None else DualViewCalibrationConfig()
        self._samples: List[DualViewCalibrationSample] = []
        self._completed = False
        self._result: Optional[DualViewCalibrationResult] = None
        self._last_error: Optional[str] = None

    @property
    def progress(self) -> int:
        """已成功 add_pair 的样本对数。"""
        return len(self._samples)

    @property
    def required_count(self) -> int:
        """finalize 所需的最小样本对数（默认 15）。"""
        return int(self.cfg.required_pair_count)

    @property
    def completed(self) -> bool:
        """try_finalize 是否已成功完成。"""
        return self._completed

    @property
    def last_error(self) -> Optional[str]:
        """最近一次 try_finalize 失败原因；成功完成后为 None。"""
        return self._last_error

    def reset(self) -> None:
        """清空样本与标定结果，重新开始采集。"""
        self._samples.clear()
        self._completed = False
        self._result = None
        self._last_error = None

    def add_pair(
        self,
        *,
        pair_id: int,
        pair_dt_ms: float,
        top: ViewCalibrationObservation,
        side: ViewCalibrationObservation,
        return_rejection_reason: bool = False,
    ) -> bool:
        """
        尝试接收一对标定观测。

        全部入样条件见模块文档「add_pair() 拒收条件」。
        成功时 append 到内部列表并返回 True；任一条件不满足返回 False。
        """
        def reject(reason: str):
            return (False, reason) if return_rejection_reason else False

        def accept():
            return (True, "accepted") if return_rejection_reason else True

        if self._completed:
            return reject("already_completed")

        cfg = self.cfg

        # --- 配对元数据 ---
        if not math.isfinite(float(pair_dt_ms)):
            return reject("invalid_pair_delay")
        if any(sample.pair_id == pair_id for sample in self._samples):
            return reject("duplicate_pair_id")
        if float(pair_dt_ms) > float(cfg.pair_max_dt_ms):
            return reject(f"pair_delay>{cfg.pair_max_dt_ms}ms")

        # --- 候选数量 ---
        # 上视仍要求唯一候选，避免标定对象存在歧义。
        if top.candidate_count != 1:
            return reject(f"top_candidate_count={top.candidate_count},need=1")

        # 侧视允许存在多个面积粗筛候选。
        # SideDetector 已经使用其中面积最大的候选作为磁珠。
        if side.candidate_count < 1:
            return reject(f"side_candidate_count={side.candidate_count},need>=1")
        if not _is_finite_observation(top) or not _is_finite_observation(side):
            return reject("non_finite_observation")

        # --- 直径与圆度粗筛 ---
        top_diameter = float(top.diameter_px)
        ### 新增：侧视必须同时具有有限、正值的水平和竖直直径才能参与标定。
        if side.diameter_u_px is None or side.diameter_v_px is None:
            return reject("side_projection_diameter_missing")
        side_diameters = (float(side.diameter_u_px), float(side.diameter_v_px))
        if not all(math.isfinite(d) and d > 0.0 for d in side_diameters):
            return reject(f"invalid_side_diameters={side_diameters}")
        ### 方向直径有效性检查结束。
        if not (cfg.top_diameter_min_px <= top_diameter <= cfg.top_diameter_max_px):
            return reject(f"top_diameter={top_diameter:.2f}px,outside=[{cfg.top_diameter_min_px},{cfg.top_diameter_max_px}]")
        ### 调整：现有侧视标定直径上下界同时约束 Y/Z 两个方向。
        if not all(cfg.side_diameter_min_px <= d <= cfg.side_diameter_max_px
                   for d in side_diameters):
            return reject(f"side_diameters={side_diameters},outside=[{cfg.side_diameter_min_px},{cfg.side_diameter_max_px}]")
        ### 方向直径上下界检查结束。
        if float(top.circularity) < float(cfg.top_circularity_min):
            return reject(f"top_circularity={top.circularity:.3f},need>={cfg.top_circularity_min}")
        if float(side.circularity) < float(cfg.side_circularity_min):
            return reject(f"side_circularity={side.circularity:.3f},need>={cfg.side_circularity_min}")

        self._samples.append(
            DualViewCalibrationSample(
                pair_id=int(pair_id),
                pair_dt_ms=float(pair_dt_ms),
                top=top,
                side=side,
            )
        )
        return accept()

    def try_finalize(self) -> Optional[DualViewCalibrationResult]:
        """
        样本数足够时执行稳健统计；成功则 completed=True。

        返回：
            DualViewCalibrationResult — 标定成功；
            None — 样本不足，或内点/静止性检查失败（失败时清空样本）。

        已成功标定后再次调用会直接返回缓存结果。
        """
        if self._completed and self._result is not None:
            return self._result

        cfg = self.cfg
        if len(self._samples) < cfg.required_pair_count:
            return None

        ### 调整步骤 1：统一计算上视、侧视 Y/Z 直径中位数及原有相对容差。
        diameters = np.asarray(
            [[s.top.diameter_px, s.side.diameter_u_px, s.side.diameter_v_px]
             for s in self._samples], dtype=float,
        )
        medians = np.median(diameters, axis=0)
        tolerances = float(cfg.relative_diameter_tolerance) * medians
        ### 三方向直径统计结束。

        ### 调整步骤 2：三个直径同时合格，确保 Y/Z 比例尺来自同一批联合内点。
        inlier_mask = np.all(np.abs(diameters - medians) <= tolerances, axis=1)
        ### 联合内点筛选结束。
        inlier_samples = [sample for sample, keep in zip(self._samples, inlier_mask) if keep]
        inlier_count = len(inlier_samples)

        if inlier_count < cfg.minimum_inlier_count:
            self._last_error = "直径有效内点不足"
            self._samples.clear()
            self._completed = False
            self._result = None
            return None

        # ==================================================================
        # 步骤 3：静止性检查 — 磁珠在底面标定时 center 应几乎不动
        # ==================================================================
        # [3D_PID] BEGIN: 在同一批联合直径内点上检查四个图像方向的静止性。
        # 旧逻辑已有 top x/y 和 side y；这里把 side x 纳入相同的 P90-P10
        # 检查，因为 side x 后续直接用于物理 y 标定，移动样本不能被接受。
        top_center_x = [s.top.center_x_px for s in inlier_samples]
        top_center_y = [s.top.center_y_px for s in inlier_samples]
        side_center_x = [s.side.center_x_px for s in inlier_samples]
        side_center_y = [s.side.center_y_px for s in inlier_samples]

        top_spread_x = _percentile_spread(top_center_x)
        top_spread_y = _percentile_spread(top_center_y)
        side_spread_x = _percentile_spread(side_center_x)
        side_spread_y = _percentile_spread(side_center_y)

        if (
            top_spread_x > cfg.top_center_spread_max_px
            or top_spread_y > cfg.top_center_spread_max_px
            or side_spread_x > cfg.side_center_spread_max_px
            or side_spread_y > cfg.side_center_spread_max_px
        ):
            self._last_error = "标定期间磁珠发生运动"
            self._samples.clear()
            self._completed = False
            self._result = None
            return None
        # [3D_PID] END: 四方向静止性检查。

        # ==================================================================
        # 步骤 4：对内点取中位数，计算比例尺与 side_zero_y_px
        # ==================================================================
        # [3D_PID] BEGIN: 输出联合内点的上下视中心中位数和侧视 px/cm。
        # 中心统计量用于诊断和侧视落底 Z 零点；侧视 Y 零点由调用方取画面中心列。
        inlier_side_center_y = np.asarray([s.side.center_y_px for s in inlier_samples], dtype=float)
        inlier_top_center_x = np.asarray([s.top.center_x_px for s in inlier_samples], dtype=float)
        inlier_top_center_y = np.asarray([s.top.center_y_px for s in inlier_samples], dtype=float)
        inlier_side_center_x = np.asarray([s.side.center_x_px for s in inlier_samples], dtype=float)

        ### 调整：取联合内点的直径中位数；side_diameter_px 对应竖直 Z 直径。
        top_diameter_px, side_diameter_u_px, side_diameter_px = map(
            float, np.median(diameters[inlier_mask], axis=0)
        )
        ### 联合内点直径中位数计算结束。
        side_zero_y_px = float(np.median(inlier_side_center_y))
        top_center_x_px = float(np.median(inlier_top_center_x))
        top_center_y_px = float(np.median(inlier_top_center_y))
        side_center_x_px = float(np.median(inlier_side_center_x))
        side_center_y_px = side_zero_y_px

        bead_diameter_mm = float(cfg.bead_diameter_mm)
        if bead_diameter_mm <= 0.0 or not math.isfinite(bead_diameter_mm):
            self._last_error = "bead_diameter_mm 无效"
            self._samples.clear()
            self._completed = False
            self._result = None
            return None

        top_px_per_mm = top_diameter_px / bead_diameter_mm
        top_px_per_cm = 10.0 * top_px_per_mm
        side_px_per_mm = side_diameter_px / bead_diameter_mm
        side_px_per_cm = 10.0 * side_px_per_mm

        self._result = DualViewCalibrationResult(
            top_px_per_mm=top_px_per_mm,
            top_px_per_cm=top_px_per_cm,
            side_px_per_mm=side_px_per_mm,
            side_px_per_cm=side_px_per_cm,
            side_zero_y_px=side_zero_y_px,
            top_center_x_px=top_center_x_px,
            top_center_y_px=top_center_y_px,
            side_center_x_px=side_center_x_px,
            side_center_y_px=side_center_y_px,
            top_diameter_px=top_diameter_px,
            side_diameter_px=side_diameter_px,
            ### 新增：水平直径中位数除以磁珠真实直径，得到 Y 的 px/mm。
            side_y_px_per_mm=side_diameter_u_px / bead_diameter_mm,
            ### 独立 Y 比例尺计算结束。
            accepted_pair_count=len(self._samples),
            inlier_pair_count=inlier_count,
        )
        self._completed = True
        self._last_error = None
        return self._result
        # [3D_PID] END: 联合内点中心及比例结果。
