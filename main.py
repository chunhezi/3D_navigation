import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from Widgets_window import Ui_Form
import cv2
import numpy as np
import threading
import serial.tools.list_ports
# ===== 设备层、运行时工厂与可扩展模式管理 =====
from core.system_factory import create_runtime_context
from devices.device_manager import DeviceManager
from mode_manager import ModeManager
from vision.robot_detector import DetectorConfig, RobotDetector
from vision.overlay_renderer import OverlayRenderer, OverlayRendererConfig
from app.runtime_loop import RuntimeLoop, RuntimeLoopConfig
from app.vision_control_loop import VisionControlLoop, VisionControlLoopConfig

# UI 控制模式字符串 → ModeManager 内部模式名映射
# [3D_PID] BEGIN: UI 模式名 -> ModeManager 内部模式名映射。
# 当前仅注册三维模式，保留映射和下拉框以便后续扩展。
MODE_NAME_MAP = {
    "stabilize_3d_mode": "stabilize_3d",
}
# [3D_PID] END: UI/内部模式映射。

# 各模式正式实验 PKL 文件名（保持与历史分析脚本兼容）
# 主流程通过当前模式的 save_motion_log() 保存实验数据。
MODE_PKL_FILENAME_MAP = {
    "stabilize_3d": "stabilize_3d_data.pkl",
}

EXPERIMENT_RECORD_ROOT = Path("recordings")

# ==========================================================
# GUI DISPLAY FRAME DECIMATION
#
# 仅影响 Qt GUI 显示。
#
# 不允许影响：
# - 相机采集
# - 上视检测
# - 侧视检测
# - 双目标定
# - 3D measurement pair
# - XYZ状态更新
# - PID控制
# - force/current计算
# - 串口驱动
#
# 示例：
#
# CYCLE=3, KEEP=2:
#
# source:
# 1 2 3 4 5 6 7 8 9 ...
#
# GUI:
# 1 2 - 4 5 - 7 8 - ...
#
# 即每3帧显示2帧。
# ==========================================================
GUI_FRAME_CYCLE = 4
GUI_FRAME_KEEP = 3

# Set False to run vision and coil control without the broken Arduino pump.
PUMP_ENABLED = False

port_state = False
motion_state = False
camera_state = False
video_state = False
i_motion = 0
roi_l = 500        # 长
# roi_w = 185        # 宽
roi_w = 185        # 宽

# ============== 本次视觉修改【上视参数】：面积初值开始 ==============
# 作用：配合两路统一预处理，改用未腐蚀暗珠前景的面积；重新标定前须在候选窗确认。
min_area = 10   # 轮廓面积下限（px²）；覆盖 DetectorConfig 默认值
max_area = 200   # 轮廓面积上限（px²）；适当放宽，避免约 12 px 磁珠轮廓被误筛
# ============== 本次视觉修改【上视参数】：面积初值结束 ==============
Camera_index = 0

# 控制周期保持原值，三维参考由 reference_generator_3d 提供。
dt = 0.15


class WinForm(QtWidgets.QWidget, Ui_Form):
    camera_frames_ready = QtCore.pyqtSignal(object)
    video_recording_started = QtCore.pyqtSignal(str)
    video_recording_failed = QtCore.pyqtSignal(str)
    video_recording_stopped = QtCore.pyqtSignal(str, object)
    drive_close_finished = QtCore.pyqtSignal(bool, str)

    def __init__(self):
        self._gui_source_frame_seq = 0
        self._gui_emit_frame_seq = 0

        super().__init__()
        self.setupUi(self)
        self._setup_multi_camera_view_layout()
        self.camera_frames_ready.connect(self._show_frame_bundle_on_labels)
        self.video_recording_started.connect(self._on_video_recording_started)
        self.video_recording_failed.connect(self._on_video_recording_failed)
        self.video_recording_stopped.connect(self._on_video_recording_stopped)
        self.drive_close_finished.connect(self._on_drive_close_finished)

        # 实验会话状态（每次 start motion 冻结一组路径）
        self._experiment_mode_name: str = ""
        self._experiment_timestamp: str = ""
        self._experiment_session_dir: Optional[Path] = None
        self._experiment_video_path: Optional[Path] = None
        self._experiment_video_info_path: Optional[Path] = None
        self._experiment_pkl_path: Optional[Path] = None
        self._experiment_motion_log_saved: bool = False
        self._experiment_video_saved: bool = False
        self._experiment_stop_pending: bool = False
        self._motion_start_pending: bool = False
        self._video_recording_active: bool = False
        self._drive_close_requested = False
        self._drive_close_in_progress = False

        # 波特率选项
        self.comboBox_baud.addItem("921600", "921600")
        self.comboBox_baud.addItem("460800", "460800")
        self.comboBox_baud.addItem("230400", "230400")
        self.comboBox_baud.addItem("1000000", "1000000")
        self.comboBox_baud.addItem("38400", "38400")
        self.comboBox_baud.addItem("19200", "19200")
        self.comboBox_baud.addItem("9600", "9600")
        self.comboBox_baud.addItem("4800", "4800")
        self.comboBox_baud.setCurrentIndex(3)

        # ==================== 泵相关：扰动时序参数开始 ====================
        # 功能：保留强弱周期、预留状态与恢复时长；dt 同时供控制循环使用。
        self.dt = dt
        self.N_strong = int(2 / self.dt)  # 40
        self.N_weak = int(6 / self.dt)    # 60
        self.N_period = self.N_strong + self.N_weak
        self.dist_flag = 0  # 0=弱, 1=强
        self.prev_dist_flag = None
        self.T_recover = 0.00
        self.N_recover = int(round(self.T_recover / self.dt))
        # ==================== 泵相关：扰动时序参数结束 ====================

        # ===== 【REFACTOR_PHASE6】设备层 + 运行时上下文 =====
        self.cfg = {
            "dt": self.dt,
            "model_dt": 0.15,
            "bead_diameter_mm": 1.0,
            # ============== 本次视觉修改【两路公用】：腐蚀配置开始 ==============
            # 作用：上视、侧视的核大小和次数只在这里统一调整；0 表示完全不腐蚀。
            "bead_erode_kernel_size": 2,
            "bead_erode_iterations": 0,
            # ============== 本次视觉修改【两路公用】：腐蚀配置结束 ==============
            "dual_calibration_required_pairs": 15,
            "dual_calibration_minimum_inliers": 10,
            "dual_calibration_relative_diameter_tolerance": 0.25,
            "dual_calibration_pair_max_dt_ms": 80.0,
            "top_calibration_diameter_min_px": 5.0,
            "top_calibration_diameter_max_px": 35.0,
            "top_calibration_circularity_min": 0.40,
            "side_calibration_diameter_min_px": 4.0,
            "side_calibration_diameter_max_px": 20.0,
            "side_calibration_circularity_min": 0.30,
            "side_threshold": 90,
            "side_roi_center_offset_x": 0,
            "side_roi_center_offset_y": -50,
            "top_calibration_center_spread_max_px": 3.0,
            "side_calibration_center_spread_max_px": 2.0,
        }
        self.devices = DeviceManager(cfg=self.cfg)
        self.runtime_ctx = create_runtime_context(cfg=self.cfg, devices=self.devices)
        self.control_mode = "stabilize_3d_mode"
        self.mode_manager = ModeManager(self.runtime_ctx)
        self.mode_manager.switch_mode(MODE_NAME_MAP.get(self.control_mode, "stabilize_3d"))

        # ===== 【REFACTOR_PHASE10】单帧控制循环 =====
        self.runtime_loop = RuntimeLoop(
            ctx=self.runtime_ctx,
            mode_manager=self.mode_manager,
            devices=self.devices,
            cfg=RuntimeLoopConfig(
                dt=self.dt,
                # ==================== 泵相关：调度配置开始 ====================
                # 功能：传入强弱周期及恢复计数，不改变原有时序。
                n_weak=self.N_weak,
                n_period=self.N_period,
                n_recover=self.N_recover,
                pump_enabled=PUMP_ENABLED,
                # ==================== 泵相关：调度配置结束 ====================
                current_limit=8.0,
            ),
        )

        # 初始化槽函数（需在 devices / runtime_ctx 创建之后）
        self.slot_init()

        # 扫描串口
        self.refresh_serial_ports()

        # 摄像头线程（每次打开相机时重建 Thread，避免重复 start 报错）
        self.thread_camera = None

        # ==================== 泵相关：连接状态开始 ====================
        # 功能：缓存 Arduino 泵句柄与串口连接状态，与设备管理器同步。
        self.pump = None
        self.arduino_port_state = False
        if not PUMP_ENABLED:
            for widget_name in ("label_arduino_port", "comboBox_arduino_port", "button_open_arduino", "button_test_pump"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setEnabled(False)
            print("[INFO] Pump control disabled (PUMP_ENABLED=False)")
        # ==================== 泵相关：连接状态结束 ====================

        # ===== 【REFACTOR_PHASE7】机器人视觉检测（从 detect 剥离）=====
        self.robot_detector = RobotDetector(
            DetectorConfig(
                min_area=min_area,
                max_area=max_area,
                roi_half_length=roi_l,
                roi_half_width=roi_w,
                # ============== 本次视觉修改【上视接入】：公用参数传入开始 ==============
                # 作用：读取两路公用主配置，侧视创建时直接复用这里的实际值。
                erode_kernel_size=self.cfg["bead_erode_kernel_size"],
                erode_iterations=self.cfg["bead_erode_iterations"],
                # ============== 本次视觉修改【上视接入】：公用参数传入结束 ==============
            )
        )
        # ===== 【REFACTOR_PHASE9】画面叠加渲染（从 detect 剥离）=====
        self.overlay_renderer = OverlayRenderer(
            cfg=OverlayRendererConfig(),
        )

        # ==================== 泵相关：测试按钮开始 ====================
        # 功能：创建现有测试泵按钮，点击后请求强扰动档。
        try:
            # 你原来的按钮在 (190,4,90,22)，右侧放一个 90 宽的按钮
            self.button_test_pump = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
            self.button_test_pump.setGeometry(QtCore.QRect(286, 4, 90, 22))
            self.button_test_pump.setObjectName("button_test_pump")
            self.button_test_pump.setText("测试泵")
        except Exception as e:
            print(f"[WARN] create test pump button failed: {e}")
            self.button_test_pump = None
        # ==================== 泵相关：测试按钮结束 ====================

        # 保留模式选择所在第二行的显示空间。
        try:
            self.textEdit.setGeometry(QtCore.QRect(0, 56, 481, 113))
        except Exception as e:
            print(f"[WARN] adjust textEdit geometry failed: {e}")

        # ===== 模式选择：保留后续扩展入口 =====
        try:
            self.comboBox_remote_mode = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
            # 【REMOTE_STEP2】第二行放置，避免与串口/泵按钮等控件重叠
            self.comboBox_remote_mode.setGeometry(QtCore.QRect(110, 30, 250, 22))
            self.comboBox_remote_mode.setObjectName("comboBox_remote_mode")
            # 当前只显示三维模式，回调和切换生命周期继续保留。
            self.comboBox_remote_mode.addItem("stabilize_3d_mode")
            self.comboBox_remote_mode.setCurrentIndex(0)
        except Exception as e:
            print(f"[WARN][REMOTE_STEP2] create remote mode combo failed: {e}")
            self.comboBox_remote_mode = None

        if self.comboBox_remote_mode is not None:
            self.comboBox_remote_mode.currentIndexChanged.connect(self.on_remote_mode_changed)
        # ===== 模式选择结束 =====

        # ==================== 泵相关：测试按钮绑定开始 ====================
        # 功能：将测试按钮连接到现有强扰动请求。
        if self.button_test_pump is not None:
            self.button_test_pump.clicked.connect(self.test_pump)
        # ==================== 泵相关：测试按钮绑定结束 ====================

    def _drive_motion_up(self):
        if self.devices.drive is not None:
            self.devices.drive.motion_up()

    def _drive_motion_down(self):
        if self.devices.drive is not None:
            self.devices.drive.motion_down()

    def _drive_motion_left(self):
        if self.devices.drive is not None:
            self.devices.drive.motion_left()

    def _drive_motion_right(self):
        if self.devices.drive is not None:
            self.devices.drive.motion_right()

    def _drive_motion_stop(self):
        if self.devices.drive is not None:
            self.devices.drive.motion_stop()

    def slot_init(self):
        # 串口按钮
        self.button_open_serial.clicked.connect(self.serial_open_off)
        # 刷新串口按钮
        self.button_refresh_serial.clicked.connect(self.refresh_serial_ports)
        # 退出
        self.button_exit.clicked.connect(self.safe_exit)

        # 摄像头函数
        self.button_camera.clicked.connect(self.camera_event)
        # path motion
        self.button_motion.clicked.connect(self.motion_event)

        # up/down/left/right/stop
        self.button_up.clicked.connect(self._drive_motion_up)
        self.button_down.clicked.connect(self._drive_motion_down)
        self.button_left.clicked.connect(self._drive_motion_left)
        self.button_right.clicked.connect(self._drive_motion_right)
        self.button_stop.clicked.connect(self._drive_motion_stop)

        # Arduino 泵串口打开/关闭
        # ==================== 泵相关：连接按钮绑定开始 ====================
        # 功能：连接泵串口按钮与原有连接/关闭回调。
        self.button_open_arduino.clicked.connect(self.arduino_open_off)
        # ==================== 泵相关：连接按钮绑定结束 ====================

    def on_remote_mode_changed(self, idx: int):
        """
        UI 模式选择变化时调用。

        该函数只负责：
        1. 更新 control_mode 字符串（供 overlay、日志等沿用旧逻辑）；
        2. 通知 ModeManager 切换模式，不直接修改具体控制算法。
        """
        # ===== 【REMOTE_STEP2】mode string 映射：开始 =====
        # 顺序必须与 comboBox_remote_mode.addItem 一致
        # 下拉索引对应表必须与 addItem 顺序完全一致，供后续新增模式使用。
        # 把新模式放在末尾可避免改变旧模式已有索引及默认行为。
        modes = [
            "stabilize_3d_mode",
        ]
        idx = int(idx)
        idx = max(0, min(idx, len(modes) - 1))
        self.control_mode = modes[idx]
        print(f"[REMOTE_STEP2] control_mode set to: {self.control_mode}")
        # [3D_PID] END: 模式下拉索引映射。
        # ===== 【REMOTE_STEP2】mode string 映射：结束 =====

        # ===== 【REFACTOR_PHASE5】通过 ModeManager 切换运动模式：开始 =====
        mode_key = MODE_NAME_MAP.get(self.control_mode)
        if mode_key is not None and hasattr(self, "mode_manager"):
            self.mode_manager.switch_mode(mode_key)
            print(f"[REFACTOR_PHASE5] mode_manager switched to: {mode_key}")
        # ===== 【REFACTOR_PHASE5】通过 ModeManager 切换运动模式：结束 =====


    def reset_overlay_layers(self):
        self.overlay_renderer.reset()

    def _is_camera_active(self) -> bool:
        return bool(camera_state)

    def _is_motion_active(self) -> bool:
        return bool(motion_state)

    def _is_video_enabled(self) -> bool:
        return bool(video_state)

    def _is_drive_enabled(self) -> bool:
        return bool(port_state)

    def _is_drive_close_requested(self) -> bool:
        return bool(self._drive_close_requested)

    def _emit_drive_close_finished(self, success: bool, message: str) -> None:
        self.drive_close_finished.emit(bool(success), str(message))

    @QtCore.pyqtSlot(bool, str)
    def _on_drive_close_finished(self, success: bool, message: str) -> None:
        global port_state
        self._drive_close_requested = False
        self._drive_close_in_progress = False
        # 无论驱动失能是否收到完整应答，本地串口已关闭后 UI 进入未连接状态
        port_state = False
        self.button_open_serial.setText("打开串口")
        self.button_open_serial.setEnabled(True)
        if success:
            print("[Drive] serial closed after stop motion")
        else:
            print(f"[WARN] drive close completed with warning: {message}")

    def _get_i_motion(self) -> int:
        return int(i_motion)

    def _set_i_motion(self, value: int) -> None:
        global i_motion
        i_motion = int(value)

    def _get_control_mode(self) -> str:
        return self.control_mode




    # =====================================================================
    # 多相机图像显示区布局（仅 UI 显示层，与检测/控制算法无关）
    #
    # 坐标系说明（Qt 窗口内）：
    #   - 原点 (0,0) 在窗口左上角
    #   - x 向右增大，y 向下增大
    #   - setGeometry(x, y, w, h) 中 (x,y) 为控件左上角，w/h 为宽高（像素）
    #
    # 2×2 布局示意（修改布局时主要改下面几个变量）：
    #
    #   ┌─────────────────────┬─────────────────────┐  ← 第一行 y = y0
    #   │ (1,1) label_top_main│ (1,2) 上视调试 2×2  │
    #   │   640 × 360  16:9   │ 二值化 | 腐蚀+ROI   │
    #   │                     │ 候选   | 筛选结果   │
    #   ├─────────────────────┼─────────────────────┤  ← 第二行 y = y0+view_h+gap_y
    #   │ (2,1) label_side_  │ (2,2) 侧视调试 2×2  │
    #   │       main          │ 二值化 | 腐蚀+ROI   │
    #   │   640 × 360  16:9   │ 候选   | 筛选结果   │
    #   └─────────────────────┴─────────────────────┘
    #     ↑ x0              ↑ x0+view_w+gap_x
    #
    # 常用修改入口：
    #   - 整体下移/右移：改 x0, y0
    #   - 主画面变大：改 view_w, view_h（保持 view_w/view_h=16/9 不变形）
    #   - 左右两列间距：改 gap_x
    #   - 上下两行间距：改 gap_y
    #   - 窗口装不下时：改 self.resize / setMinimumSize（约 2*view_w+gap_x+边距）
    # =====================================================================

    def _setup_multi_camera_view_layout(self):
        """创建 2×2 图像显示区，并复用原 label_show_camera 作为 (1,1) 主视角。"""
        # 窗口总尺寸：需容纳两列 view_w + 一列 gap_x + 左右边距 x0
        # 高度：需容纳两行 view_h + 一行 gap_y + 顶部控件区 y0 + 底边距
        self.resize(1000, 900)
        self.setMinimumSize(1360, 1040)

        # --- 全局布局参数（改界面优先改这里）---
        x0 = 20          # 显示区整体左边距（距窗口左边缘）
        y0 = 250         # 显示区整体顶边距（距窗口上边缘；上方留给串口/模式选择等控件）
        view_w = 640     # 每个「大格」的宽度（16:9 时对应 view_h=360）
        view_h = 360     # 每个「大格」的高度；务必满足 view_w : view_h = 16 : 9
        gap_x = 20       # 左列与右列之间的水平间距
        gap_y = 20       # 上行与下行之间的垂直间距

        # --- 四个象限左上角坐标（由上面参数自动推导，一般不用手改）---
        q1_x, q1_y = x0, y0                                    # (1,1) 左上：上视主视角
        q2_x, q2_y = x0 + view_w + gap_x, y0                    # (1,2) 右上：上视调试
        q3_x, q3_y = x0, y0 + view_h + gap_y                    # (2,1) 左下：侧视主视角预留
        q4_x, q4_y = x0 + view_w + gap_x, y0 + view_h + gap_y   # (2,2) 右下：侧视调试预留

        # ----- (1,1) 上视相机主视角：复用 Widgets_window 里原有的 label_show_camera -----
        self.label_top_main = self.label_show_camera
        self.label_top_main.setGeometry(QtCore.QRect(q1_x, q1_y, view_w, view_h))
        self.label_top_main.setObjectName("label_top_main")
        self._style_video_label(self.label_top_main, "上视相机主视角")

        # ----- (1,2) 上视调试：在右上格内再 2×2 均分（与主格同尺寸，比例更接近 16:9）-----
        sub_gap = 8      # 四个小图之间的水平/垂直间距
        cell_w = (view_w - sub_gap) // 2
        cell_h = (view_h - sub_gap) // 2

        # 左上：二值化（key = "top_binary"）
        self.label_top_binary = self._create_video_label(
            "二值化",
            q2_x,
            q2_y,
            cell_w,
            cell_h,
        )
        # 右上：腐蚀 + ROI（key = "top_eroded_roi"）
        self.label_top_eroded_roi = self._create_video_label(
            "腐蚀 + ROI",
            q2_x + cell_w + sub_gap,
            q2_y,
            cell_w,
            cell_h,
        )
        # 左下：候选轮廓 + 圆心（key = "top_candidates"）
        self.label_top_candidates = self._create_video_label(
            "候选轮廓",
            q2_x,
            q2_y + cell_h + sub_gap,
            cell_w,
            cell_h,
        )
        # 右下：面积筛选结果（key = "top_filter_result"）
        self.label_top_filter_result = self._create_video_label(
            "筛选结果",
            q2_x + cell_w + sub_gap,
            q2_y + cell_h + sub_gap,
            cell_w,
            cell_h,
        )

        # ----- (2,1) 侧视相机主视角 -----
        self.label_side_main = self._create_video_label(
            "侧视相机主视角",
            q3_x,
            q3_y,
            view_w,
            view_h,
        )

        # ----- (2,2) 侧视调试：在右下格内 2×2 均分 -----
        # 左上：侧视二值化（key = "side_binary"）
        self.label_side_binary = self._create_video_label(
            "侧视二值化",
            q4_x,
            q4_y,
            cell_w,
            cell_h,
        )
        # 右上：侧视腐蚀 + ROI（key = "side_eroded_roi"）
        self.label_side_eroded_roi = self._create_video_label(
            "侧视腐蚀 + ROI",
            q4_x + cell_w + sub_gap,
            q4_y,
            cell_w,
            cell_h,
        )
        # 左下：侧视候选轮廓（key = "side_candidates"）
        self.label_side_candidates = self._create_video_label(
            "侧视候选轮廓",
            q4_x,
            q4_y + cell_h + sub_gap,
            cell_w,
            cell_h,
        )
        # 右下：侧视筛选结果（key = "side_filtered"）
        self.label_side_filtered = self._create_video_label(
            "侧视筛选结果",
            q4_x + cell_w + sub_gap,
            q4_y + cell_h + sub_gap,
            cell_w,
            cell_h,
        )

    def _style_video_label(self, label, text=""):
        """
        统一视频格样式：深灰底 + 居中占位文字 + 细边框。

        无画面时显示 text；有画面时 _show_frame_on_label 会 setText("") 并 setPixmap。
        修改外观：改下面 setStyleSheet 中的颜色/字号即可。
        """
        label.setText(text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "QLabel {"
            "background-color: #202020;"   # 无信号时的背景色
            "color: #cccccc;"              # 占位文字颜色
            "border: 1px solid #707070;" # 边框
            "font-size: 14px;"
            "}"
        )
        # False：图像按 _show_frame_on_label 中 resize 后的尺寸显示，不拉伸变形
        label.setScaledContents(False)

    def _create_video_label(self, text, x, y, w, h):
        """
        动态创建一个 QLabel 并放入当前窗口。

        参数：
            text: 无画面时的占位说明文字
            x, y, w, h: 与 setGeometry 相同，单位像素
        返回：
            新建的 QLabel，保存到 self.label_* 供后续显示帧使用
        """
        label = QtWidgets.QLabel(self)
        label.setGeometry(QtCore.QRect(x, y, w, h))
        self._style_video_label(label, text)
        label.show()
        return label

    def _emit_camera_frame_bundle(self, frame_bundle):
        """
        相机线程回调入口（由 VisionControlLoop.on_frame_ready 调用）。

        禁止在相机子线程里直接 label.setPixmap()，否则可能闪退；
        这里只 emit 信号，实际绘制在 Qt 主线程的 _show_frame_bundle_on_labels 中完成。

        GUI 显示抽帧仅发生在此处（控制链完成之后），不影响检测/PID/串口。
        """
        if not hasattr(self, "_gui_source_frame_seq"):
            self._gui_source_frame_seq = 0

        if not hasattr(self, "_gui_emit_frame_seq"):
            self._gui_emit_frame_seq = 0

        self._gui_source_frame_seq += 1

        cycle = max(1, int(GUI_FRAME_CYCLE))
        keep = max(1, min(int(GUI_FRAME_KEEP), cycle))

        cycle_pos = (self._gui_source_frame_seq - 1) % cycle

        # 只丢 GUI 显示帧
        if cycle_pos >= keep:
            return

        self._gui_emit_frame_seq += 1

        if isinstance(frame_bundle, dict):
            # 浅拷贝 dict 即可，不要复制里面的大型 ndarray 图像。
            gui_bundle = dict(frame_bundle)
            gui_bundle["_debug_gui_seq"] = self._gui_emit_frame_seq
            gui_bundle["_debug_source_seq"] = self._gui_source_frame_seq
            gui_bundle["_debug_emit_time"] = time.perf_counter()
            gui_bundle["_debug_gui_cycle"] = cycle
            gui_bundle["_debug_gui_keep"] = keep
        else:
            gui_bundle = frame_bundle

        self.camera_frames_ready.emit(gui_bundle)

    def _show_frame_bundle_on_labels(self, frame_bundle):
        if frame_bundle is None:
            return

        # ==========================================================
        # DEBUG: GUI 性能计时
        # ==========================================================
        gui_t0 = time.perf_counter()

        debug_seq = None
        debug_source_seq = None
        debug_lag_ms = None
        debug_cycle = None
        debug_keep = None

        if isinstance(frame_bundle, dict):
            debug_seq = frame_bundle.get("_debug_gui_seq")
            debug_source_seq = frame_bundle.get("_debug_source_seq")
            debug_cycle = frame_bundle.get("_debug_gui_cycle")
            debug_keep = frame_bundle.get("_debug_gui_keep")
            emit_time = frame_bundle.get("_debug_emit_time")

            if emit_time is not None:
                debug_lag_ms = (
                    time.perf_counter() - emit_time
                ) * 1000.0

        # ==========================================================
        # 兼容旧版：如果直接传入单张 ndarray
        # ==========================================================
        if isinstance(frame_bundle, np.ndarray):
            self._show_frame_on_label(
                self.label_top_main,
                frame_bundle,
            )
            return

        # ==========================================================
        # 正常双视角 frame bundle
        # ==========================================================
        mapping = {
            "top_main": self.label_top_main,
            "top_binary": self.label_top_binary,
            "top_eroded_roi": self.label_top_eroded_roi,
            "top_candidates": self.label_top_candidates,
            "top_filter_result": self.label_top_filter_result,

            "side_main": self.label_side_main,
            "side_binary": self.label_side_binary,
            "side_eroded_roi": self.label_side_eroded_roi,
            "side_candidates": self.label_side_candidates,
            "side_filtered": self.label_side_filtered,
        }

        for key, label in mapping.items():
            frame = frame_bundle.get(key, None)

            if frame is not None:
                self._show_frame_on_label(
                    label,
                    frame,
                )

        # ==========================================================
        # DEBUG: GUI 实际处理时间
        # ==========================================================
        gui_dt_ms = (
            time.perf_counter() - gui_t0
        ) * 1000.0

        # 每 30 个实际显示帧打印一次，避免 print 本身造成明显负载
        if (
            debug_seq is not None
            and debug_seq % 30 == 0
        ):
            if debug_lag_ms is None:
                lag_text = "N/A"
            else:
                lag_text = f"{debug_lag_ms:.1f}"

            print(
                "[GUI PERF] "
                f"source_seq={debug_source_seq}, "
                f"emit_seq={debug_seq}, "
                f"keep={debug_keep}/{debug_cycle}, "
                f"queue_lag={lag_text} ms, "
                f"gui_process={gui_dt_ms:.1f} ms"
            )

    def _show_frame_on_label(self, label, frame):
        """
        将单张图像按比例缩放到 label 内并居中显示（KeepAspectRatio）。

        - 支持 BGR 三通道或单通道灰度（灰度会自动转 RGB）
        - 多余区域以黑色背景填充，避免 16:9 画面被压扁
        - 缩放尺寸取自 label.width()/height()，因此改 setGeometry 后画面会自动适配
        """
        if frame is None or label is None:
            return

        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_w = max(1, label.width())
        target_h = max(1, label.height())
        src_h, src_w = rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        if not canvas.flags["C_CONTIGUOUS"]:
            canvas = np.ascontiguousarray(canvas)

        h, w, ch = canvas.shape
        bytes_per_line = ch * w
        qimg = QImage(
            canvas.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()  # copy 避免 numpy 缓冲区被回收后花屏
        label.setText("")  # 有画面时清掉占位文字
        label.setPixmap(QPixmap.fromImage(qimg))


    # ==================== 泵相关：测试请求开始 ====================
    # 功能：请求强扰动档；重复档位是否重发由 ArduinoPump 自行判断。
    def test_pump(self):
        if self.pump is None:
            print("[TEST] pump not connected")
            return
        try:
            self.pump.set_disturbance(1)
            print(f"[TEST] pump command sent, dist_flag={1}")
        except Exception as e:
            print(f"[TEST] pump error: {e}")

    # 串口检测
    # ==================== 泵相关：测试请求结束 ====================

    def refresh_serial_ports(self):
        _ports = self.devices.list_serial_ports()
        self.comboBox_port.clear()
        if len(_ports) == 0:
            self.comboBox_port.addItem('')
        else:
            for item in _ports:
                self.comboBox_port.addItem(item)
        self.refresh_arduino_ports()

    # ==================== 泵相关：refresh_arduino_ports 开始 ====================
    # 功能：枚举 Arduino 泵可用串口。
    def refresh_arduino_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.comboBox_arduino_port.clear()
        if len(ports) == 0:
            self.comboBox_arduino_port.addItem("")
        else:
            for p in ports:
                self.comboBox_arduino_port.addItem(p)
    # ==================== 泵相关：refresh_arduino_ports 结束 ====================

    def serial_open_off(self):
        global port_state
        if self._drive_close_in_progress:
            return
        str_btn = self.button_open_serial.text()
        port_name = self.comboBox_port.currentText()
        baud_rate = int(self.comboBox_baud.currentText())

        if str_btn == '关闭串口':
            if self.devices.close_drive():
                self.button_open_serial.setText('打开串口')
                port_state = False
            else:
                print("Close Uart Fail!!")

        elif str_btn == '打开串口':
            if self.devices.open_drive(port_name, baud_rate):
                self.button_open_serial.setText('关闭串口')
                port_state = True
            else:
                print("Client Fail!!")

    # ==================== 泵相关：arduino_open_off 开始 ====================
    # 功能：连接或关闭泵串口，同步句柄和界面状态。
    def arduino_open_off(self):
        btn_text = self.button_open_arduino.text()
        port_name = self.comboBox_arduino_port.currentText().strip()

        if btn_text == "关闭泵串口":
            self.devices.close_pump()
            self.pump = None
            self.arduino_port_state = False
            self.button_open_arduino.setText("打开泵串口")
            return

        # 打开
        if not port_name:
            print("Arduino 串口为空，请先选择串口")
            return

        if self.devices.open_pump(port_name, baud=115200, timeout=0.3):
            self.pump = self.devices.pump
            self.arduino_port_state = True
            self.button_open_arduino.setText("关闭泵串口")
            print(f"Arduino pump connected on {port_name}")
        else:
            self.pump = None
            self.arduino_port_state = False
            print(f"Arduino pump open failed on {port_name}")
    # ==================== 泵相关：arduino_open_off 结束 ====================


    # ========================
    #  泵控制相关方法（保持接口一致：优先用 set_disturbance）
    # ========================

    # ==================== 泵相关：pump_on 开始 ====================
    # 功能：预留强扰动请求，保留原接口。
    def pump_on(self):
        """示例：强扰动（高速开泵）"""
        if self.pump is None:
            print("泵未连接，无法启动")
            return
        try:
            self.pump.set_disturbance(1)
        except Exception as e:
            print(f"泵启动失败: {e}")
    # ==================== 泵相关：pump_on 结束 ====================

    # ==================== 泵相关：pump_off 开始 ====================
    # 功能：请求弱扰动档，是否发送 STOP 由弱档速度配置决定。
    def pump_off(self):
        """示例：弱扰动（停泵或低速，取决于 ArduinoPump 内部实现）"""
        if self.pump is None:
            return
        try:
            self.pump.set_disturbance(0)
        except Exception:
            pass
    # ==================== 泵相关：pump_off 结束 ====================

    def _start_camera_thread(self):
        """启动相机采集线程；Python Thread 只能 start 一次，故每次打开相机时重建。"""
        global camera_state
        if self.thread_camera is not None and self.thread_camera.is_alive():
            camera_state = True
            return

        self.robot_detector.reset()
        self.reset_overlay_layers()

        self.thread_camera = threading.Thread(target=self.detect, daemon=True)
        camera_state = True
        self.thread_camera.start()

    def safe_exit(self):
        """程序退出前的统一安全出口"""
        global camera_state, motion_state, video_state
        print("程序退出，执行安全关泵...")

        had_active_experiment = bool(
            motion_state
            or self._video_recording_active
            or self._motion_start_pending
            or self._experiment_stop_pending
        )

        motion_state = False
        video_state = False
        camera_state = False

        try:
            if port_state:
                self._drive_motion_stop()
        except Exception as e:
            print(f"[WARN] zero drive current failed: {e}")

        # ==================== 泵相关：退出请求开始 ====================
        # 功能：先请求弱扰动，随后由 stop_all 执行停泵及关闭。
        if self.arduino_port_state:
            try:
                self.pump_off()
            except Exception as e:
                print(f"[WARN] pump off on exit failed: {e}")
        # ==================== 泵相关：退出请求结束 ====================

        if had_active_experiment and not self._experiment_motion_log_saved:
            self._experiment_motion_log_saved = (
                self._save_current_motion_log_to_session()
            )

        try:
            if self.thread_camera is not None and self.thread_camera.is_alive():
                self.thread_camera.join(timeout=3.0)
        except Exception as e:
            print(f"[WARN] wait camera thread failed: {e}")

        try:
            self.devices.stop_all()
        except Exception as e:
            print(f"[WARN] devices.stop_all failed: {e}")

        try:
            if hasattr(self, "mode_manager"):
                self.mode_manager.stop()
        except Exception as e:
            print(f"[WARN] mode_manager.stop failed: {e}")

        self._clear_experiment_session_state()
        QtWidgets.QApplication.quit()

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "unknown"

    def _clear_experiment_session_state(self) -> None:
        self._experiment_mode_name = ""
        self._experiment_timestamp = ""
        self._experiment_session_dir = None
        self._experiment_video_path = None
        self._experiment_video_info_path = None
        self._experiment_pkl_path = None
        self._experiment_motion_log_saved = False
        self._experiment_video_saved = False
        self._experiment_stop_pending = False
        self._motion_start_pending = False
        self._video_recording_active = False

    def _prepare_experiment_session(self) -> Dict[str, str]:
        mode_name = MODE_NAME_MAP.get(self.control_mode)
        if mode_name is None:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

        mode_name = self._sanitize_path_component(mode_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir_name = f"{mode_name}_{timestamp}"
        # 保持相对路径（ASCII），避免 .resolve() 把中文工程绝对路径传给 OpenCV
        session_dir = EXPERIMENT_RECORD_ROOT / session_dir_name

        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            session_dir_name = f"{mode_name}_{timestamp}"
            session_dir = EXPERIMENT_RECORD_ROOT / session_dir_name
            session_dir.mkdir(parents=True, exist_ok=False)

        pkl_filename = MODE_PKL_FILENAME_MAP.get(mode_name)
        if not pkl_filename:
            raise ValueError(f"No PKL filename configured for mode: {mode_name}")

        video_filename = f"experiment_{mode_name}_{timestamp}.avi"
        video_info_filename = f"experiment_{mode_name}_{timestamp}_video_info.txt"
        video_path = session_dir / video_filename
        video_info_path = session_dir / video_info_filename
        pkl_path = session_dir / pkl_filename

        self._experiment_mode_name = mode_name
        self._experiment_timestamp = timestamp
        self._experiment_session_dir = session_dir
        self._experiment_video_path = video_path
        self._experiment_video_info_path = video_info_path
        self._experiment_pkl_path = pkl_path
        self._experiment_motion_log_saved = False
        self._experiment_video_saved = False
        self._experiment_stop_pending = False

        session_info = {
            "mode_name": mode_name,
            "timestamp": timestamp,
            "session_dir": str(session_dir),
            "video_path": str(video_path),
            "video_info_path": str(video_info_path),
            "pkl_path": str(pkl_path),
        }
        print(
            "[Experiment] session prepared:\n"
            f"  mode={mode_name}\n"
            f"  timestamp={timestamp}\n"
            f"  directory={session_dir}\n"
            f"  video={video_path.name}\n"
            f"  video_info={video_info_path.name}\n"
            f"  pkl={pkl_path.name}"
        )
        return session_info

    def _get_experiment_session_info(self) -> Dict[str, str]:
        if self._experiment_session_dir is None:
            return {}
        if self._experiment_video_path is None:
            return {}
        if self._experiment_video_info_path is None:
            return {}
        if self._experiment_pkl_path is None:
            return {}
        return {
            "mode_name": self._experiment_mode_name,
            "timestamp": self._experiment_timestamp,
            "session_dir": str(self._experiment_session_dir),
            "video_path": str(self._experiment_video_path),
            "video_info_path": str(self._experiment_video_info_path),
            "pkl_path": str(self._experiment_pkl_path),
        }

    def _save_current_motion_log_to_session(self) -> bool:
        mode = self.mode_manager.current_mode
        pkl_path = self._experiment_pkl_path
        if mode is None:
            print("[Experiment] cannot save PKL: current mode is None")
            return False
        if pkl_path is None:
            print("[Experiment] cannot save PKL: session PKL path is None")
            return False
        if not hasattr(mode, "save_motion_log"):
            print("[Experiment] current mode has no save_motion_log method")
            return False
        try:
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            mode.save_motion_log(pkl_path=str(pkl_path))
            if not pkl_path.is_file():
                print(
                    "[WARN] save_motion_log returned without creating PKL file: "
                    f"{pkl_path}"
                )
                return False
            if pkl_path.stat().st_size <= 0:
                print(f"[WARN] saved PKL is empty: {pkl_path}")
                return False
            print(f"[Experiment] motion log saved: {pkl_path}")
            return True
        except Exception as exc:
            print(f"[WARN] motion log save failed: {exc}")
            return False

    def _try_finalize_experiment_session(self) -> None:
        if not self._experiment_stop_pending:
            return
        if not self._experiment_video_saved:
            return
        if not self._experiment_motion_log_saved:
            print(
                "[WARN] video saved, but PKL log "
                "was not saved successfully"
            )
        self.button_motion.setText("start motion")
        self.button_motion.setEnabled(True)
        if self.comboBox_remote_mode is not None:
            self.comboBox_remote_mode.setEnabled(True)
        print(
            "[Experiment] session finalized: "
            f"{self._experiment_session_dir}"
        )
        self._experiment_stop_pending = False

    def _on_video_recording_started(self, path: str) -> None:
        global motion_state
        print(f"[Experiment] video recording started: {path}")
        self._video_recording_active = True
        if not self._motion_start_pending:
            return
        self._motion_start_pending = False
        motion_state = True
        self.button_motion.setText("stop motion")
        self.button_motion.setEnabled(True)
        if self.comboBox_remote_mode is not None:
            self.comboBox_remote_mode.setEnabled(False)
        print("[Experiment] motion_state=True after video ready")

    def _on_video_recording_failed(self, reason: str) -> None:
        global video_state, motion_state
        print(f"[Experiment] video recording failed: {reason}")
        video_state = False
        motion_state = False
        self._motion_start_pending = False
        self._video_recording_active = False

        session_dir = self._experiment_session_dir
        if session_dir is not None and session_dir.exists():
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except Exception as exc:
                print(
                    f"[WARN] failed to remove empty "
                    f"session directory: {exc}"
                )

        self._clear_experiment_session_state()
        self.button_motion.setText("start motion")
        self.button_motion.setEnabled(True)
        if self.comboBox_remote_mode is not None:
            self.comboBox_remote_mode.setEnabled(True)

    def _on_video_recording_stopped(self, path: str, statistics) -> None:
        print(f"[Experiment] video recording stopped: {path}")
        self._video_recording_active = False
        self._experiment_video_saved = True
        self._last_video_path = str(path)
        if isinstance(statistics, dict):
            print(
                "[VideoStats] "
                f"actual_write_fps={statistics.get('actual_write_fps', 0):.3f}, "
                f"written_frames={statistics.get('written_frames', 0)}, "
                f"real_duration_s={statistics.get('real_duration_s', 0):.3f}"
            )
        self._try_finalize_experiment_session()

    def camera_event(self):
        str_btn = self.button_camera.text()

        if str_btn == '打开相机':
            self._start_camera_thread()
            self.button_camera.setText('关闭相机')
        elif str_btn == '关闭相机':
            global camera_state
            camera_state = False
            self.button_camera.setText('打开相机')

    def motion_event(self):
        global i_motion, motion_state, video_state
        if not motion_state and not self._motion_start_pending:
            print("start motion")
            if not camera_state:
                print("[Experiment] camera is not open; refuse start motion")
                return
            if not port_state:
                print("[Motion] cannot start: drive serial is not open")
                return
            if self._drive_close_in_progress:
                print("[Motion] cannot start: drive serial is closing")
                return
            if self._video_recording_active or self._experiment_stop_pending:
                print("[Experiment] previous experiment still finishing; refuse start")
                return

            self.button_motion.setEnabled(False)
            if self.comboBox_remote_mode is not None:
                self.comboBox_remote_mode.setEnabled(False)

            try:
                self._prepare_experiment_session()
            except Exception as exc:
                print(f"[Experiment] failed to prepare session: {exc}")
                self._clear_experiment_session_state()
                self.button_motion.setText("start motion")
                self.button_motion.setEnabled(True)
                if self.comboBox_remote_mode is not None:
                    self.comboBox_remote_mode.setEnabled(True)
                return

            i_motion = 0
            self.reset_overlay_layers()

            mode_key = MODE_NAME_MAP.get(self.control_mode)
            if mode_key is not None and hasattr(self, "mode_manager"):
                self.mode_manager.switch_mode(mode_key)

            try:
                self.mode_manager.on_motion_start(
                    control_mode=self.control_mode,
                )
            except Exception as e:
                print(f"[WARN] mode on_motion_start failed: {e}")

            self.button_motion.setText("starting video...")
            self._motion_start_pending = True
            video_state = True
            # motion_state 等录像启动成功后再置 True
        elif motion_state or self._motion_start_pending:
            print("stop motion")

            # 禁止下一轮继续运动。
            motion_state = False
            self._motion_start_pending = False

            # 请求视觉线程在当前控制步结束后归零并关闭线圈串口。
            # UI 线程不直接访问线圈串口。
            if port_state:
                self._drive_close_requested = True
                self._drive_close_in_progress = True
                self.button_open_serial.setText("关闭中...")
                self.button_open_serial.setEnabled(False)

            # ==================== 泵相关：停止运动请求开始 ====================
            # 功能：请求弱扰动档，保留原停止运动时的泵行为。
            if self.arduino_port_state:
                try:
                    self.pump_off()
                except Exception as exc:
                    print(f"[WARN] pump off failed: {exc}")
            # ==================== 泵相关：停止运动请求结束 ====================

            # 保存 PKL 到当前统一实验目录。
            try:
                self._experiment_motion_log_saved = (
                    self._save_current_motion_log_to_session()
                )
            except Exception as exc:
                self._experiment_motion_log_saved = False
                print(f"[WARN] save motion log failed: {exc}")

            # 前面的停止操作完成后，请求视觉线程结束录像。
            video_state = False
            self._experiment_stop_pending = True
            self._experiment_video_saved = False
            self.button_motion.setText("saving video...")
            self.button_motion.setEnabled(False)
            if self.comboBox_remote_mode is not None:
                self.comboBox_remote_mode.setEnabled(False)

            # 录像从未真正启动时，视觉线程不会发 stopped 信号。
            if not self._video_recording_active:
                self._experiment_video_saved = True
                session_dir = self._experiment_session_dir
                if session_dir is not None and session_dir.exists():
                    try:
                        if not any(session_dir.iterdir()):
                            session_dir.rmdir()
                    except Exception as exc:
                        print(
                            f"[WARN] failed to remove empty "
                            f"session directory: {exc}"
                        )
                self._try_finalize_experiment_session()

            # 不等待录像 release / 串口关闭，直接返回 Qt 事件循环。
            return

    def detect(self):
        loop = VisionControlLoop(
            runtime_ctx=self.runtime_ctx,
            robot_detector=self.robot_detector,
            overlay_renderer=self.overlay_renderer,
            runtime_loop=self.runtime_loop,
            cfg=VisionControlLoopConfig(
                camera_index=Camera_index,
                fps=60.0,
                width=1920,
                height=1080,
                flip_horizontal=True,
                video_output_path="output.avi",
                display_width=640,
                display_height=360,
                # 【侧视】ROI 以图像中心对称扩展（半高/半宽，单位 px）
                side_roi_half_height=120,
                side_roi_half_width=300,
                side_roi_center_offset_x=self.cfg["side_roi_center_offset_x"],
                side_roi_center_offset_y=self.cfg["side_roi_center_offset_y"],
                side_threshold=self.cfg["side_threshold"],
                bead_diameter_mm=self.cfg["bead_diameter_mm"],
                dual_calibration_required_pairs=self.cfg["dual_calibration_required_pairs"],
                dual_calibration_minimum_inliers=self.cfg["dual_calibration_minimum_inliers"],
                dual_calibration_relative_diameter_tolerance=self.cfg[
                    "dual_calibration_relative_diameter_tolerance"
                ],
                dual_calibration_pair_max_dt_ms=self.cfg["dual_calibration_pair_max_dt_ms"],
                top_calibration_diameter_min_px=self.cfg["top_calibration_diameter_min_px"],
                top_calibration_diameter_max_px=self.cfg["top_calibration_diameter_max_px"],
                top_calibration_circularity_min=self.cfg["top_calibration_circularity_min"],
                side_calibration_diameter_min_px=self.cfg["side_calibration_diameter_min_px"],
                side_calibration_diameter_max_px=self.cfg["side_calibration_diameter_max_px"],
                side_calibration_circularity_min=self.cfg["side_calibration_circularity_min"],
                top_calibration_center_spread_max_px=self.cfg[
                    "top_calibration_center_spread_max_px"
                ],
                side_calibration_center_spread_max_px=self.cfg[
                    "side_calibration_center_spread_max_px"
                ],
            ),
            is_camera_active=self._is_camera_active,
            is_motion_active=self._is_motion_active,
            is_video_enabled=self._is_video_enabled,
            is_drive_enabled=self._is_drive_enabled,
            get_i_motion=self._get_i_motion,
            set_i_motion=self._set_i_motion,
            get_control_mode=self._get_control_mode,
            on_frame_ready=self._emit_camera_frame_bundle,
            on_scene_initialized=self.reset_overlay_layers,
            get_experiment_session_info=self._get_experiment_session_info,
            on_video_recording_started=lambda p: self.video_recording_started.emit(p),
            on_video_recording_failed=lambda r: self.video_recording_failed.emit(r),
            on_video_recording_stopped=lambda p, s: self.video_recording_stopped.emit(p, s),
            is_drive_close_requested=self._is_drive_close_requested,
            on_drive_close_finished=self._emit_drive_close_finished,
        )
        loop.run()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = WinForm()
    form.show()
    sys.exit(app.exec())
