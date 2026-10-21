"""
文件功能：
    相机输入源封装（OpenCV VideoCapture）。

设计原则：
    1. 只负责相机打开、读帧、释放及基础采集参数设置；
    2. 不负责机器人检测、控制算法或 UI 绘制；
    3. 采集参数与 main.py 原 detect() 中保持一致，本阶段不修改算法阈值。

与其他模块关系：
    被 main.py 或未来 app/runtime_loop.py 调用。
    不 import motion.py、motion_modes 或 devices。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """
    相机采集参数（与 main.py detect() 原设置对齐）。

    注意：
        本阶段只做搬迁，不修改分辨率、帧率与镜像方向。
    """

    # 目标帧率
    fps: float = 60.0
    # 目标分辨率宽/高（像素）
    width: int = 1920
    height: int = 1080
    # 是否水平镜像（原 detect 中 cv2.flip(frame, 1)）
    flip_horizontal: bool = True


class OpenCVCamera:
    """
    OpenCV 相机封装。

    功能：
        1. 打开指定索引的 USB/内置相机；
        2. 设置 FPS 与分辨率；
        3. 按配置读帧（可选水平翻转）；
        4. 释放相机资源。

    后续若增加海康、双目或多相机，可新增 CameraSource 子类，
    而不必修改 main.py 的控制与检测逻辑。
    """

    def __init__(
        self,
        camera_index: int = 0,
        cfg: Optional[CameraConfig] = None,
    ):
        """
        参数：
            camera_index: 相机设备索引，对应 cv2.VideoCapture(index)；
            cfg: 采集配置；None 时使用 CameraConfig 默认值。
        """
        self.camera_index = int(camera_index)
        self.cfg = cfg if cfg is not None else CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None

    @property
    def is_opened(self) -> bool:
        """相机是否已成功打开。"""
        return self.cap is not None and self.cap.isOpened()

    def open(self) -> bool:
        """
        打开相机并应用 cfg 中的 FPS / 分辨率。

        返回：
            True 表示 VideoCapture 已成功打开。
        """
        self.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.is_opened:
            return False

        self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像。

        返回：
            ret: 是否读取成功；
            frame: BGR 图像；失败时为 None。

        注意：
            若 cfg.flip_horizontal 为 True，成功读帧后会做水平镜像，
            与原 main.py detect() 中 cv2.flip(cv_img, 1) 行为一致。
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret and frame is not None and self.cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def get_property(self, prop_id: int) -> float:
        """读取 OpenCV 相机属性（如 CAP_PROP_FRAME_WIDTH）。"""
        if self.cap is None:
            return 0.0
        return float(self.cap.get(prop_id))

    def log_properties(self) -> None:
        """打印当前相机实际 HEIGHT / WIDTH / FPS（与原 detect 调试输出一致）。"""
        if self.cap is None:
            print("[OpenCVCamera] cap not opened")
            return
        print("HEIGHT:", self.get_property(cv2.CAP_PROP_FRAME_HEIGHT))
        print("WIDTH:", self.get_property(cv2.CAP_PROP_FRAME_WIDTH))
        print("FPS:", round(self.get_property(cv2.CAP_PROP_FPS)))

    def release(self) -> None:
        """释放相机资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        """析构时尽量释放资源，避免相机占用未关闭。"""
        try:
            self.release()
        except Exception:
            pass
