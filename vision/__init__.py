"""
视觉层模块包。

当前包含：
    - robot_detector：磁控微机器人二值化轮廓检测与像素→厘米坐标换算。
    - camera_source：OpenCV 相机打开、读帧与释放。
    - overlay_renderer：三维参考投影、轨迹与目标点叠加渲染。
"""

from vision.camera_source import CameraConfig, OpenCVCamera
from vision.overlay_renderer import OverlayRenderer, OverlayRendererConfig
from vision.robot_detector import DetectorConfig, RobotDetectionResult, RobotDetector

__all__ = [
    "CameraConfig",
    "OpenCVCamera",
    "OverlayRenderer",
    "OverlayRendererConfig",
    "DetectorConfig",
    "RobotDetectionResult",
    "RobotDetector",
]
