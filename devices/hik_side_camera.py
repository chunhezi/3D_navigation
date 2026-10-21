"""
文件功能：
    海康 MVS 黑白侧视相机连续取流封装。

设计原则：
    1. 只负责 SDK 枚举、打开、取流、释放，不做识别与 UI；
    2. grab_gray() 在释放 SDK buffer 前必须 copy 灰度数据；
    3. 不写入 magnetic_model，不接入控制循环。

与其他模块关系：
    被 app/vision_control_loop.py 调用。
    不 import motion_modes、controllers 或 RuntimeLoop。
"""

from __future__ import annotations

import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

DEFAULT_MVS_RUNTIME_DIR = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"

_sdk_loaded = False
_MvCamera: Any = None
_MV_CC_DEVICE_INFO_LIST: Any = None
_MV_CC_DEVICE_INFO: Any = None
_MV_FRAME_OUT: Any = None
_MV_USB_DEVICE: Any = None
_MV_ACCESS_Exclusive: Any = None
_PixelType_Gvsp_Mono8: Any = None


def _resolve_mv_import_dir() -> str:
    """在工程目录及上级目录中查找 MvImport。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "MvImport"),
        os.path.join(here, "..", "..", "MvImport"),
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.isdir(normalized):
            return normalized
    raise RuntimeError(
        "未找到 MvImport 目录，请将海康 MVS Python SDK 放在工程或上级目录中。"
    )


def _ensure_mvs_sdk(runtime_dir: Optional[str] = None) -> None:
    """懒加载 MVS SDK（仅首次调用时执行）。"""
    global _sdk_loaded
    global _MvCamera
    global _MV_CC_DEVICE_INFO_LIST
    global _MV_CC_DEVICE_INFO
    global _MV_FRAME_OUT
    global _MV_USB_DEVICE
    global _MV_ACCESS_Exclusive
    global _PixelType_Gvsp_Mono8

    if _sdk_loaded:
        return

    mv_import_dir = _resolve_mv_import_dir()
    if mv_import_dir not in sys.path:
        sys.path.insert(0, mv_import_dir)

    mvs_runtime = runtime_dir or DEFAULT_MVS_RUNTIME_DIR
    if os.path.isdir(mvs_runtime):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(mvs_runtime)
        os.environ["PATH"] = mvs_runtime + os.pathsep + os.environ.get("PATH", "")
    else:
        raise RuntimeError(f"MVS 运行库目录不存在: {mvs_runtime}")

    from MvCameraControl_class import MvCamera  # noqa: WPS433
    from CameraParams_const import MV_ACCESS_Exclusive, MV_USB_DEVICE  # noqa: WPS433
    from CameraParams_header import MV_CC_DEVICE_INFO, MV_CC_DEVICE_INFO_LIST, MV_FRAME_OUT  # noqa: WPS433
    from PixelType_header import PixelType_Gvsp_Mono8  # noqa: WPS433

    _MvCamera = MvCamera
    _MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
    _MV_CC_DEVICE_INFO = MV_CC_DEVICE_INFO
    _MV_FRAME_OUT = MV_FRAME_OUT
    _MV_USB_DEVICE = MV_USB_DEVICE
    _MV_ACCESS_Exclusive = MV_ACCESS_Exclusive
    _PixelType_Gvsp_Mono8 = PixelType_Gvsp_Mono8
    _sdk_loaded = True


@dataclass
class SideFrame:
    gray: np.ndarray
    timestamp: float
    frame_number: Optional[int] = None


class HikSideCamera:
    """
    海康 USB 黑白相机连续取流。

    生命周期：
        open() → grab_gray() 循环 → close()
    """

    def __init__(self, index: int = 0, runtime_dir: Optional[str] = None):
        self.index = int(index)
        self.runtime_dir = runtime_dir
        self._camera: Any = None
        self._frame_out: Any = None
        self._handle_created = False
        self._device_opened = False
        self._grabbing = False
        self._last_error = ""

    @property
    def is_opened(self) -> bool:
        return bool(self._device_opened and self._grabbing)

    @property
    def last_error(self) -> str:
        return self._last_error

    def open(self) -> bool:
        """枚举 USB 设备、独占打开并开始取流。"""
        self.close()
        try:
            _ensure_mvs_sdk(self.runtime_dir)
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[HikSideCamera] SDK 加载失败: {exc}")
            return False

        devices = _MV_CC_DEVICE_INFO_LIST()
        ret = _MvCamera.MV_CC_EnumDevices(_MV_USB_DEVICE, devices)
        if ret != 0 or devices.nDeviceNum == 0:
            self._last_error = "未找到海康 USB 相机"
            print(f"[HikSideCamera] {self._last_error} (ret={ret})")
            return False

        if not 0 <= self.index < devices.nDeviceNum:
            self._last_error = (
                f"相机索引无效: {self.index}，可用范围 0-{devices.nDeviceNum - 1}"
            )
            print(f"[HikSideCamera] {self._last_error}")
            return False

        info = cast(devices.pDeviceInfo[self.index], POINTER(_MV_CC_DEVICE_INFO)).contents
        self._camera = _MvCamera()
        self._frame_out = _MV_FRAME_OUT()
        memset(byref(self._frame_out), 0, sizeof(self._frame_out))

        try:
            if self._camera.MV_CC_CreateHandle(info) != 0:
                raise RuntimeError("创建相机句柄失败")
            self._handle_created = True

            if self._camera.MV_CC_OpenDevice(_MV_ACCESS_Exclusive, 0) != 0:
                raise RuntimeError("打开相机失败，请关闭 MVS 或其他占用相机的程序")
            self._device_opened = True

            self._try_set_mono8()

            if self._camera.MV_CC_StartGrabbing() != 0:
                raise RuntimeError("相机开始取流失败")
            self._grabbing = True

            self._last_error = ""
            print(f"[HikSideCamera] 侧视相机已打开 (index={self.index})")
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[HikSideCamera] 打开失败: {exc}")
            self.close()
            return False

    def _try_set_mono8(self) -> None:
        """尽量将 PixelFormat 设为 Mono8；失败时依赖后续帧长度校验。"""
        assert self._camera is not None
        ret = self._camera.MV_CC_SetEnumValueByString("PixelFormat", "Mono8")
        if ret == 0:
            return
        ret = self._camera.MV_CC_SetEnumValue("PixelFormat", _PixelType_Gvsp_Mono8)
        if ret != 0:
            print(
                "[HikSideCamera] 未能显式设置 PixelFormat=Mono8，"
                "将依赖帧长度校验 (nFrameLen == width * height)"
            )

    @staticmethod
    def _frame_dimensions(info) -> tuple[int, int]:
        width = int(info.nExtendWidth) if int(info.nExtendWidth) > 0 else int(info.nWidth)
        height = int(info.nExtendHeight) if int(info.nExtendHeight) > 0 else int(info.nHeight)
        return width, height

    @staticmethod
    def _frame_length(info) -> int:
        frame_len_ex = int(info.nFrameLenEx)
        if frame_len_ex > 0:
            return frame_len_ex
        return int(info.nFrameLen)

    def _buffer_to_gray(self, frame_out) -> np.ndarray:
        info = frame_out.stFrameInfo
        width, height = self._frame_dimensions(info)
        frame_len = self._frame_length(info)
        expected_len = width * height

        if frame_len != expected_len:
            raise ValueError(
                f"Mono8 帧长度不匹配: nFrameLen={frame_len}, "
                f"期望 width*height={expected_len} ({width}x{height})"
            )

        if info.enPixelType != _PixelType_Gvsp_Mono8:
            print(
                f"[HikSideCamera] 警告: enPixelType={int(info.enPixelType)} "
                f"非 Mono8，但帧长度符合 Mono8 布局"
            )

        if frame_out.pBufAddr is None:
            raise ValueError("SDK 返回空图像 buffer")

        data = cast(frame_out.pBufAddr, POINTER(c_ubyte * frame_len))
        gray = np.frombuffer(data.contents, dtype=np.uint8).reshape(height, width)
        return np.ascontiguousarray(gray.copy())

    def grab_gray(self, timeout_ms: int = 50) -> Optional[SideFrame]:
        """
        取一帧 Mono8 灰度图。

        成功返回 SideFrame；超时或失败返回 None。
        """
        if not self.is_opened or self._camera is None or self._frame_out is None:
            return None

        ret = self._camera.MV_CC_GetImageBuffer(self._frame_out, int(timeout_ms))
        if ret != 0:
            return None

        try:
            gray = self._buffer_to_gray(self._frame_out)
            info = self._frame_out.stFrameInfo
            frame_number = int(info.nFrameNum) if int(info.nFrameNum) >= 0 else None
            return SideFrame(
                gray=gray,
                timestamp=time.perf_counter(),
                frame_number=frame_number,
            )
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[HikSideCamera] grab_gray 失败: {exc}")
            return None
        finally:
            self._camera.MV_CC_FreeImageBuffer(self._frame_out)

    def close(self) -> None:
        """释放相机资源；可重复调用。"""
        camera = self._camera
        if camera is not None:
            if self._grabbing:
                try:
                    camera.MV_CC_StopGrabbing()
                except Exception as exc:
                    print(f"[HikSideCamera] StopGrabbing 异常: {exc}")
                self._grabbing = False

            if self._device_opened:
                try:
                    camera.MV_CC_CloseDevice()
                except Exception as exc:
                    print(f"[HikSideCamera] CloseDevice 异常: {exc}")
                self._device_opened = False

            if self._handle_created:
                try:
                    camera.MV_CC_DestroyHandle()
                except Exception as exc:
                    print(f"[HikSideCamera] DestroyHandle 异常: {exc}")
                self._handle_created = False

        self._camera = None
        self._frame_out = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
