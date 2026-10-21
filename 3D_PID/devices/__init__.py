"""
文件功能：
    硬件设备层包入口。

设计原则：
    devices 包只负责串口、线圈驱动和泵的硬件通信。
    不计算控制力，不计算电流，不参与运动模式决策。

导出符号：
    UartSerial      — 串口底层通信
    Drive           — 线圈驱动（电流 → PWM 协议帧）
    ArduinoPump     — Arduino 泵控制
    DeviceManager   — 硬件统一管理器
"""

from devices.uart_serial import UartSerial
from devices.coil_drive import Drive
from devices.arduino_pump import ArduinoPump
from devices.device_manager import DeviceManager

__all__ = [
    "UartSerial",
    "Drive",
    "ArduinoPump",
    "DeviceManager",
]
