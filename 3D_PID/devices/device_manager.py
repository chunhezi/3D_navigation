"""
文件功能：
    统一管理线圈驱动和泵的创建、启停及安全关闭。

设计原则：
    DeviceManager 只负责硬件对象的创建、启动、停止和安全关闭。
    不负责控制算法，不生成目标点。

与其他模块关系：
    被 main.py 或 app/runtime_loop.py 调用。
    内部组合 uart_serial、coil_drive、arduino_pump。
    不允许反向调用 motion_modes、core 控制算法。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from devices.uart_serial import UartSerial
from devices.coil_drive import Drive
from devices.arduino_pump import ArduinoPump


class DeviceManager:
    """
    硬件设备管理器。

    主要职责：
        1. 初始化串口与线圈驱动；
        2. 按需打开 Arduino 泵；
        3. 程序退出时统一释放硬件；
        4. 紧急停止时输出零电流。

    不负责：
        根据磁模型计算线圈电流命令。
    """

    def __init__(self, cfg: Optional[Any] = None):
        """
        创建设备管理器（延迟打开硬件）。

        参数：
            cfg: 可选配置字典，可包含：
                - uart_port / uart_baud — 线圈驱动串口默认参数；
                - pump_port / pump_baud / pump_timeout — 泵串口参数。
        """
        self.cfg = cfg if cfg is not None else {}

        # 串口底层对象（线圈驱动共用）
        self.uart: Optional[UartSerial] = None
        # 六线圈驱动器
        self.drive: Optional[Drive] = None
        # ==================== 泵相关：设备对象开始 ====================
        # 功能：保存按需创建的 Arduino 泵串口封装。
        self.pump: Optional[ArduinoPump] = None
        # ==================== 泵相关：设备对象结束 ====================

        # 线圈驱动串口是否已成功 init_power
        self._drive_opened = False
        # ==================== 泵相关：连接状态开始 ====================
        # 功能：记录泵串口是否成功连接，供界面与设备状态查询使用。
        self._pump_opened = False
        # ==================== 泵相关：连接状态结束 ====================

    def _ensure_uart(self) -> UartSerial:
        """懒创建 UartSerial 实例。"""
        if self.uart is None:
            default_port = self.cfg.get("uart_port", "COM11")
            default_baud = int(self.cfg.get("uart_baud", 1000000))
            self.uart = UartSerial(port=default_port, baud_rate=default_baud)
        return self.uart

    def list_serial_ports(self):
        """枚举本机可用串口名称列表。"""
        return self._ensure_uart().get_all_port()

    def _ensure_drive(self) -> Drive:
        """懒创建 Drive 实例（需先 open_drive 才真正使能）。"""
        if self.drive is None:
            self.drive = Drive(self._ensure_uart())
        return self.drive

    def open_drive(self, port_name: str, baud_rate: int) -> bool:
        """
        打开线圈驱动相关串口并使能驱动器。

        参数：
            port_name: 串口名，例如 COM3；
            baud_rate: 波特率。

        返回：
            True 表示打开并使能成功。
        """
        uart = self._ensure_uart()
        if not uart.try_port_open(port_name, baud_rate):
            self._drive_opened = False
            return False
        drive = self._ensure_drive()
        try:
            drive.init_power()
            self._drive_opened = True
            return True
        except Exception as e:
            print(f"[DeviceManager] open_drive init_power failed: {e}")
            self._drive_opened = False
            return False

    def close_drive(self) -> bool:
        """关闭线圈驱动：零电流 + 失能 + 关闭本地串口句柄。"""
        if self.drive is None:
            self._drive_opened = False
            return True
        ok = True
        try:
            ok = bool(self.drive.uninit_power())
        except Exception as e:
            ok = False
            print(f"[WARN] drive uninit failed: {e}")
        finally:
            try:
                self.drive.close_serial()
            except Exception as e:
                ok = False
                print(f"[WARN] drive close_serial failed: {e}")
            self._drive_opened = False
        return ok

    # ==================== 泵相关：连接与关闭开始 ====================
    # 功能：创建泵串口、维护连接状态；关闭时由 ArduinoPump.close() 先发 STOP 再释放串口。
    def open_pump(self, port_name: str, baud: int = 115200, timeout: float = 0.3) -> bool:
        """
        打开泵控制串口。

        参数：
            port_name: Arduino 串口名；
            baud: 波特率，默认 115200；
            timeout: 读超时（秒）。

        返回：
            True 表示创建 ArduinoPump 成功。
        """
        if not port_name:
            return False
        try:
            self.pump = ArduinoPump(port_name, baud=baud, timeout=timeout)
            self._pump_opened = True
            return True
        except Exception as e:
            print(f"[DeviceManager] open_pump failed: {e}")
            self.pump = None
            self._pump_opened = False
            return False

    def close_pump(self) -> None:
        """关闭泵串口。"""
        if self.pump is not None:
            try:
                self.pump.close()
            except Exception as e:
                print(f"[DeviceManager] close_pump failed: {e}")
        self.pump = None
        self._pump_opened = False

    # ==================== 泵相关：连接与关闭结束 ====================

    def set_currents(self, current: Sequence[float]) -> None:
        """
        下发线圈电流。

        参数：
            current: 长度为 6 的线圈电流命令 [IX1, IX2, IY1, IY2, IZ1, IZ2]。

        注意：
            该函数只负责下发，不负责计算 current。
        """
        if self.drive is None or not self._drive_opened:
            return
        c = np.asarray(current, dtype=float).reshape(-1)
        if c.size < 6:
            c = np.pad(c, (0, 6 - c.size))
        self.drive.set_currents(
            float(c[0]), float(c[1]), float(c[2]),
            float(c[3]), float(c[4]), float(c[5]),
        )

    # ==================== 泵相关：扰动档位下发开始 ====================
    # 功能：将强弱档位交给 ArduinoPump；弱档仅在 weak_speed<=0 时发送 STOP。
    def set_pump_disturbance(self, flag: int) -> None:
        """
        按扰动档位切换泵状态。

        参数：
            flag: 0=弱扰动, 1=强扰动。
        """
        if self.pump is not None:
            self.pump.set_disturbance(flag)

    # ==================== 泵相关：扰动档位下发结束 ====================

    def stop_all(self) -> None:
        """
        安全停止所有硬件。

        执行：
            1. 输出零电流；
            2. 停止泵；
            3. 关闭泵和线圈驱动串口。
        """
        if self.drive is not None and self._drive_opened:
            try:
                self.drive.set_currents(0, 0, 0, 0, 0, 0)
            except Exception as e:
                print(f"[DeviceManager] stop_all zero currents failed: {e}")

        # ==================== 泵相关：安全停止开始 ====================
        # 功能：发送 STOP 停泵，保持原有异常处理和随后关闭串口的执行顺序。
        if self.pump is not None:
            try:
                self.pump.stop()
            except Exception as e:
                print(f"[DeviceManager] stop_all pump stop failed: {e}")
        # ==================== 泵相关：安全停止结束 ====================

        # ==================== 泵相关：安全关闭开始 ====================
        # 功能：停泵并释放其串口，不影响后续线圈驱动关闭。
        self.close_pump()
        # ==================== 泵相关：安全关闭结束 ====================
        self.close_drive()

    @property
    def is_drive_open(self) -> bool:
        """线圈驱动是否已使能。"""
        return self._drive_opened

    # ==================== 泵相关：连接查询开始 ====================
    # 功能：向调用方提供泵串口连接状态。
    @property
    def is_pump_open(self) -> bool:
        """泵是否已连接。"""
        return self._pump_opened
    # ==================== 泵相关：连接查询结束 ====================
