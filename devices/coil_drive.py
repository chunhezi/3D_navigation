"""
文件功能：
    六线圈磁驱动器封装：将目标电流转换为 PWM 协议帧并通过串口下发。

设计原则：
    本文件只负责电流限幅、占空比计算和驱动器通信。
    不计算控制力，不参与运动模式决策。

与其他模块关系：
    依赖 devices/uart_serial.py 完成底层串口读写。
    被 devices/device_manager.py 或 main.py 调用。
    不允许 import motion.py 或任何控制算法模块。

迁移说明：
    从根目录 Drive.py 复制而来，协议与线圈参数保持不变。
"""

import logging
import time
import math

import numpy as np

from devices.uart_serial import UartSerial


class Drive:
    """
    六线圈磁驱动器控制类。

    职责：
        1. 将 IX1/IX2/IY1/IY2/IZ1/IZ2 六路目标电流转为 PWM；
        2. 通过 UartSerial 下发到各从站 ID（6~1）；
        3. 提供 init_power / uninit_power 安全启停。

    不负责：
        根据磁模型计算目标电流（由控制层完成）。
    """

    def __init__(self, _uart_port):
        # 串口通信对象，类型为 UartSerial
        self.uart_port = _uart_port
        # 驱动器是否已完成 init_power
        self.power_state = False

    def calculate_int16_from_duty_cycle(self, duty_cycle):
        """
        将 PWM 占空比转换为 int16 值。

        参数：
            duty_cycle: 占空比，范围 -1.0 ~ 1.0。
        """
        int16_value = int(duty_cycle * 32768)
        int16_value = np.int16(int16_value)
        return int16_value

    def calculate_pwm_and_uint16(self, target_current, supply_voltage, resistance):
        """
        根据目标电流、供电电压和线圈电阻计算 PWM 占空比及 uint16 发送值。

        参数：
            target_current: 目标电流（A），可正可负；
            supply_voltage: 供电电压（V）；
            resistance: 线圈电阻（Ω）。

        返回：
            (duty_cycle, pwm_value_uint16)
        """
        max_current = supply_voltage / resistance
        if abs(target_current) > max_current:
            print("超过电流上限")
        # 使用 clip 保留电流正负方向。
        # 不能在 abs(target_current) 超限时直接赋值为 +max_current，
        # 否则负电流超限会被错误变成正电流，导致线圈方向错误。
        target_current = float(np.clip(target_current, -max_current, max_current))
        duty_cycle = target_current / max_current
        pwm_value_int16 = int(duty_cycle * 32767)
        if pwm_value_int16 >= 0:
            pwm_value_uint16 = pwm_value_int16
        else:
            pwm_value_uint16 = pwm_value_int16 & 0xffff
        return duty_cycle, pwm_value_uint16

    def init_power(self):
        """使能驱动器并输出零电流，标记 power_state=True。"""
        self.uart_port.Enable(0, 1)
        time.sleep(0.01)
        self.set_currents(0, 0, 0, 0, 0, 0)
        print("初始化成功")
        self.power_state = True

    def uninit_power(self):
        """输出零电流并关闭驱动器，标记 power_state=False。"""
        try:
            self.set_currents(0, 0, 0, 0, 0, 0)
            self.uart_port.Enable(0, 0)
            self.power_state = False
            return True
        except Exception as e:
            print(f"关闭port发生错误: {e}")
            return False

    def close_serial(self):
        """关闭本地串口句柄（即使归零/失能失败也必须可调用）。"""
        try:
            if self.uart_port is not None:
                self.uart_port.CloseSerialPort()
        except Exception as e:
            print(f"[WARN] Drive.close_serial failed: {e}")
        self.power_state = False

    def motion_up(self):
        """UI 方向键：向上（当前为空实现，保留接口）。"""
        pass

    def motion_down(self):
        """UI 方向键：向下（当前为空实现，保留接口）。"""
        pass

    def motion_left(self):
        """UI 方向键：向左（当前为空实现，保留接口）。"""
        pass

    def motion_right(self):
        """UI 方向键：向右（当前为空实现，保留接口）。"""
        pass

    def motion_stop(self):
        """停止运动：六路电流清零。"""
        self.set_currents(0, 0, 0, 0, 0, 0)

    def set_current_single(self, target_current, supply_voltage, resistance, id, address):
        """
        设置单路驱动器 PWM。

        参数：
            target_current: 目标电流；
            supply_voltage: 供电电压；
            resistance: 线圈电阻；
            id: 从站 ID；
            address: 寄存器地址（当前固定为 1）。
        """
        desired_dutycycle, setvalue = self.calculate_pwm_and_uint16(
            target_current=target_current,
            supply_voltage=supply_voltage,
            resistance=resistance,
        )
        return self.uart_port.SetPWM(id, setvalue)

    def enable_current(self):
        """同步应用所有已设置的 PWM。"""
        return self.uart_port.AppPWM(0)

    def read_current_single(self, id):
        """读取单路驱动器电流反馈。"""
        return self.uart_port.ReadCurrent(id)

    def set_currents(self, IX1, IX2, IY1, IY2, IZ1, IZ2):
        """
        同时设置六路线圈电流并同步更新。

        参数：
            IX1, IX2, IY1, IY2, IZ1, IZ2: 六路目标电流（A）。

        注意：
            各轴供电电压与线圈电阻为实验标定常数，勿在重构中修改。
        """
        supply_voltage_x = 30
        resistance_x1 = 2.7
        resistance_x2 = 2.67
        self.set_current_single(IX1, supply_voltage_x, resistance_x1, 6, 1)
        time.sleep(0.001)
        self.set_current_single(IX2, supply_voltage_x, resistance_x2, 5, 1)
        time.sleep(0.001)
        supply_voltage_y = 15
        resistance_y1 = 1.31
        resistance_y2 = 1.275
        self.set_current_single(IY1, supply_voltage_y, resistance_y1, 4, 1)
        time.sleep(0.001)
        self.set_current_single(IY2, supply_voltage_y, resistance_y2, 3, 1)
        time.sleep(0.001)
        supply_voltage_z = 6
        resistance_z1 = 0.571
        resistance_z2 = 0.562
        self.set_current_single(IZ1, supply_voltage_z, resistance_z1, 2, 1)
        time.sleep(0.001)
        self.set_current_single(IZ2, supply_voltage_z, resistance_z2, 1, 1)
        time.sleep(0.001)
        self.enable_current()

    def read_currents(self):
        """依次读取六路电流反馈，返回 (IX1, IX2, IY1, IY2, IZ1, IZ2)。"""
        IX1 = self.read_current_single(6)
        time.sleep(0.001)
        IX2 = self.read_current_single(5)
        time.sleep(0.001)
        IY1 = self.read_current_single(4)
        time.sleep(0.001)
        IY2 = self.read_current_single(3)
        time.sleep(0.001)
        IZ1 = self.read_current_single(2)
        time.sleep(0.001)
        IZ2 = self.read_current_single(1)
        time.sleep(0.001)
        return IX1, IX2, IY1, IY2, IZ1, IZ2
