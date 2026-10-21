"""
文件功能：
    串口通信底层封装。

设计原则：
    本文件只负责串口打开、关闭、发送和接收。
    不计算控制力，不计算电流，不参与运动模式决策。

与其他模块关系：
    被 devices/coil_drive.py 或 devices/device_manager.py 调用。
    不允许反向调用 motion_modes、core 控制算法或 main.py。

迁移说明：
    从根目录 UartSerial.py 复制而来，类名与协议帧格式保持不变。
"""

import serial
import serial.tools.list_ports
import time
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from pymodbus.client import ModbusSerialClient as ModbusClient


class UartSerial(QObject):
    """
    线圈驱动器串口通信类。

    职责：
        1. 枚举可用串口；
        2. 打开/关闭串口；
        3. 按 Modbus 风格协议发送 Enable、SetPWM、AppPWM、ReadCurrent 等命令。

    不负责：
        电流到 PWM 的物理换算（由 coil_drive.Drive 完成）。
    """

    def __init__(self, port='COM11', baud_rate=1000000):
        # 底层 pyserial 句柄，None 表示尚未打开
        self.ser = None
        # 当前串口名，例如 COM3
        self.port = port
        # 波特率，默认 1000000（与 main.py 下拉框选项一致）
        self.baud_rate = baud_rate

    def is_port_open(self):
        """
        检测串口是否已打开。

        返回：
            True  — 串口未打开，可以尝试打开；
            False — 串口已经打开。
        """
        try:
            if self.ser.is_open:
                print("串口已经打开")
                return False
            else:
                print("串口未打开")
                return True
        except Exception as e:
            print(f"检查串口时发生错误: {e}")
            return True

    def get_all_port(self):
        """
        枚举本机所有可用串口名称。

        返回：
            串口名字符串列表，例如 ['COM3', 'COM11']。
        """
        self.port_list_name = []
        port_list = list(serial.tools.list_ports.comports())
        i = 0

        if len(port_list) <= 0:
            return []
        else:
            for port in port_list:
                i = i + 1
                self.port_list_name.append(port[0])

        return self.port_list_name

    def OpenSerialPort(self):
        """按当前 self.port / self.baud_rate 打开串口。"""
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=0.2,
            write_timeout=0.2,
        )
        if self.ser.is_open:
            print('串口打开成功')
        else:
            print('串口打开失败')

    def CloseSerialPort(self):
        """关闭已打开的串口。"""
        if self.ser is None:
            return
        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception as e:
            print(f"关闭串口发生错误: {e}")
            return
        if self.ser.is_open:
            print('串口未关闭')
        else:
            print('串口已关闭')

    def try_port_open(self, port_name, baud_rate):
        """
        尝试打开指定串口。

        参数：
            port_name: 串口名；
            baud_rate: 波特率。

        返回：
            True 表示打开成功，False 表示失败。
        """
        self.port = port_name
        self.baud_rate = baud_rate
        try:
            self.OpenSerialPort()
            return True
        except Exception as e:
            print(f"打开串口发生错误: {e}")
            return False

    def Write(self, message):
        """向串口写入原始字节帧。"""
        self.ser.write(message)

    def Read(self, length):
        """从串口读取指定长度字节。"""
        return self.ser.read(length)

    def Enable(self, id, enable):
        """驱动器使能/失能（寄存器 0x00）。"""
        self.Write(self.CalTx(0x00, enable, id, 0x06))
        if 0 < id < 8:
            RX = self.Read(8)
            if RX is None or len(RX) != 8:
                return False
            if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                if RX[:6] == bytes([id, 0x06, 0x00, 0x00, 0x00, enable]):
                    print('Enable')
                    return True
            return False

    def SetPWM(self, id, pwm):
        """设置单路驱动器 PWM 寄存器（0x01）。"""
        self.Write(self.CalTx(0x01, pwm, id, 0x06))
        if 0 < id < 8:
            RX = self.Read(8)
            if RX is None or len(RX) != 8:
                return False
            if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                expected_response = bytes([id, 0x06, 0x00, 0x01, (pwm >> 8) & 0xFF, pwm & 0xFF])
                if RX[:6] == expected_response:
                    return True
            return False

    def AppPWM(self, id):
        """应用已设置的 PWM（寄存器 0x02），同步更新所有驱动通道。"""
        self.Write(self.CalTx(0x02, 0x01, id, 0x06))
        if 0 < id < 8:
            RX = self.Read(8)
            if RX is None or len(RX) != 8:
                return False
            if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                if RX[:6] == bytes([id, 0x06, 0x00, 0x02, 0x00, 0x01]):
                    return True
            return False

    def ReadCurrent(self, id):
        """读取单路驱动器电流反馈（寄存器 0x05）。"""
        if 0 < id < 8:
            self.Write(self.CalTx(0x05, 1, id, 0x03))
            RX = self.Read(7)
            if RX is None or len(RX) != 7:
                return False
            if self.Crc16_MODBUS(RX[:5]) == int.from_bytes(RX[5:], byteorder='little'):
                current_raw = (RX[3] << 8) | RX[4]
                if current_raw >= 0x8000:
                    current = current_raw - 0x10000
                else:
                    current = current_raw
                return current
            return False
        else:
            print('WrongID')
            return False

    def Crc16_MODBUS(self, datas):
        """计算 Modbus CRC16 校验码。"""
        crc = 0xFFFF
        poly = 0xA001
        for byte in datas:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
        return crc

    def CalTx(self, register_address, value, id, function_code):
        """
        按驱动器协议封装发送帧。

        参数：
            register_address: 寄存器地址；
            value: 写入值；
            id: 从站 ID；
            function_code: Modbus 功能码。
        """
        modbus_frame = [
            id,
            function_code,
            (register_address >> 8) & 0xFF,
            register_address & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF
        ]
        crc = self.Crc16_MODBUS(modbus_frame)
        modbus_frame.append(crc & 0xFF)
        modbus_frame.append((crc >> 8) & 0xFF)
        return bytes(modbus_frame)
