"""
文件功能：
    Arduino 泵控制串口封装。

设计原则：
    本文件只负责泵的启停、强弱扰动档位切换。
    不计算控制律，不修改强弱扰动时序（由上层调度 pump_flag 决定）。

与其他模块关系：
    被 devices/device_manager.py 或 main.py 调用。
    不允许 import motion.py 或任何控制算法模块。

迁移说明：
    从根目录 arduino_pump.py 复制而来，串口协议与强弱档位逻辑保持不变。

Arduino 固件命令约定：
    SET <dir> <speed>   — dir: 1 正转, 0 反转; speed: 0..100
    STOP
    PING
    应答：OK / ERR / PONG
"""

import time
import serial

# ==================== 泵相关：速度限制开始 ====================
# 功能：约束 SET 命令的速度百分比，保持固件允许的 0～100 范围。
def clamp(x, lo, hi):
    """将标量限制在 [lo, hi] 区间内。"""
    return lo if x < lo else hi if x > hi else x

# ==================== 泵相关：速度限制结束 ====================


class ArduinoPump:
    """
    Arduino 蠕动泵控制类。

    职责：
        1. 打开/关闭泵串口；
        2. 按 flag（0=弱扰动, 1=强扰动）切换泵速；
        3. 边沿触发，避免重复发送相同命令。

    不负责：
        pump_flag 的相位计算（由 runtime_loop / main.py 完成）。
    """

    # ==================== 泵相关：连接与档位参数开始 ====================
    # 功能：连接 Arduino、等待启动并清空输入；保留强弱速度、方向及边沿检测状态。
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.2):
        # 泵控制专用串口
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(1.5)
        self.ser.reset_input_buffer()
        # 上一次下发的扰动档位，用于边沿检测
        self._last_flag = None

        # 强扰动档位速度（百分比），与固件 SET 命令对应
        self.strong_speed = 1
        # 弱扰动档位速度；0 表示停泵
        self.weak_speed = 0

        # 泵转动方向：1=forward
        self.dir_strong = 1
        self.dir_weak = 1

    # ==================== 泵相关：连接与档位参数结束 ====================

    # ==================== 泵相关：命令与应答开始 ====================
    # 功能：按原 ASCII 行协议发送命令，并按需读取一行应答。
    def _send(self, cmd: str, read_ack: bool = True) -> str:
        """向 Arduino 发送一行 ASCII 命令并可选读取应答。"""
        self.ser.write((cmd.strip() + "\n").encode("ascii", errors="ignore"))
        if not read_ack:
            return ""
        return self.ser.readline().decode("ascii", errors="ignore").strip()

    # ==================== 泵相关：命令与应答结束 ====================

    # ==================== 泵相关：停泵与关闭开始 ====================
    # 功能：关闭串口前先调用 stop() 发送 STOP；失败时仍继续关闭串口。
    def close(self):
        """停泵并关闭串口。"""
        try:
            self.stop()
        except Exception:
            pass
        self.ser.close()

    # ==================== 泵相关：停泵与关闭结束 ====================

    # ==================== 泵相关：在线检测与控制命令开始 ====================
    # 功能：保留 PING 预备接口、STOP 停泵和 SET 方向/速度命令。
    def ping(self) -> str:
        """发送 PING 检测 Arduino 是否在线。"""
        return self._send("PING")

    def stop(self) -> str:
        """发送 STOP 停泵。"""
        return self._send("STOP")

    def set(self, direction: int, speed_percent: int) -> str:
        """
        发送 SET 命令设置泵方向与速度。

        参数：
            direction: 1 正转，0 反转；
            speed_percent: 0~100 速度百分比。
        """
        direction = 1 if int(direction) != 0 else 0
        speed_percent = clamp(int(speed_percent), 0, 100)
        return self._send(f"SET {direction} {speed_percent}")

    # ==================== 泵相关：在线检测与控制命令结束 ====================

    # ==================== 泵相关：强弱档位切换开始 ====================
    # 功能：仅在档位变化时下发；弱档在 weak_speed<=0 时 STOP，否则按弱档速度运行。
    def set_disturbance(self, flag: int) -> None:
        """
        按扰动档位切换泵状态（边沿触发）。

        参数：
            flag: 0=弱扰动, 1=强扰动。

        注意：
            强弱扰动时序由上层 i_motion 相位决定，本函数只执行硬件切换。
        """
        flag = 1 if int(flag) != 0 else 0
        if self._last_flag == flag:
            return

        if flag == 1:
            self.set(self.dir_strong, self.strong_speed)
        else:
            if self.weak_speed <= 0:
                self.stop()
            else:
                self.set(self.dir_weak, self.weak_speed)

        self._last_flag = flag
    # ==================== 泵相关：强弱档位切换结束 ====================
