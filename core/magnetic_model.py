import math
import time
from core import magnetic_field_solver as calculate_6coils
import numpy as np

miu_0 = 4 * math.pi * 10 ** -7
default_current = 0

# [3D] 无效 z 的日志占位符（单位 cm）。
# 含义：侧视未给出有效高度时，日志里用 -10 标记“无 z”，避免与真实 z=0（落底）混淆。
# 作用：只出现在 measurement / target / error 等日志与显示字段。
# 禁止：写入 self.position、self.vel、return_matrix() 磁场位置。
INVALID_Z_LOG_VALUE_CM = -10.0

class magnetic_motion_model:
    """
    磁珠运动学/磁学模型。

    [3D] 公共状态统一为三维：
        position / vel / accel / error / current / fx : (3,1)
        gx / gx_inv : (3,3)
    坐标约定（cm）：
        x,y — 平面位置；z — 相对落底高度，z=0 为落底标定，z>0 向上。
    状态来源：
        position 由上视 xy 与侧视 z 的有效视觉测量直接更新；
        vel 由相邻有效测量的位移差和对应时间差计算（x/y 用上视时间，z 用侧视时间）。
    """

    def __init__(
        self,
        dt,
        position=None,
        vel=None,
        accel=None,
        theta=0,
        current_limit=10,
    ) -> None:
        # [3D] 默认初值改为三维；旧二维配置仍可通过 _to_col3 自动补 z=0
        if position is None:
            position = [[0.0], [0.0], [0.0]]
        if vel is None:
            vel = [[0.0], [0.0], [0.0]]
        if accel is None:
            accel = [[0.0], [0.0], [0.0]]

        # [3D] 直接测量状态（控制/磁场用，不含日志占位符 -10）
        self.position = self._to_col3(position, name="position")  # [x,y,z]^T，cm
        self.vel = self._to_col3(vel, name="vel")                # [vx,vy,vz]^T，cm/s
        self.accel = self._to_col3(accel, name="accel")          # [ax,ay,az]^T，cm/s^2
        # [3D] current：模型收到的三维力 [Fx,Fy,Fz]^T（N），不是六路线圈电流
        self.current = np.zeros((3, 1), dtype=float)
        # [3D] error：三维跟踪误差 [ex, ey, ez]^T
        self.error = np.zeros((3, 1), dtype=float)
        # [3D] valid_input：经 dead_zone 安全处理后的有效三维力
        self.valid_input = np.zeros((3, 1), dtype=float)
        # [3D] fx：与速度相关的阻尼项 f(x)，三维
        self.fx = np.zeros((3, 1), dtype=float)

        self.theta = theta
        # dead_zone 静摩擦/启动力安全阈值（保留，不作延迟补偿）
        self.min_limit = 1.5
        self.max_limit = 3
        self.dt = float(dt)

        # ----- [3D] 原始视觉测量元数据（与 position/vel 分离）-----
        # measurement_position_cm：本帧原始测量快照；无效 z 可为 -10，禁止进磁场求解
        self.measurement_position_cm = np.array(
            [[0.0], [0.0], [INVALID_Z_LOG_VALUE_CM]],
            dtype=float,
        )
        # measurement_valid_mask：[x有效, y有效, z有效]
        self.measurement_valid_mask = np.array([False, False, False], dtype=bool)
        # measurement_pair_id / timestamp：与双视 pair 对齐，便于日志回溯
        self.measurement_pair_id = None
        self.measurement_timestamp = None
        # z_measurement_valid：本帧侧视 z 是否有效（False 时 z 位置保持最后有效值、vz=0）
        self.z_measurement_valid = False

        # ----- [3D] 差分速度历史（按轴独立时间戳）-----
        self._last_valid_measurement_position_cm = np.full(3, np.nan, dtype=float)
        self._last_valid_measurement_timestamp_s = np.full(3, np.nan, dtype=float)
        self.measurement_axis_timestamp_s = np.full(3, np.nan, dtype=float)
        self.measurement_dt_s = np.full(3, np.nan, dtype=float)
        self.measurement_delta_cm = np.zeros(3, dtype=float)
        self._last_measurement_state_pair_id = None
        self.velocity_dt_min_s = 1e-4
        self.velocity_gap_reset_s = 0.5

        self.current_log = []
        # [3D] 统一状态日志（measurement / control / dynamics）
        self.state_log = []
        self.input_log = []
        self.f_des_log = []
        # dead_zone 输出日志（安全限幅后的力）；delay/smith 相关日志已删除
        self.dead_input_log = []
        self.real_current_log = []
        # ==================== 泵相关：扰动状态与恢复预留开始 ====================
        # 功能：按控制步保留泵强弱档、恢复计数和控制状态日志；当前未使用的预留不删除。
        self.dist_log = []  # [(i_motion, dist_flag)] 这个里头放的是pump_flag
        self.T_recover = 0.2  # s，可调
        self.N_recover = int(round(self.T_recover / self.dt))
        self.recover_cnt = 0
        self.prev_pump_flag = 0
        self.prev_pump_cmd = 0
        self.mode_log = []  # [(i_motion, dist_flag)] 这个里头放的是control_flag
        # ==================== 泵相关：扰动状态与恢复预留结束 ====================
        # [只读验证] return_matrix 调用计数；限频打印进入磁场的 xyz，验证后可删
        self._return_matrix_debug_count = 0

        self.resistance = 500 * 1e-6 * 970  ## 粘滞阻力系数
        self.current_limit = current_limit

        # 线圈参数
        self.coil_number_x = 297
        self.coil_radius_x = 0.236
        self.coil_number_y = 202
        self.coil_radius_y = 0.162
        self.coil_number_z = 129
        self.coil_radius_z = 0.1

        # 磁体参数
        self.Br = 1.28
        self.density_object = 7.5 * 1000  # _kg_per_m3
        self.diameter_mm = 1 / 1000  # 底面直径为 2 毫米
        self.height_mm = 1 / 1000  # 高度为 2 毫米
        self.Volume = math.pi * (self.diameter_mm / 2) ** 3 * 4 / 3
        self.density_flow = 0.978 * 1000
        self.Magnetic = self.Br / miu_0
        self.mass = self.Volume * self.density_object
        self.MVm = self.Volume * self.Magnetic

        # [3D] 三轴磁场梯度标定系数（用于构造 gx）
        self.gradient_B_x = 0.0028
        self.gradient_B_y = 0.0039
        self.gradient_B_z = 0.0067

        # [3D] 模型 ddot{p} = f(p,v) + g(p) u
        # gx：力→加速度增益对角阵 diag(gx_x, gx_y, gx_z)；gx_inv 用于力/电流相关反解
        gx_x = (self.Volume * self.Magnetic * self.gradient_B_x) / self.mass
        gx_y = (self.Volume * self.Magnetic * self.gradient_B_y) / self.mass
        gx_z = (self.Volume * self.Magnetic * self.gradient_B_z) / self.mass
        self.gx = np.diag([gx_x, gx_y, gx_z]).astype(float)
        self.gx_inv = np.linalg.inv(self.gx)


    @staticmethod
    def _to_col3(value, *, name: str) -> np.ndarray:
        """
        [3D] 模型计算用：把输入整理成有限 (3,1) 列向量。

        作用：统一 position/vel/accel/力 等进入估计与动力学的维度。
        规则：3 元 → (3,1)；2 元 → 补 z=0（旧配置兼容）；禁止非有限值。
        注意：缺 z 时补 0，绝不用 -10（-10 只给日志）。
        """
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 3:
            out = arr.reshape(3, 1)
        elif arr.size == 2:
            out = np.array([[arr[0]], [arr[1]], [0.0]], dtype=float)
        else:
            raise ValueError(f"{name} must have 2 or 3 elements, got {arr.size}")
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{name} contains non-finite value")
        return out

    @staticmethod
    def _to_optional_col3(
        value,
        *,
        name: str,
        missing_z_value: float = INVALID_Z_LOG_VALUE_CM,
    ) -> np.ndarray:
        """
        [3D] 日志用：把目标/误差等整理成 (3,1)。

        作用：二维模式只给 xy 时，第三维填 -10，表示“本模式未定义 z”。
        与 _to_col3 的区别：缺 z 补的是日志占位符，不得用于模型计算。
        """
        if value is None:
            return np.array(
                [[INVALID_Z_LOG_VALUE_CM], [INVALID_Z_LOG_VALUE_CM], [missing_z_value]],
                dtype=float,
            )
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 3:
            out = arr.reshape(3, 1).astype(float, copy=True)
        elif arr.size == 2:
            out = np.array([[arr[0]], [arr[1]], [float(missing_z_value)]], dtype=float)
        else:
            raise ValueError(f"{name} must have 2 or 3 elements, got {arr.size}")
        return out

    def _update_measured_axis(
        self,
        *,
        axis: int,
        position_cm: float,
        timestamp_s: float,
    ) -> None:
        """
        [3D] 单轴：用有效测量直接写 position，并用相邻有效测量差分写 vel。

        不滤波、不预测、不更新 accel；不使用固定 self.dt。
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        pos = float(position_cm)
        ts = float(timestamp_s)
        if not np.isfinite(pos):
            raise ValueError("position_cm must be finite")
        if not np.isfinite(ts):
            raise ValueError("timestamp_s must be finite")

        last_position = float(self._last_valid_measurement_position_cm[axis])
        last_timestamp = float(self._last_valid_measurement_timestamp_s[axis])

        delta = 0.0
        dt = float("nan")
        velocity = 0.0

        if np.isfinite(last_position) and np.isfinite(last_timestamp):
            dt = ts - last_timestamp
            delta = pos - last_position
            if (
                (not np.isfinite(dt))
                or dt <= float(self.velocity_dt_min_s)
                or dt > float(self.velocity_gap_reset_s)
            ):
                velocity = 0.0
            else:
                velocity = delta / dt

        self.position[axis, 0] = pos
        self.vel[axis, 0] = float(velocity)
        self.measurement_axis_timestamp_s[axis] = ts
        self.measurement_dt_s[axis] = dt
        self.measurement_delta_cm[axis] = float(delta)
        self._last_valid_measurement_position_cm[axis] = pos
        self._last_valid_measurement_timestamp_s[axis] = ts

    def update_state_from_measurement(
        self,
        *,
        xy_cm=None,
        z_cm=None,
        xy_valid: bool,
        z_valid: bool,
        pair_id,
        pair_timestamp_s: float,
        xy_timestamp_s: float,
        z_timestamp_s: float,
    ) -> tuple:
        """
        [3D] 原始三维视觉测量 → 测量元数据 → 直接更新 position / 差分 vel → measurement 日志。

        每个合法新 pair 最多更新一次（pair_id 单调递增保护）。
        """
        try:
            pair_id_int = int(pair_id)
        except (TypeError, ValueError):
            raise ValueError(f"pair_id must be a positive integer, got {pair_id}")
        if pair_id_int <= 0:
            raise ValueError(f"pair_id must be a positive integer, got {pair_id}")
        pair_ts = float(pair_timestamp_s)
        if not np.isfinite(pair_ts):
            raise ValueError("pair_timestamp_s must be finite")

        if (
            self._last_measurement_state_pair_id is not None
            and pair_id_int <= int(self._last_measurement_state_pair_id)
        ):
            return (
                self.position.reshape(3).copy(),
                self.vel.reshape(3).copy(),
            )

        xy_valid = bool(xy_valid)
        z_valid = bool(z_valid)

        if xy_valid:
            if xy_cm is None:
                raise ValueError("xy_valid=True requires xy_cm")
            xy = np.asarray(xy_cm, dtype=float).reshape(-1)
            if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
                raise ValueError("xy_cm must provide two finite values")
            xy_ts = float(xy_timestamp_s)
            if not np.isfinite(xy_ts):
                raise ValueError("xy_timestamp_s must be finite")
            x_cm = float(xy[0])
            y_cm = float(xy[1])
            self.measurement_position_cm[0, 0] = x_cm
            self.measurement_position_cm[1, 0] = y_cm
            self.measurement_valid_mask[0] = True
            self.measurement_valid_mask[1] = True
            self._update_measured_axis(axis=0, position_cm=x_cm, timestamp_s=xy_ts)
            self._update_measured_axis(axis=1, position_cm=y_cm, timestamp_s=xy_ts)
        else:
            self.measurement_valid_mask[0] = False
            self.measurement_valid_mask[1] = False
            self.vel[0, 0] = 0.0
            self.vel[1, 0] = 0.0

        if z_valid:
            if z_cm is None:
                raise ValueError("z_valid=True requires z_cm")
            z_val = float(np.asarray(z_cm, dtype=float).reshape(-1)[0])
            if not np.isfinite(z_val):
                raise ValueError("z_cm must be finite")
            if abs(z_val - float(INVALID_Z_LOG_VALUE_CM)) < 1e-12:
                raise ValueError("z_cm must not be INVALID_Z_LOG_VALUE_CM (-10)")
            z_ts = float(z_timestamp_s)
            if not np.isfinite(z_ts):
                raise ValueError("z_timestamp_s must be finite")
            self.measurement_position_cm[2, 0] = z_val
            self.measurement_valid_mask[2] = True
            self.z_measurement_valid = True
            self._update_measured_axis(axis=2, position_cm=z_val, timestamp_s=z_ts)
        else:
            self.measurement_position_cm[2, 0] = float(INVALID_Z_LOG_VALUE_CM)
            self.measurement_valid_mask[2] = False
            self.z_measurement_valid = False
            self.vel[2, 0] = 0.0

        self.measurement_pair_id = pair_id_int
        self.measurement_timestamp = pair_ts
        self._last_measurement_state_pair_id = pair_id_int

        self.append_state_log(
            event_type="measurement",
            control_timestamp=None,
        )
        return (
            self.position.reshape(3).copy(),
            self.vel.reshape(3).copy(),
        )

    def reset_measurement_history(self) -> None:
        """
        [3D] 重置差分速度历史（双视重标定 / 相机重初始化时调用）。

        不清零 position，避免模式/标定切换时位置突然回到原点。
        """
        self._last_valid_measurement_position_cm[:] = np.nan
        self._last_valid_measurement_timestamp_s[:] = np.nan
        self.measurement_axis_timestamp_s[:] = np.nan
        self.measurement_dt_s[:] = np.nan
        self.measurement_delta_cm[:] = 0.0
        self._last_measurement_state_pair_id = None
        self.vel = np.zeros((3, 1), dtype=float)


    def set_error_3d(self, error_xyz) -> None:
        """
        [3D] 写入完整三维跟踪误差 [ex,ey,ez]^T。

        作用：记录三维控制器的完整位置误差。
        """
        self.error = self._to_col3(error_xyz, name="error_xyz")

    def append_state_log(
        self,
        *,
        event_type: str,
        control_timestamp=None,
        target_position=None,
        error=None,
        force_cmd=None,
        coil_current=None,
        control_mode=None,
    ) -> None:
        """
        [3D] 追加一条统一状态日志到 self.state_log。

        作用：把视觉测量、直接更新后的 position/差分 vel、控制命令、动力学按 pair_id 对齐记录。
        event_type 常用：measurement / control / dynamics。
        日志规则：无效/未定义 z → measurement 用 -10；position/vel 始终为真实状态值，不用 -10。
        """
        force_out = None
        if force_cmd is not None:
            force_arr = np.asarray(force_cmd, dtype=float).reshape(-1)
            force_out = force_arr.copy()

        coil_out = None
        if coil_current is not None:
            coil_arr = np.asarray(coil_current, dtype=float).reshape(-1)
            coil_out = coil_arr.copy()

        # 每条记录均为 copy，避免后续状态改写污染历史日志
        record = {
            "event_type": str(event_type),
            "pair_id": self.measurement_pair_id,
            "measurement_timestamp": self.measurement_timestamp,
            "control_timestamp": control_timestamp,
            "z_measurement_valid": bool(self.z_measurement_valid),
            "measurement_position_cm": self.measurement_position_cm.reshape(3).copy(),
            "measurement_valid_mask": self.measurement_valid_mask.copy(),
            "position_cm": self.position.reshape(3).copy(),
            "velocity_cm_s": self.vel.reshape(3).copy(),
            "measurement_axis_timestamp_s": self.measurement_axis_timestamp_s.copy(),
            "measurement_dt_s": self.measurement_dt_s.copy(),
            "measurement_delta_cm": self.measurement_delta_cm.copy(),
            "model_acceleration_cm_s2": self.accel.reshape(3).copy(),
            "target_position_cm": self._to_optional_col3(
                target_position, name="target_position"
            ).reshape(3),
            "error_cm": self._to_optional_col3(error, name="error").reshape(3),
            "force_cmd_n": force_out,
            "coil_current_a": coil_out,
            "control_mode": control_mode,
        }
        self.state_log.append(record)

    def update_fx(self):
        # [3D] 由三维速度更新阻尼项 fx（与 vel 同形 (3,1)）
        eta = 50 * 1e-6 * 970
        self.fx = -(6 * math.pi * eta * (self.diameter_mm / 2) * self.vel) / self.mass

    def update_accel(self, current):
        # [3D] current 为三维力；accel = fx + gx @ force
        self.current = self._to_col3(current, name="current")
        self.accel = self.fx + self.gx @ self.current

    def update_status(self, current_position):
        # [3D] DEPRECATED：旧直写位置接口；正式路径为 update_state_from_measurement。
        self.position = self._to_col3(current_position, name="current_position")

    def return_position(self):
        """[3D] 返回最近有效视觉测量直接更新的三维位置 (3,1) cm。"""
        return self.position

    def return_vel(self):
        """[3D] 返回由相邻有效视觉测量差分得到的三维速度 (3,1) cm/s。"""
        return self.vel

    def return_accel(self):
        """[3D] 返回模型加速度 (3,1)。"""
        return self.accel

    def return_error(self):
        """[3D] 返回跟踪误差 (3,1)；未定义的 z 误差在计算态为 0。"""
        return self.error

    def return_current(self):
        """[3D] 返回模型三维力 (3,1) N（非六路线圈电流）。"""
        return self.current

    def dead_zone(self, input):
        """
        [3D] 力死区 / 启动力安全处理（保留）。

        作用：
            静止时合力过小则清零，避免微扰推动；
            运动中合力过小则清零，抑制噪声力；
            超出阈值时按比例缩放，写入 self.valid_input。
        说明：
            这是安全限幅，不是 Smith 延迟补偿；delay_input / smith_com 已删除。
        """
        force = self._to_col3(input, name="dead_zone_input")
        total_force = float(np.linalg.norm(force))

        if np.any(self.vel == 0):
            if total_force <= self.max_limit:
                self.valid_input = np.zeros((3, 1), dtype=float)
            else:
                adjusted_force = total_force - self.max_limit
                scale = adjusted_force / total_force
                self.valid_input = force * scale
        elif np.any(self.vel != 0):
            if total_force <= self.min_limit:
                self.valid_input = np.zeros((3, 1), dtype=float)
            else:
                adjusted_force = total_force - self.min_limit
                scale = adjusted_force / total_force
                self.valid_input = force * scale
        else:
            self.valid_input = force.copy()
        self.dead_input_log.append(self.valid_input.copy())

    def dynamic_position(self, force_3d, dt: float) -> None:
        """
        [3D] 用三维力更新模型动力学快照（含 dead_zone 安全处理）。

        流程：
            1. force 整理为 (3,1)；
            2. dead_zone 得到 valid_input；
            3. self.current = valid_input（模型三维力，非线圈电流）；
            4. update_fx + accel = (valid_input - resistance*vel)/mass；
            5. 不积分 position/vel（由视觉直接测量/差分速度提供）。
        """
        force = self._to_col3(force_3d, name="force_3d")
        dt_val = float(dt)
        if (not np.isfinite(dt_val)) or dt_val <= 0.0:
            raise ValueError(f"dt must be finite positive, got {dt}")

        self.input_log.append(force.copy())
        self.dead_zone(force)

        self.current = self.valid_input.copy()
        self.update_fx()
        self.accel = (
            self.valid_input - self.resistance * self.vel
        ) / self.mass

        # 保证形状；不执行开环积分覆盖视觉测量状态
        assert self.current.shape == (3, 1)
        assert self.vel.shape == (3, 1)
        assert self.accel.shape == (3, 1)

        self.append_state_log(
            event_type="dynamics",
            control_timestamp=time.perf_counter(),
            force_cmd=self.valid_input.reshape(3,),
        )


    def return_matrix(self):
        """
        [3D] 用最近有效视觉测量直接更新的三维位置计算六线圈磁场矩阵。

        约定：
            self.position 单位为 cm；
            position[2]==0 为落底标定位置；position[2]>0 相对落底向上。
        注意：
            只用 self.position（最后有效高度），不读 measurement_position_cm；
            禁止日志占位符 -10 进入磁场模型；
            0.01 仅为 cm→m，不增加绝对高度偏置或安全高度。
        """
        x_m = float(self.position[0, 0]) * 0.01
        y_m = float(self.position[1, 0]) * 0.01
        z_m = float(self.position[2, 0]) * 0.01
        if not np.all(np.isfinite([x_m, y_m, z_m])):
            raise ValueError("3D position contains non-finite value")

        # [只读验证] 确认磁场矩阵用的是状态 xyz（绝非 measurement 的 -10）
        self._return_matrix_debug_count += 1
        if self._return_matrix_debug_count % 100 == 0:
            print(
                "[MAGNETIC MATRIX XYZ] "
                f"pair={self.measurement_pair_id} "
                f"z_valid={bool(self.z_measurement_valid)} "
                f"state_xyz_cm=("
                f"{float(self.position[0, 0]):.3f},"
                f"{float(self.position[1, 0]):.3f},"
                f"{float(self.position[2, 0]):.3f}) "
                f"xyz_m=("
                f"{x_m:.6f},"
                f"{y_m:.6f},"
                f"{z_m:.6f})"
            )

        return calculate_6coils.integrate_magnetic_field(x_m, y_m, z_m)

    def return_position_m(self) -> np.ndarray:
        """
        [3D] 返回直接测量更新的位置（米），shape=(3,)，由 cm 换算。

        作用：给需要 SI 单位的调用方；不引入额外高度偏置。
        """
        return self.position.reshape(3).copy() * 0.01

    def return_MVm(self):
        return self.MVm

    def save_current(self, I_real):
        self.real_current_log.append(I_real)
