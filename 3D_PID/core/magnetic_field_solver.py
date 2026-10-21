import sympy as sp
import numpy as np
import time

# 定义符号变量
x, y, z, theta = sp.symbols('x y z theta')

# 公共参数
mu0 = 4 * np.pi * 1e-7


# 三轴线圈的参数（直径和匝数）
R = [0.236, 0.162, 0.1]  # 半径：x轴、y轴、z轴线圈
N = [297, 202, 129]  # 匝数：x轴、y轴、z轴线圈


t3 = time.time()
# 定义单轴线圈的磁场公式
def define_magnetic_fields_x(R, N):
    C = mu0 / (4 * np.pi) * N * R

    r1 = sp.sqrt((x - R * 0.5) ** 2 + (y - R * sp.sin(theta)) ** 2 + (z - R * sp.cos(theta)) ** 2)
    r2 = sp.sqrt((x + R * 0.5) ** 2 + (y - R * sp.sin(theta)) ** 2 + (z - R * sp.cos(theta)) ** 2)

    dBx1 = -C * (R - z * sp.cos(theta) - y * sp.sin(theta)) / (r1 ** 3)
    dBy1 = -C * ((x - R * 0.5) * sp.sin(theta)) / (r1 ** 3)
    dBz1 = -C * ((x - R * 0.5) * sp.cos(theta)) / (r1 ** 3)

    dBx2 = -C * (R - z * sp.cos(theta) - y * sp.sin(theta)) / (r2 ** 3)
    dBy2 = -C * ((x + R * 0.5) * sp.sin(theta)) / (r2 ** 3)
    dBz2 = -C * ((x + R * 0.5) * sp.cos(theta)) / (r2 ** 3)

    # x1线圈产生的梯度大小
    dBx_x_1 = sp.diff(dBx1, x)
    dBy_y_1 = sp.diff(dBy1, y)
    dBz_z_1 = sp.diff(dBz1, z)

    # x2线圈产生的梯度大小
    dBx_x_2 = sp.diff(dBx2, x)
    dBy_y_2 = sp.diff(dBy2, y)
    dBz_z_2 = sp.diff(dBz2, z)

    return {
        "dBx_h": sp.lambdify([x, y, z, theta], dBx1, "numpy"),
        "dBy_h": sp.lambdify([x, y, z, theta], dBy1, "numpy"),
        "dBz_h": sp.lambdify([x, y, z, theta], dBz1, "numpy"),
        "dBx_x_h": sp.lambdify([x, y, z, theta], dBx_x_1, "numpy"),
        "dBy_y_h": sp.lambdify([x, y, z, theta], dBy_y_1, "numpy"),
        "dBz_z_h": sp.lambdify([x, y, z, theta], dBz_z_1, "numpy"),
        "dBx_m": sp.lambdify([x, y, z, theta], dBx2, "numpy"),
        "dBy_m": sp.lambdify([x, y, z, theta], dBy2, "numpy"),
        "dBz_m": sp.lambdify([x, y, z, theta], dBz2, "numpy"),
        "dBx_x_m": sp.lambdify([x, y, z, theta], dBx_x_2, "numpy"),
        "dBy_y_m": sp.lambdify([x, y, z, theta], dBy_y_2, "numpy"),
        "dBz_z_m": sp.lambdify([x, y, z, theta], dBz_z_2, "numpy"),

    }

def define_magnetic_fields_y(R, N):
    C = mu0 / (4 * np.pi) * N * R

    r1 = sp.sqrt((y - R * 0.5) ** 2 + (x - R * sp.sin(theta)) ** 2 + (z - R * sp.cos(theta)) ** 2)
    r2 = sp.sqrt((y + R * 0.5) ** 2 + (x - R * sp.sin(theta)) ** 2 + (z - R * sp.cos(theta)) ** 2)

    dBx1 = C * ((y - R * 0.5) * sp.sin(theta)) / (r1 ** 3)
    dBy1 = C * (R - z * sp.cos(theta) - x * sp.sin(theta)) / (r1 ** 3)
    dBz1 = C * ((y - R * 0.5) * sp.cos(theta)) / (r1 ** 3)

    dBx2 = C * ((y + R * 0.5) * sp.sin(theta)) / (r2 ** 3)
    dBy2 = C * (R - z * sp.cos(theta) - x * sp.sin(theta)) / (r2 ** 3)
    dBz2 = C * ((y + R * 0.5) * sp.cos(theta)) / (r2 ** 3)

    # y1线圈产生的梯度大小
    dBx_x_1 = sp.diff(dBx1, x)
    dBy_y_1 = sp.diff(dBy1, y)
    dBz_z_1 = sp.diff(dBz1, z)

    # y2线圈产生的梯度大小
    dBx_x_2 = sp.diff(dBx2, x)
    dBy_y_2 = sp.diff(dBy2, y)
    dBz_z_2 = sp.diff(dBz2, z)

    return {
        "dBx_h": sp.lambdify([x, y, z, theta], dBx1, "numpy"),
        "dBy_h": sp.lambdify([x, y, z, theta], dBy1, "numpy"),
        "dBz_h": sp.lambdify([x, y, z, theta], dBz1, "numpy"),
        "dBx_x_h": sp.lambdify([x, y, z, theta], dBx_x_1, "numpy"),
        "dBy_y_h": sp.lambdify([x, y, z, theta], dBy_y_1, "numpy"),
        "dBz_z_h": sp.lambdify([x, y, z, theta], dBz_z_1, "numpy"),
        "dBx_m": sp.lambdify([x, y, z, theta], dBx2, "numpy"),
        "dBy_m": sp.lambdify([x, y, z, theta], dBy2, "numpy"),
        "dBz_m": sp.lambdify([x, y, z, theta], dBz2, "numpy"),
        "dBx_x_m": sp.lambdify([x, y, z, theta], dBx_x_2, "numpy"),
        "dBy_y_m": sp.lambdify([x, y, z, theta], dBy_y_2, "numpy"),
        "dBz_z_m": sp.lambdify([x, y, z, theta], dBz_z_2, "numpy"),

    }

def define_magnetic_fields_z(R, N):
    C = mu0 / (4 * np.pi) * N * R

    r1 = sp.sqrt((z - R * 0.5) ** 2 + (x - R * sp.sin(theta)) ** 2 + (y - R * sp.cos(theta)) ** 2)
    r2 = sp.sqrt((z + R * 0.5) ** 2 + (x - R * sp.sin(theta)) ** 2 + (y - R * sp.cos(theta)) ** 2)

    dBx1 = C * ((z - R * 0.5) * sp.sin(theta)) / (r1 ** 3)
    dBy1 = C * ((z - R * 0.5) * sp.cos(theta)) / (r1 ** 3)
    dBz1 = C * (R - y * sp.cos(theta) - x * sp.sin(theta)) / (r1 ** 3)

    dBx2 = C * ((z + R * 0.5) * sp.sin(theta)) / (r2 ** 3)
    dBy2 = C * ((z + R * 0.5) * sp.cos(theta)) / (r2 ** 3)
    dBz2 = C * (R - y * sp.cos(theta) - x * sp.sin(theta)) / (r2 ** 3)

    # z1线圈产生的梯度大小
    dBx_x_1 = sp.diff(dBx1, x)
    dBy_y_1 = sp.diff(dBy1, y)
    dBz_z_1 = sp.diff(dBz1, z)

    # z2线圈产生的梯度大小
    dBx_x_2 = sp.diff(dBx2, x)
    dBy_y_2 = sp.diff(dBy2, y)
    dBz_z_2 = sp.diff(dBz2, z)

    return {
        "dBx_h": sp.lambdify([x, y, z, theta], dBx1, "numpy"),
        "dBy_h": sp.lambdify([x, y, z, theta], dBy1, "numpy"),
        "dBz_h": sp.lambdify([x, y, z, theta], dBz1, "numpy"),
        "dBx_x_h": sp.lambdify([x, y, z, theta], dBx_x_1, "numpy"),
        "dBy_y_h": sp.lambdify([x, y, z, theta], dBy_y_1, "numpy"),
        "dBz_z_h": sp.lambdify([x, y, z, theta], dBz_z_1, "numpy"),
        "dBx_m": sp.lambdify([x, y, z, theta], dBx2, "numpy"),
        "dBy_m": sp.lambdify([x, y, z, theta], dBy2, "numpy"),
        "dBz_m": sp.lambdify([x, y, z, theta], dBz2, "numpy"),
        "dBx_x_m": sp.lambdify([x, y, z, theta], dBx_x_2, "numpy"),
        "dBy_y_m": sp.lambdify([x, y, z, theta], dBy_y_2, "numpy"),
        "dBz_z_m": sp.lambdify([x, y, z, theta], dBz_z_2, "numpy"),

    }

# 生成三轴磁场公式
fields_x = define_magnetic_fields_x(R[0], N[0])
fields_y = define_magnetic_fields_y(R[1], N[1])
fields_z = define_magnetic_fields_z(R[2], N[2])

# 生成三轴磁场公式
# fields_x = define_magnetic_fields(R_m[0], N_m[0])
# fields_y = define_magnetic_fields(R_m[1], N_m[1])
# fields_z = define_magnetic_fields(R_m[2], N_m[2])

# # 生成三轴磁场公式
# fields_x = define_magnetic_fields(R_h[0], N_h[0])
# fields_y = define_magnetic_fields(R_h[1], N_h[1])
# fields_z = define_magnetic_fields(R_h[2], N_h[2])


theta0 = np.linspace(0, 2*np.pi, 100)
B_g = np.zeros((6, 6))
def integrate_magnetic_field(x0, y0, z0):
    idx = 0

    for i, field_func in fields_x.items():
        # 计算磁场分量的值
        field_values = field_func(x0, y0, z0, theta0)
        # 对 theta0 进行梯形积分
        B_g[idx % 6, idx // 6] = np.trapz(field_values, theta0)
        idx += 1

    for i, field_func in fields_y.items():
        # 计算磁场分量的值
        field_values = field_func(x0, y0, z0, theta0)
        # 对 theta0 进行梯形积分
        B_g[idx % 6, idx // 6] = np.trapz(field_values, theta0)
        idx += 1

    for i, field_func in fields_z.items():
        # 计算磁场分量的值
        field_values = field_func(x0, y0, z0, theta0)
        # 对 theta0 进行梯形积分
        B_g[idx % 6, idx // 6] = np.trapz(field_values, theta0)
        idx += 1
    B_g_1 = B_g[:, [0, 1, 2, 3, 4, 5]]
    #调节了输出的矩阵的表达式，在0点附近为一个
    return B_g_1
##考虑工作空间为半径为5cm，高为6cm的圆柱体
# t1 = time.perf_counter()
# B_g_1 = integrate_magnetic_field(0.00, 0.00, 0.00)
# np.set_printoptions(linewidth=1000)
# print(B_g_1)
# rank = np.linalg.matrix_rank(B_g_1)
# eig = np.linalg.eigvals(B_g_1)
# #print("矩阵的秩:", rank)
# #print("特征值:", eig)
# #print(B_g_1.shape)
# t2 = time.perf_counter()
# print(t2 - t1)
# print(t2 - t3)
