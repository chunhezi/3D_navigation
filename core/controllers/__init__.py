"""Controller package public interfaces."""

# [3D_PID] BEGIN: 公开三维 P/PI 控制器。
# 这里仅做包级重导出，方便运动模式和离线测试使用稳定的公共入口；
# 控制算法与可调参数仍集中在 pid_3d_controller.py，不会改变旧控制器。
from core.controllers.pid_3d_controller import PID3DController, PID3DResult

__all__ = ["PID3DController", "PID3DResult"]
# [3D_PID] END: 三维 P/PI 控制器公共入口。
