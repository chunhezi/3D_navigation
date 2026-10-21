"""Debug-only console output for stabilize_3d PID tuning.

This module must never participate in the controller calculation.
It only receives already-computed runtime values and prints them.
"""

from __future__ import annotations

import numpy as np

from core.controllers import pid_3d_controller as pid_cfg


# ============================================================================
# [PID3D DEBUG] BEGIN: user configuration
#
# This entire module is for PID tuning/debugging only.
#
# True  -> print PID3D tuning information
# False -> completely disable PID3D debug output
# ============================================================================
ENABLE_PID3D_DEBUG = True

# Print every N valid PID control updates.
#
# 1 = every valid control update
# 2 = every second valid control update
# 5 = every fifth valid control update
PID3D_DEBUG_PRINT_EVERY_N = 1
# ============================================================================
# [PID3D DEBUG] END: user configuration
# ============================================================================


class PID3DDebugPrinter:
    """Debug-only console printer for stabilize_3d PID tuning."""

    def __init__(self) -> None:
        self._valid_step_count = 0

    def reset(self) -> None:
        self._valid_step_count = 0

    def print_step(
        self,
        *,
        pair_id,
        dt_s,
        position_xyz,
        target_xyz,
        pid_result,
        gravity_force_xyz,
        final_force_xyz,
        current_a,
    ) -> None:
        if not ENABLE_PID3D_DEBUG:
            return

        self._valid_step_count += 1

        every_n = max(1, int(PID3D_DEBUG_PRINT_EVERY_N))
        if self._valid_step_count % every_n != 0:
            return

        position_xyz = np.asarray(position_xyz, dtype=float).reshape(3)
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        gravity_force_xyz = np.asarray(gravity_force_xyz, dtype=float).reshape(3)
        final_force_xyz = np.asarray(final_force_xyz, dtype=float).reshape(3)
        current_a = np.asarray(current_a, dtype=float).reshape(-1)

        error = np.asarray(pid_result.error_xyz, dtype=float).reshape(3)
        p_force = np.asarray(pid_result.p_force_xyz, dtype=float).reshape(3)
        i_force = np.asarray(pid_result.i_force_xyz, dtype=float).reshape(3)
        pid_force = np.asarray(pid_result.pid_force_xyz, dtype=float).reshape(3)
        integral = np.asarray(pid_result.integral_xyz, dtype=float).reshape(3)

        # Read-only consistency check: Fcmd should equal PID + FF.
        if not np.allclose(
            pid_force + gravity_force_xyz,
            final_force_xyz,
            rtol=1e-6,
            atol=1e-12,
        ):
            print(
                "[PID3D DEBUG WARNING] "
                "final_force_xyz != pid_force_xyz + gravity_force_xyz"
            )

        print()
        print("=" * 100)
        print(
            "[PID3D DEBUG] "
            f"valid_step={self._valid_step_count} "
            f"pair={pair_id} "
            f"dt={float(dt_s):.4f} s "
            f"integral_enabled={pid_cfg.ENABLE_INTEGRAL}"
        )

        print(
            "X | "
            f"target={target_xyz[0]:+.4f} cm | "
            f"pos={position_xyz[0]:+.4f} cm | "
            f"err={error[0]:+.4f} cm | "
            f"KP={pid_cfg.KP_X:.3e} | "
            f"KI={pid_cfg.KI_X:.3e} | "
            f"Int={integral[0]:+.3e} | "
            f"P={p_force[0]:+.3e} N | "
            f"I={i_force[0]:+.3e} N | "
            f"PID={pid_force[0]:+.3e} N | "
            f"FF={gravity_force_xyz[0]:+.3e} N | "
            f"Fcmd={final_force_xyz[0]:+.3e} N"
        )
        print(
            "Y | "
            f"target={target_xyz[1]:+.4f} cm | "
            f"pos={position_xyz[1]:+.4f} cm | "
            f"err={error[1]:+.4f} cm | "
            f"KP={pid_cfg.KP_Y:.3e} | "
            f"KI={pid_cfg.KI_Y:.3e} | "
            f"Int={integral[1]:+.3e} | "
            f"P={p_force[1]:+.3e} N | "
            f"I={i_force[1]:+.3e} N | "
            f"PID={pid_force[1]:+.3e} N | "
            f"FF={gravity_force_xyz[1]:+.3e} N | "
            f"Fcmd={final_force_xyz[1]:+.3e} N"
        )
        print(
            "Z | "
            f"target={target_xyz[2]:+.4f} cm | "
            f"pos={position_xyz[2]:+.4f} cm | "
            f"err={error[2]:+.4f} cm | "
            f"KP={pid_cfg.KP_Z:.3e} | "
            f"KI={pid_cfg.KI_Z:.3e} | "
            f"Int={integral[2]:+.3e} | "
            f"P={p_force[2]:+.3e} N | "
            f"I={i_force[2]:+.3e} N | "
            f"PID={pid_force[2]:+.3e} N | "
            f"FF={gravity_force_xyz[2]:+.3e} N | "
            f"Fcmd={final_force_xyz[2]:+.3e} N"
        )

        if current_a.size > 0:
            current_text = " | ".join(
                f"I{i + 1}={current_a[i]:+.4f} A"
                for i in range(min(6, current_a.size))
            )
            print(f"COIL | {current_text}")

        print("=" * 100)
