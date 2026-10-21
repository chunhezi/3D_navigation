"""Common 3D reference generator for 3D magnetic-control modes.

Coordinate convention:
    target_xyz = [x, y, z]

Position unit:
    cm

This module is intentionally independent from the existing 2D
``reference_generator.py``.

Current implementation:
    - fixed 3D reference
    - discrete step reference
    - time-parameterized helix reference
    - no arrival detection
    - no interpolation / carrot / look-ahead

Core rule:
    Geometry and time advance live only in this file.
    Motion modes only consume ``get_reference_3d(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


# ============================================================================
# [REFERENCE 3D] BEGIN: user configuration
# ============================================================================

# Available modes:
#   "fixed"
#   "step"
#   "helix"
REFERENCE_MODE_3D = "helix"


# Fixed reference, used only when REFERENCE_MODE_3D == "fixed".
FIXED_TARGET_3D_CM = np.array(
    [0.0, 0.0, 0.5],
    dtype=float,
)


# Discrete waypoint sequence.
#
# Every adjacent waypoint changes both X and Y by 1 cm.
# Z is held at 0.5 cm during the current commissioning experiment.
STEP_WAYPOINTS_3D_CM = np.array(
    [
        [0.0, 0.0, 0.5],
        [-0.2, 0, 1.0],
        [0.0, 0.2, 0.5],
        [0.2, 0.0, 1.0],
    ],
    dtype=float,
)


# Number of NEW VALID 3D measurement pairs for which one waypoint is held.
#
# Current measured interval is roughly 0.155 s / pair:
#
#     130 * 0.155 ~= 20.2 s
#
# This is intentionally pair-count based, not exact wall-clock timing.
STEP_HOLD_VALID_PAIRS = 500


# True:
#   last waypoint -> first waypoint -> continue indefinitely.
#
# False:
#   remain at the final waypoint.
STEP_LOOP = True


# ============================================================================
# [REFERENCE 3D - HELIX] BEGIN: helix user configuration
# ============================================================================

HELIX_CENTER_X_CM = 0.0
HELIX_CENTER_Y_CM = 0.0

HELIX_RADIUS_CM = 0.75

HELIX_Z_START_CM = 0.50
HELIX_Z_END_CM = 1.2

HELIX_TURNS = 3.0

# 0 rad means the initial point is:
# x = +R, y = 0, z = Z_START
HELIX_START_PHASE_RAD = 0.0

# +1.0 / -1.0 controls rotation direction.
HELIX_DIRECTION = 1.0

# Total time required for the reference to complete the entire helix.
HELIX_DURATION_S = 60.0

# Number of points only used to draw the complete reference trajectory.
# This does NOT define the actual controller update rate.
HELIX_OVERLAY_POINTS = 361

# Hold the final point after the trajectory is completed.
HELIX_HOLD_FINAL = True

# ============================================================================
# [REFERENCE 3D - HELIX] END
# ============================================================================

# ============================================================================
# [REFERENCE 3D] END: user configuration
# ============================================================================


# ============================================================================
# [REFERENCE 3D] Internal runtime state
# ============================================================================

_step_index = 0
_step_valid_pair_count = 0

_helix_elapsed_s = 0.0
_helix_valid_pair_count = 0
_helix_finished_reported = False


def _validated_path_3d(path) -> np.ndarray:
    arr = np.asarray(path, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"3D reference path must be 2-D, got shape {arr.shape}"
        )

    if arr.shape[1] != 3:
        raise ValueError(
            "3D reference path must have shape (N, 3), "
            f"got {arr.shape}"
        )

    if arr.shape[0] <= 0:
        raise ValueError("3D reference path cannot be empty")

    if not np.all(np.isfinite(arr)):
        raise ValueError("3D reference path contains NaN or Inf")

    return arr.copy()


def _validate_helix_configuration() -> None:
    values = (
        HELIX_CENTER_X_CM,
        HELIX_CENTER_Y_CM,
        HELIX_RADIUS_CM,
        HELIX_Z_START_CM,
        HELIX_Z_END_CM,
        HELIX_TURNS,
        HELIX_START_PHASE_RAD,
        HELIX_DIRECTION,
        HELIX_DURATION_S,
        HELIX_OVERLAY_POINTS,
    )
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("HELIX configuration contains NaN or Inf")

    if float(HELIX_RADIUS_CM) <= 0.0:
        raise ValueError("HELIX_RADIUS_CM must be > 0")
    if float(HELIX_Z_END_CM) <= float(HELIX_Z_START_CM):
        raise ValueError("HELIX_Z_END_CM must be > HELIX_Z_START_CM")
    if float(HELIX_TURNS) <= 0.0:
        raise ValueError("HELIX_TURNS must be > 0")
    if float(HELIX_DURATION_S) <= 0.0:
        raise ValueError("HELIX_DURATION_S must be > 0")
    if int(HELIX_OVERLAY_POINTS) < 2:
        raise ValueError("HELIX_OVERLAY_POINTS must be >= 2")
    if float(HELIX_DIRECTION) == 0.0:
        raise ValueError("HELIX_DIRECTION must be non-zero")


def _build_helix_path_3d() -> np.ndarray:
    """Build the full theoretical helix path used only for overlay / PKL."""
    _validate_helix_configuration()

    n_points = int(HELIX_OVERLAY_POINTS)
    s = np.linspace(0.0, 1.0, n_points, dtype=float)

    theta = (
        float(HELIX_START_PHASE_RAD)
        + float(HELIX_DIRECTION)
        * 2.0
        * np.pi
        * float(HELIX_TURNS)
        * s
    )

    x = (
        float(HELIX_CENTER_X_CM)
        + float(HELIX_RADIUS_CM) * np.cos(theta)
    )
    y = (
        float(HELIX_CENTER_Y_CM)
        + float(HELIX_RADIUS_CM) * np.sin(theta)
    )
    z = (
        float(HELIX_Z_START_CM)
        + (float(HELIX_Z_END_CM) - float(HELIX_Z_START_CM)) * s
    )

    path = np.column_stack((x, y, z)).astype(float)
    if path.shape != (n_points, 3):
        raise RuntimeError(
            f"Unexpected helix path shape {path.shape}, "
            f"expected ({n_points}, 3)"
        )
    if not np.all(np.isfinite(path)):
        raise ValueError("Helix path contains NaN or Inf")

    path.setflags(write=False)
    return path


# Full overlay path is built once; runtime returns the same object identity.
_HELIX_PATH_3D = _build_helix_path_3d()


def _evaluate_helix_target(
    elapsed_s: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Analytically evaluate the continuous helix target at elapsed time."""
    _validate_helix_configuration()

    duration_s = float(HELIX_DURATION_S)
    elapsed_raw = max(0.0, float(elapsed_s))

    if bool(HELIX_HOLD_FINAL):
        progress = float(np.clip(elapsed_raw / duration_s, 0.0, 1.0))
    else:
        # Still clip: wrapping would cause an instantaneous z jump 1.5→0.5.
        progress = float(np.clip(elapsed_raw / duration_s, 0.0, 1.0))

    theta = (
        float(HELIX_START_PHASE_RAD)
        + float(HELIX_DIRECTION)
        * 2.0
        * np.pi
        * float(HELIX_TURNS)
        * progress
    )

    x = float(HELIX_CENTER_X_CM) + float(HELIX_RADIUS_CM) * float(np.cos(theta))
    y = float(HELIX_CENTER_Y_CM) + float(HELIX_RADIUS_CM) * float(np.sin(theta))
    z = (
        float(HELIX_Z_START_CM)
        + (float(HELIX_Z_END_CM) - float(HELIX_Z_START_CM)) * progress
    )

    target_xyz = np.array([x, y, z], dtype=float)
    info = {
        "mode": "helix",
        "elapsed_s": float(elapsed_raw),
        "duration_s": float(duration_s),
        "progress": float(progress),
        "theta_rad": float(theta),
        "completed_turns": float(HELIX_TURNS * progress),
        "finished": bool(progress >= 1.0),
    }
    return target_xyz, info


def reset_reference_3d() -> None:
    """Reset the 3D reference sequence to its initial state."""

    global _step_index
    global _step_valid_pair_count
    global _helix_elapsed_s
    global _helix_valid_pair_count
    global _helix_finished_reported

    _step_index = 0
    _step_valid_pair_count = 0
    _helix_elapsed_s = 0.0
    _helix_valid_pair_count = 0
    _helix_finished_reported = False


def reference_parameter_snapshot_3d() -> Dict[str, Any]:
    """Return reference-generator parameters for experiment PKL snapshots."""
    return {
        "REFERENCE_MODE_3D": str(REFERENCE_MODE_3D),
        "STEP_HOLD_VALID_PAIRS": int(STEP_HOLD_VALID_PAIRS),
        "STEP_LOOP": bool(STEP_LOOP),
        "HELIX_CENTER_X_CM": float(HELIX_CENTER_X_CM),
        "HELIX_CENTER_Y_CM": float(HELIX_CENTER_Y_CM),
        "HELIX_RADIUS_CM": float(HELIX_RADIUS_CM),
        "HELIX_Z_START_CM": float(HELIX_Z_START_CM),
        "HELIX_Z_END_CM": float(HELIX_Z_END_CM),
        "HELIX_TURNS": float(HELIX_TURNS),
        "HELIX_START_PHASE_RAD": float(HELIX_START_PHASE_RAD),
        "HELIX_DIRECTION": float(HELIX_DIRECTION),
        "HELIX_DURATION_S": float(HELIX_DURATION_S),
        "HELIX_OVERLAY_POINTS": int(HELIX_OVERLAY_POINTS),
        "HELIX_HOLD_FINAL": bool(HELIX_HOLD_FINAL),
    }


def get_reference_3d(
    *,
    advance: bool,
    dt_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return the current 3D reference.

    Parameters
    ----------
    advance:
        False:
            Read the current reference only.
            Do NOT increment any counter / elapsed time.

        True:
            Exactly one NEW VALID 3D measurement pair has arrived.
            Advance step counter or helix elapsed time once.

    dt_s:
        Used only by helix mode when ``advance=True``.
        Ignored by fixed / step modes.

    Notes
    -----
    No arrival detection is used.
    No distance/error threshold is used.
    No carrot algorithm is used.
    """

    global _step_index
    global _step_valid_pair_count
    global _helix_elapsed_s
    global _helix_valid_pair_count
    global _helix_finished_reported

    mode = str(REFERENCE_MODE_3D).strip().lower()

    # ------------------------------------------------------------------------
    # Fixed reference mode
    # ------------------------------------------------------------------------
    if mode == "fixed":
        target_xyz = np.asarray(
            FIXED_TARGET_3D_CM,
            dtype=float,
        ).reshape(3).copy()

        ref_path_3d = target_xyz.reshape(1, 3).copy()

        info = {
            "mode": "fixed",
            "index": 0,
            "valid_pair_count": 0,
            "hold_valid_pairs": None,
            "loop": False,
        }

        return (
            target_xyz,
            ref_path_3d,
            info,
        )

    # ------------------------------------------------------------------------
    # Discrete step reference mode
    # ------------------------------------------------------------------------
    if mode == "step":
        path = _validated_path_3d(STEP_WAYPOINTS_3D_CM)

        if advance:
            _step_valid_pair_count += 1

            hold_pairs = max(1, int(STEP_HOLD_VALID_PAIRS))

            if _step_valid_pair_count >= hold_pairs:
                _step_valid_pair_count = 0

                old_index = int(_step_index)
                next_index = old_index + 1

                if next_index >= len(path):
                    if STEP_LOOP:
                        next_index = 0
                    else:
                        next_index = len(path) - 1

                _step_index = int(next_index)

                if _step_index != old_index:
                    new_target = path[_step_index]
                    print(
                        "[REFERENCE 3D SWITCH] "
                        f"{old_index} -> {_step_index}, "
                        f"target_xyz_cm="
                        f"({new_target[0]:+.3f}, "
                        f"{new_target[1]:+.3f}, "
                        f"{new_target[2]:+.3f})"
                    )

        # Defensive protection in case configuration changes at runtime.
        _step_index = int(
            np.clip(
                _step_index,
                0,
                len(path) - 1,
            )
        )

        target_xyz = path[_step_index].copy()

        info = {
            "mode": "step",
            "index": int(_step_index),
            "valid_pair_count": int(_step_valid_pair_count),
            "hold_valid_pairs": int(STEP_HOLD_VALID_PAIRS),
            "loop": bool(STEP_LOOP),
        }

        return (
            target_xyz,
            path.copy(),
            info,
        )

    # ------------------------------------------------------------------------
    # Continuous helix reference mode
    # ------------------------------------------------------------------------
    if mode == "helix":
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt):
                raise ValueError(f"dt_s must be finite, got {dt_s!r}")
            if dt < 0.0:
                raise ValueError(f"dt_s must be >= 0, got {dt_s!r}")
            _helix_elapsed_s += dt
            _helix_valid_pair_count += 1

        target_xyz, info = _evaluate_helix_target(_helix_elapsed_s)

        if advance and _helix_valid_pair_count % 20 == 0:
            print(
                "[REFERENCE 3D HELIX] "
                f"elapsed={info['elapsed_s']:.2f} s "
                f"progress={info['progress']:.4f} "
                f"turn={info['completed_turns']:.4f}/"
                f"{float(HELIX_TURNS):.4f} "
                f"target="
                f"({target_xyz[0]:+.3f}, "
                f"{target_xyz[1]:+.3f}, "
                f"{target_xyz[2]:+.3f}) "
                f"finished={info['finished']}"
            )

        if info["finished"] and not _helix_finished_reported:
            _helix_finished_reported = True
            print(
                "[REFERENCE 3D HELIX FINISHED] "
                f"elapsed={info['elapsed_s']:.2f} s "
                f"target="
                f"({target_xyz[0]:+.3f}, "
                f"{target_xyz[1]:+.3f}, "
                f"{target_xyz[2]:+.3f}) cm"
            )

        # Return the SAME immutable overlay path object for cache identity.
        return (
            target_xyz,
            _HELIX_PATH_3D,
            info,
        )

    raise ValueError(
        "Unsupported REFERENCE_MODE_3D: "
        f"{REFERENCE_MODE_3D!r}. "
        "Supported modes are 'fixed', 'step', and 'helix'."
    )
