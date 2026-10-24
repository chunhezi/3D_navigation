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
    - wave circle reference (two height oscillations per circle)
    - spatial figure-eight reference
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
#   "wave_circle" (起伏圆)
#   "spatial_figure8" (空间八字)
#   "rising_rounded_rectangle" (上升圆角矩形)
#   "rising_butterfly" (上升空间蝴蝶)
#   "rising_flower" (上升五瓣花)
REFERENCE_MODE_3D = "helix"


# Fixed reference, used only when REFERENCE_MODE_3D == "fixed".
FIXED_TARGET_3D_CM = np.array(
    [0.75, 0.0, 0.5],
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

# Wave circle (起伏圆): x=r*cos(theta), y=r*sin(theta),
# z=z_center+z_amplitude*sin(2*theta). Hold the endpoint after two circles.
WAVE_CIRCLE_CENTER_X_CM = 0.0
WAVE_CIRCLE_CENTER_Y_CM = 0.0
WAVE_CIRCLE_RADIUS_CM = 1.0
WAVE_CIRCLE_Z_CENTER_CM = 0.5
WAVE_CIRCLE_Z_AMPLITUDE_CM = 0.1
WAVE_CIRCLE_PERIOD_S = 40.0
WAVE_CIRCLE_TURNS = 2.0
WAVE_CIRCLE_OVERLAY_POINTS = 721

# Spatial figure eight (cm): x=Ax*sin(theta), y=Ay*sin(2*theta),
# Z rises linearly from START to END. Both camera overlays use the same XYZ.
FIGURE8_X_AMPLITUDE_CM = 0.75
FIGURE8_Y_AMPLITUDE_CM = 0.50
FIGURE8_Z_START_CM = 0.50
FIGURE8_Z_END_CM = 1.20
FIGURE8_PERIOD_S = 60.0
FIGURE8_CYCLES = 1.0
FIGURE8_OVERLAY_POINTS = 721

# Rounded XY rectangle with an ascending Z reference (centimetres).
RECT_X_HALF_CM = 0.75
RECT_Y_HALF_CM = 0.40
RECT_Z_START_CM = 0.50
RECT_Z_END_CM = 1.20
RECT_DURATION_S = 60.0
RECT_ROUNDNESS = 2.0
RECT_OVERLAY_POINTS = 721

# Ascending 2:3 Lissajous curve (space butterfly), all dimensions in cm.
BUTTERFLY_X_AMPLITUDE_CM = 0.75
BUTTERFLY_Y_AMPLITUDE_CM = 0.40
BUTTERFLY_Z_START_CM = 0.50
BUTTERFLY_Z_END_CM = 1.20
BUTTERFLY_DURATION_S = 60.0
BUTTERFLY_OVERLAY_POINTS = 721

# Smooth five-lobed polar flower, ascending in Z. Centimetres and seconds.
FLOWER_X_RADIUS_CM = 0.75
FLOWER_Y_RADIUS_CM = 0.40
FLOWER_Z_START_CM = 0.50
FLOWER_Z_END_CM = 1.20
FLOWER_DURATION_S = 75.0
FLOWER_PETALS = 5
FLOWER_PETAL_DEPTH = 0.25
FLOWER_OVERLAY_POINTS = 1001

# A smooth entry tail is enabled when the caller supplies the measured XYZ.
# These are reference speeds, not guaranteed physical bead speeds.
ENTRY_ENABLED = True
ENTRY_XY_SPEED_CM_S = 0.10
ENTRY_Z_SPEED_CM_S = 0.05
ENTRY_MIN_DURATION_S = 8.0
ENTRY_XY_TOLERANCE_CM = 0.08
ENTRY_Z_TOLERANCE_CM = 0.05
ENTRY_SETTLE_S = 1.0

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

_wave_circle_elapsed_s = 0.0
_wave_circle_valid_pair_count = 0
_wave_circle_finished_reported = False

_figure8_elapsed_s = 0.0
_rectangle_elapsed_s = 0.0
_butterfly_elapsed_s = 0.0
_flower_elapsed_s = 0.0
_entry_state = None
_reference_mode_override = None


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


def _validate_wave_circle_configuration() -> None:
    values = (
        WAVE_CIRCLE_CENTER_X_CM, WAVE_CIRCLE_CENTER_Y_CM,
        WAVE_CIRCLE_RADIUS_CM, WAVE_CIRCLE_Z_CENTER_CM,
        WAVE_CIRCLE_Z_AMPLITUDE_CM, WAVE_CIRCLE_PERIOD_S,
        WAVE_CIRCLE_TURNS, WAVE_CIRCLE_OVERLAY_POINTS,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("WAVE_CIRCLE configuration must be finite")
    if WAVE_CIRCLE_RADIUS_CM <= 0 or WAVE_CIRCLE_PERIOD_S <= 0 or WAVE_CIRCLE_TURNS <= 0:
        raise ValueError("Wave circle radius, period and turns must be positive")
    if not 0 <= WAVE_CIRCLE_Z_AMPLITUDE_CM < WAVE_CIRCLE_Z_CENTER_CM:
        raise ValueError("Wave circle amplitude must keep Z above zero")
    if int(WAVE_CIRCLE_OVERLAY_POINTS) < 2:
        raise ValueError("WAVE_CIRCLE_OVERLAY_POINTS must be >= 2")


def _wave_circle_coordinates(theta) -> np.ndarray:
    """Evaluate one target or an array of path points, in centimetres."""
    return np.stack((
        WAVE_CIRCLE_CENTER_X_CM + WAVE_CIRCLE_RADIUS_CM * np.cos(theta),
        WAVE_CIRCLE_CENTER_Y_CM + WAVE_CIRCLE_RADIUS_CM * np.sin(theta),
        WAVE_CIRCLE_Z_CENTER_CM + WAVE_CIRCLE_Z_AMPLITUDE_CM * np.sin(2.0 * theta),
    ), axis=-1)


def _build_wave_circle_path_3d() -> np.ndarray:
    _validate_wave_circle_configuration()
    theta = np.linspace(0.0, 2.0 * np.pi * WAVE_CIRCLE_TURNS,
                        int(WAVE_CIRCLE_OVERLAY_POINTS))
    path = _wave_circle_coordinates(theta)
    path.setflags(write=False)
    return path


_WAVE_CIRCLE_PATH_3D = _build_wave_circle_path_3d()


def _validate_figure8_configuration() -> None:
    values = (
        FIGURE8_X_AMPLITUDE_CM, FIGURE8_Y_AMPLITUDE_CM,
        FIGURE8_Z_START_CM, FIGURE8_Z_END_CM,
        FIGURE8_PERIOD_S, FIGURE8_CYCLES, FIGURE8_OVERLAY_POINTS,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("FIGURE8 configuration must be finite")
    if min(FIGURE8_X_AMPLITUDE_CM, FIGURE8_Y_AMPLITUDE_CM,
           FIGURE8_PERIOD_S, FIGURE8_CYCLES) <= 0:
        raise ValueError("Figure-eight amplitudes, period and cycles must be positive")
    if not 0 <= FIGURE8_Z_START_CM < FIGURE8_Z_END_CM:
        raise ValueError("Figure-eight Z end must exceed its nonnegative start")
    if int(FIGURE8_OVERLAY_POINTS) < 2:
        raise ValueError("FIGURE8_OVERLAY_POINTS must be >= 2")


def _figure8_coordinates(theta) -> np.ndarray:
    return np.stack((
        FIGURE8_X_AMPLITUDE_CM * np.sin(theta),
        FIGURE8_Y_AMPLITUDE_CM * np.sin(2.0 * theta),
        FIGURE8_Z_START_CM + (FIGURE8_Z_END_CM - FIGURE8_Z_START_CM)
        * theta / (2.0 * np.pi * FIGURE8_CYCLES),
    ), axis=-1)


def _build_figure8_path_3d() -> np.ndarray:
    _validate_figure8_configuration()
    theta = np.linspace(0.0, 2.0 * np.pi * FIGURE8_CYCLES,
                        int(FIGURE8_OVERLAY_POINTS))
    path = _figure8_coordinates(theta)
    path.setflags(write=False)
    return path


_FIGURE8_PATH_3D = _build_figure8_path_3d()


def _validate_rectangle_configuration() -> None:
    values = (RECT_X_HALF_CM, RECT_Y_HALF_CM, RECT_Z_START_CM,
              RECT_Z_END_CM, RECT_DURATION_S, RECT_ROUNDNESS, RECT_OVERLAY_POINTS)
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("Rectangle configuration must be finite")
    if min(RECT_X_HALF_CM, RECT_Y_HALF_CM, RECT_DURATION_S, RECT_ROUNDNESS) <= 0:
        raise ValueError("Rectangle dimensions, duration and roundness must be positive")
    if not 0 <= RECT_Z_START_CM < RECT_Z_END_CM or int(RECT_OVERLAY_POINTS) < 2:
        raise ValueError("Rectangle needs ascending Z bounds and at least two points")


def _rectangle_coordinates(progress) -> np.ndarray:
    theta = 2.0 * np.pi * progress
    return np.stack((
        RECT_X_HALF_CM * np.tanh(RECT_ROUNDNESS * np.cos(theta)) / np.tanh(RECT_ROUNDNESS),
        RECT_Y_HALF_CM * np.tanh(RECT_ROUNDNESS * np.sin(theta)) / np.tanh(RECT_ROUNDNESS),
        RECT_Z_START_CM + (RECT_Z_END_CM - RECT_Z_START_CM) * progress,
    ), axis=-1)


_validate_rectangle_configuration()
_RECTANGLE_PATH_3D = _rectangle_coordinates(np.linspace(0., 1., int(RECT_OVERLAY_POINTS)))
_RECTANGLE_PATH_3D.setflags(write=False)


def _validate_butterfly_configuration() -> None:
    values = (BUTTERFLY_X_AMPLITUDE_CM, BUTTERFLY_Y_AMPLITUDE_CM,
              BUTTERFLY_Z_START_CM, BUTTERFLY_Z_END_CM,
              BUTTERFLY_DURATION_S, BUTTERFLY_OVERLAY_POINTS)
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("Butterfly configuration must be finite")
    if min(BUTTERFLY_X_AMPLITUDE_CM, BUTTERFLY_Y_AMPLITUDE_CM, BUTTERFLY_DURATION_S) <= 0:
        raise ValueError("Butterfly amplitudes and duration must be positive")
    if not 0 <= BUTTERFLY_Z_START_CM < BUTTERFLY_Z_END_CM or int(BUTTERFLY_OVERLAY_POINTS) < 2:
        raise ValueError("Butterfly needs ascending Z bounds and at least two points")


def _butterfly_coordinates(progress) -> np.ndarray:
    theta = 2.0 * np.pi * progress
    return np.stack((
        BUTTERFLY_X_AMPLITUDE_CM * np.sin(2.0 * theta),
        BUTTERFLY_Y_AMPLITUDE_CM * np.sin(3.0 * theta),
        BUTTERFLY_Z_START_CM + (BUTTERFLY_Z_END_CM - BUTTERFLY_Z_START_CM) * progress,
    ), axis=-1)


_validate_butterfly_configuration()
_BUTTERFLY_PATH_3D = _butterfly_coordinates(np.linspace(0., 1., int(BUTTERFLY_OVERLAY_POINTS)))
_BUTTERFLY_PATH_3D.setflags(write=False)


def _validate_flower_configuration() -> None:
    values = (FLOWER_X_RADIUS_CM, FLOWER_Y_RADIUS_CM, FLOWER_Z_START_CM,
              FLOWER_Z_END_CM, FLOWER_DURATION_S, FLOWER_PETALS,
              FLOWER_PETAL_DEPTH, FLOWER_OVERLAY_POINTS)
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("Flower configuration must be finite")
    if min(FLOWER_X_RADIUS_CM, FLOWER_Y_RADIUS_CM, FLOWER_DURATION_S) <= 0:
        raise ValueError("Flower radii and duration must be positive")
    if not 0 <= FLOWER_Z_START_CM < FLOWER_Z_END_CM:
        raise ValueError("Flower Z must ascend from a nonnegative height")
    if not 0 < FLOWER_PETAL_DEPTH < 1 or int(FLOWER_OVERLAY_POINTS) < 2:
        raise ValueError("Flower depth must be between 0 and 1 with at least two points")
    if int(FLOWER_PETALS) != FLOWER_PETALS or FLOWER_PETALS < 2:
        raise ValueError("Flower petal count must be an integer >= 2")


def _flower_coordinates(progress) -> np.ndarray:
    theta = 2.0 * np.pi * progress
    radius = (1.0 + FLOWER_PETAL_DEPTH * np.cos(FLOWER_PETALS * theta)) / (1.0 + FLOWER_PETAL_DEPTH)
    return np.stack((
        FLOWER_X_RADIUS_CM * radius * np.cos(theta),
        FLOWER_Y_RADIUS_CM * radius * np.sin(theta),
        FLOWER_Z_START_CM + (FLOWER_Z_END_CM - FLOWER_Z_START_CM) * progress,
    ), axis=-1)


_validate_flower_configuration()
_FLOWER_PATH_3D = _flower_coordinates(np.linspace(0., 1., int(FLOWER_OVERLAY_POINTS)))
_FLOWER_PATH_3D.setflags(write=False)


def reset_reference_3d(mode_override=None) -> None:
    """Reset the 3D reference sequence to its initial state."""

    global _step_index
    global _step_valid_pair_count
    global _helix_elapsed_s
    global _helix_valid_pair_count
    global _helix_finished_reported
    global _wave_circle_elapsed_s
    global _wave_circle_valid_pair_count
    global _wave_circle_finished_reported
    global _figure8_elapsed_s
    global _rectangle_elapsed_s, _entry_state
    global _reference_mode_override
    global _butterfly_elapsed_s
    global _flower_elapsed_s

    _step_index = 0
    _step_valid_pair_count = 0
    _helix_elapsed_s = 0.0
    _helix_valid_pair_count = 0
    _helix_finished_reported = False
    _wave_circle_elapsed_s = 0.0
    _wave_circle_valid_pair_count = 0
    _wave_circle_finished_reported = False
    _figure8_elapsed_s = 0.0
    _rectangle_elapsed_s = 0.0
    _butterfly_elapsed_s = 0.0
    _flower_elapsed_s = 0.0
    _entry_state = None
    _reference_mode_override = mode_override
    if (_reference_mode_override or REFERENCE_MODE_3D) == 'rrt_navigation':
        from navigation.reference import reset_navigation
        reset_navigation()


def reference_parameter_snapshot_3d() -> Dict[str, Any]:
    """Return reference-generator parameters for experiment PKL snapshots."""
    return {
        "REFERENCE_MODE_3D": str(_reference_mode_override or REFERENCE_MODE_3D),
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
        "WAVE_CIRCLE_CENTER_X_CM": float(WAVE_CIRCLE_CENTER_X_CM),
        "WAVE_CIRCLE_CENTER_Y_CM": float(WAVE_CIRCLE_CENTER_Y_CM),
        "WAVE_CIRCLE_RADIUS_CM": float(WAVE_CIRCLE_RADIUS_CM),
        "WAVE_CIRCLE_Z_CENTER_CM": float(WAVE_CIRCLE_Z_CENTER_CM),
        "WAVE_CIRCLE_Z_AMPLITUDE_CM": float(WAVE_CIRCLE_Z_AMPLITUDE_CM),
        "WAVE_CIRCLE_PERIOD_S": float(WAVE_CIRCLE_PERIOD_S),
        "WAVE_CIRCLE_TURNS": float(WAVE_CIRCLE_TURNS),
        "WAVE_CIRCLE_OVERLAY_POINTS": int(WAVE_CIRCLE_OVERLAY_POINTS),
        "FIGURE8_X_AMPLITUDE_CM": float(FIGURE8_X_AMPLITUDE_CM),
        "FIGURE8_Y_AMPLITUDE_CM": float(FIGURE8_Y_AMPLITUDE_CM),
        "FIGURE8_Z_START_CM": float(FIGURE8_Z_START_CM),
        "FIGURE8_Z_END_CM": float(FIGURE8_Z_END_CM),
        "FIGURE8_PERIOD_S": float(FIGURE8_PERIOD_S),
        "FIGURE8_CYCLES": float(FIGURE8_CYCLES),
        "FIGURE8_OVERLAY_POINTS": int(FIGURE8_OVERLAY_POINTS),
        "RECT_X_HALF_CM": float(RECT_X_HALF_CM),
        "RECT_Y_HALF_CM": float(RECT_Y_HALF_CM),
        "RECT_Z_START_CM": float(RECT_Z_START_CM),
        "RECT_Z_END_CM": float(RECT_Z_END_CM),
        "RECT_DURATION_S": float(RECT_DURATION_S),
        "RECT_ROUNDNESS": float(RECT_ROUNDNESS),
        "RECT_OVERLAY_POINTS": int(RECT_OVERLAY_POINTS),
        "BUTTERFLY_X_AMPLITUDE_CM": float(BUTTERFLY_X_AMPLITUDE_CM),
        "BUTTERFLY_Y_AMPLITUDE_CM": float(BUTTERFLY_Y_AMPLITUDE_CM),
        "BUTTERFLY_Z_START_CM": float(BUTTERFLY_Z_START_CM),
        "BUTTERFLY_Z_END_CM": float(BUTTERFLY_Z_END_CM),
        "BUTTERFLY_DURATION_S": float(BUTTERFLY_DURATION_S),
        "BUTTERFLY_OVERLAY_POINTS": int(BUTTERFLY_OVERLAY_POINTS),
        "FLOWER_X_RADIUS_CM": float(FLOWER_X_RADIUS_CM),
        "FLOWER_Y_RADIUS_CM": float(FLOWER_Y_RADIUS_CM),
        "FLOWER_Z_START_CM": float(FLOWER_Z_START_CM),
        "FLOWER_Z_END_CM": float(FLOWER_Z_END_CM),
        "FLOWER_DURATION_S": float(FLOWER_DURATION_S),
        "FLOWER_PETALS": int(FLOWER_PETALS),
        "FLOWER_PETAL_DEPTH": float(FLOWER_PETAL_DEPTH),
        "FLOWER_OVERLAY_POINTS": int(FLOWER_OVERLAY_POINTS),
        "ENTRY_ENABLED": bool(ENTRY_ENABLED),
        "ENTRY_XY_SPEED_CM_S": float(ENTRY_XY_SPEED_CM_S),
        "ENTRY_Z_SPEED_CM_S": float(ENTRY_Z_SPEED_CM_S),
        "ENTRY_MIN_DURATION_S": float(ENTRY_MIN_DURATION_S),
        "ENTRY_XY_TOLERANCE_CM": float(ENTRY_XY_TOLERANCE_CM),
        "ENTRY_Z_TOLERANCE_CM": float(ENTRY_Z_TOLERANCE_CM),
        "ENTRY_SETTLE_S": float(ENTRY_SETTLE_S),
    }


def _get_main_reference_3d(
    *,
    advance: bool,
    dt_s: float = 0.0,
    current_position_cm=None,
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
            Advance step counter or continuous trajectory time once.

    dt_s:
        Used by helix / wave_circle / spatial_figure8 when ``advance=True``.
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
    global _wave_circle_elapsed_s
    global _wave_circle_valid_pair_count
    global _wave_circle_finished_reported
    global _figure8_elapsed_s
    global _rectangle_elapsed_s
    global _butterfly_elapsed_s
    global _flower_elapsed_s

    mode = str(_reference_mode_override or REFERENCE_MODE_3D).strip().lower()
    if mode == 'rrt_navigation':
        from navigation.reference import get_navigation_reference
        return get_navigation_reference(advance=advance, dt_s=dt_s,
                                        current_position_cm=current_position_cm)

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

    if mode == "wave_circle":
        _validate_wave_circle_configuration()
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt) or dt < 0.0:
                raise ValueError(f"dt_s must be finite and >= 0, got {dt_s!r}")
            _wave_circle_elapsed_s += dt
            _wave_circle_valid_pair_count += 1

        duration_s = float(WAVE_CIRCLE_PERIOD_S * WAVE_CIRCLE_TURNS)
        progress = float(np.clip(_wave_circle_elapsed_s / duration_s, 0.0, 1.0))
        theta = 2.0 * np.pi * WAVE_CIRCLE_TURNS * progress
        target_xyz = _wave_circle_coordinates(theta)
        info = {
            "mode": "wave_circle",
            "elapsed_s": float(_wave_circle_elapsed_s),
            "duration_s": duration_s,
            "progress": progress,
            "theta_rad": float(theta),
            "completed_turns": float(WAVE_CIRCLE_TURNS * progress),
            "finished": bool(progress >= 1.0),
        }
        if advance and _wave_circle_valid_pair_count % 20 == 0:
            print(
                "[REFERENCE 3D WAVE_CIRCLE] "
                f"elapsed={_wave_circle_elapsed_s:.2f}s progress={progress:.4f} "
                f"target_xyz_cm={tuple(np.round(target_xyz, 4))}"
            )
        if info["finished"] and not _wave_circle_finished_reported:
            _wave_circle_finished_reported = True
            print("[REFERENCE 3D WAVE_CIRCLE FINISHED] Holding final target.")
        return target_xyz, _WAVE_CIRCLE_PATH_3D, info

    if mode == "spatial_figure8":
        _validate_figure8_configuration()
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt) or dt < 0.0:
                raise ValueError(f"dt_s must be finite and >= 0, got {dt_s!r}")
            _figure8_elapsed_s += dt
        duration_s = float(FIGURE8_PERIOD_S * FIGURE8_CYCLES)
        progress = float(np.clip(_figure8_elapsed_s / duration_s, 0.0, 1.0))
        theta = 2.0 * np.pi * FIGURE8_CYCLES * progress
        target_xyz = _figure8_coordinates(theta)
        info = {
            "mode": "spatial_figure8",
            "elapsed_s": float(_figure8_elapsed_s),
            "duration_s": duration_s,
            "progress": progress,
            "theta_rad": float(theta),
            "completed_cycles": float(FIGURE8_CYCLES * progress),
            "finished": bool(progress >= 1.0),
        }
        return target_xyz, _FIGURE8_PATH_3D, info

    if mode == "rising_rounded_rectangle":
        _validate_rectangle_configuration()
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt) or dt < 0:
                raise ValueError("dt_s must be finite and nonnegative")
            _rectangle_elapsed_s += dt
        progress = float(np.clip(_rectangle_elapsed_s / RECT_DURATION_S, 0., 1.))
        return _rectangle_coordinates(progress), _RECTANGLE_PATH_3D, {
            "mode": mode, "elapsed_s": _rectangle_elapsed_s,
            "duration_s": RECT_DURATION_S, "progress": progress,
            "finished": progress >= 1.,
        }

    if mode == "rising_butterfly":
        _validate_butterfly_configuration()
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt) or dt < 0:
                raise ValueError("dt_s must be finite and nonnegative")
            _butterfly_elapsed_s += dt
        progress = float(np.clip(_butterfly_elapsed_s / BUTTERFLY_DURATION_S, 0., 1.))
        return _butterfly_coordinates(progress), _BUTTERFLY_PATH_3D, {
            "mode": mode, "elapsed_s": _butterfly_elapsed_s,
            "duration_s": BUTTERFLY_DURATION_S, "progress": progress,
            "finished": progress >= 1.,
        }

    if mode == "rising_flower":
        _validate_flower_configuration()
        if advance:
            dt = float(dt_s)
            if not np.isfinite(dt) or dt < 0:
                raise ValueError("dt_s must be finite and nonnegative")
            _flower_elapsed_s += dt
        progress = float(np.clip(_flower_elapsed_s / FLOWER_DURATION_S, 0., 1.))
        return _flower_coordinates(progress), _FLOWER_PATH_3D, {
            "mode": mode, "elapsed_s": _flower_elapsed_s,
            "duration_s": FLOWER_DURATION_S, "progress": progress,
            "finished": progress >= 1.,
        }

    raise ValueError(
        "Unsupported REFERENCE_MODE_3D: "
        f"{REFERENCE_MODE_3D!r}. "
        "Supported modes: fixed, step, helix, wave_circle, spatial_figure8, rising_rounded_rectangle, rising_butterfly, rising_flower."
    )


def get_reference_3d(*, advance: bool, dt_s: float = 0.0,
                     current_position_cm=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Shared XYZ reference including a smooth measured-position entry tail.

    The main trajectory clock stays paused until the bead has reached its
    first point and remained within the configured tolerances for one second.
    Calling without a measured position preserves the standalone reference API.
    """
    global _entry_state
    dt = float(dt_s)
    if advance and (not np.isfinite(dt) or dt < 0):
        raise ValueError("dt_s must be finite and nonnegative")
    current = None
    if current_position_cm is not None:
        current = np.asarray(current_position_cm, dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(current)):
            raise ValueError("Entry measurement must be finite")

    just_started = False
    if ENTRY_ENABLED and advance and current is not None and _entry_state is None:
        values = (ENTRY_XY_SPEED_CM_S, ENTRY_Z_SPEED_CM_S, ENTRY_MIN_DURATION_S,
                  ENTRY_XY_TOLERANCE_CM, ENTRY_Z_TOLERANCE_CM, ENTRY_SETTLE_S)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Entry speeds, tolerances and durations must be positive")
        first, path, _ = _get_main_reference_3d(advance=False)
        delta = first - current
        # Quintic smoothstep has a maximum derivative of 1.875.
        duration = max(ENTRY_MIN_DURATION_S,
                       1.875 * np.linalg.norm(delta[:2]) / ENTRY_XY_SPEED_CM_S,
                       1.875 * abs(delta[2]) / ENTRY_Z_SPEED_CM_S)
        # Render only the main shape; the entry tail is used for control only.
        _entry_state = dict(start=current, first=first.copy(), duration=duration,
                            elapsed=0., stable=0., finished=False, path=path)
        just_started = True

    entry = _entry_state if ENTRY_ENABLED else None
    if entry is not None and not entry['finished']:
        if advance and not just_started:
            entry['elapsed'] = min(entry['duration'], entry['elapsed'] + dt)
        progress = entry['elapsed'] / entry['duration']
        blend = progress**3 * (10. - 15.*progress + 6.*progress**2)
        target = entry['start'] + blend * (entry['first'] - entry['start'])
        if advance and current is not None:
            error = entry['first'] - current
            reached = (progress >= 1. and np.linalg.norm(error[:2]) <= ENTRY_XY_TOLERANCE_CM
                       and abs(error[2]) <= ENTRY_Z_TOLERANCE_CM)
            entry['stable'] = entry['stable'] + dt if reached else 0.
            entry['finished'] = entry['stable'] >= ENTRY_SETTLE_S
        return target, entry['path'], {
            'mode': str(_reference_mode_override or REFERENCE_MODE_3D), 'phase': 'entry',
            'elapsed_s': entry['elapsed'], 'duration_s': entry['duration'],
            'progress': progress, 'finished': False,
            'entry_complete': entry['finished'],
        }

    target, path, info = _get_main_reference_3d(advance=advance, dt_s=dt_s,
                                              current_position_cm=current)
    info = dict(info, phase='tracking')
    return target, entry['path'] if entry is not None else path, info
