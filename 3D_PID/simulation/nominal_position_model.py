"""Nominal SI state-space interface for future output-feedback controllers.

This is the existing simulation plant, NOT the hierarchical learning algorithm
of Yu, Wu and Liu. Its mass, drag and actuator gain are not identified hardware
parameters. Input is net incremental force about the assumed equilibrium.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionStateSpace:
    """State [px, py, pz, vx, vy, vz], output position (m), input force (N)."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    dt_s: float


def build_position_state_space(mass_kg, drag_n_s_m, dt_s):
    """Return continuous matrices and their exact zero-order-hold equivalent.

    x_dot = A*x + B*(u_incremental + disturbance); y = C*x.
    Disturbance is a physical external force in N in this nominal model.
    A controller must receive only measured position and its own input history,
    not the simulation's true disturbance or velocity.
    """
    mass, drag, dt = float(mass_kg), float(drag_n_s_m), float(dt_s)
    if not np.all(np.isfinite([mass, drag, dt])) or mass <= 0 or drag < 0 or dt <= 0:
        raise ValueError("Mass and timestep must be positive; drag must be nonnegative; all finite")
    eye = np.eye(3)
    A = np.zeros((6, 6))
    A[:3, 3:] = eye
    A[3:, 3:] = -drag / mass * eye
    B = np.vstack((np.zeros((3, 3)), eye / mass))
    C = np.hstack((eye, np.zeros((3, 3))))
    if drag == 0:
        decay, position_velocity = 1., dt
        position_force, velocity_force = dt**2 / (2 * mass), dt / mass
    else:
        q = drag * dt / mass
        one_minus_decay = -np.expm1(-q)
        decay = np.exp(-q)
        position_velocity = mass / drag * one_minus_decay
        velocity_force = one_minus_decay / drag
        # Avoid cancellation in dt - (mass/drag)*(1-exp(-q)) near zero.
        if q < 1e-4:
            position_force = dt**2 / mass * (0.5 - q / 6 + q**2 / 24 - q**3 / 120)
        else:
            position_force = (dt - position_velocity) / drag
    Ad = np.block([[eye, position_velocity * eye], [np.zeros((3, 3)), decay * eye]])
    Bd = np.vstack((position_force * eye, velocity_force * eye))
    return PositionStateSpace(A=A, B=B, C=C, Ad=Ad, Bd=Bd, dt_s=dt)
