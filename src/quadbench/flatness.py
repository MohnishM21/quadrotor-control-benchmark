"""Differential flatness: full reference state from position and yaw.

A quadrotor is differentially flat in (x, y, z, yaw) (Mellinger & Kumar, 2011).
Given the reference acceleration, the required thrust direction fixes two
attitude degrees of freedom, and yaw fixes the third. The angular velocity
and acceleration then follow from the time derivatives of that attitude.

Body rates are computed by central differences of R(t). The trajectories are
analytic and smooth, so this is accurate to O(h^2) and avoids a long
closed-form derivation. tests/test_trajectories_flatness.py checks it against the
closed-form jerk expression for the roll and pitch rates.
"""

from dataclasses import dataclass

import numpy as np

from .params import QuadParams
from .rotations import cross, rotmat_to_euler, vee
from .trajectories import Trajectory


@dataclass
class FlatState:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    R: np.ndarray
    euler: np.ndarray       # roll, pitch, yaw
    omega: np.ndarray       # body angular velocity
    omega_dot: np.ndarray   # body angular acceleration
    thrust: float           # collective thrust (N)
    torque: np.ndarray      # body torque needed to follow omega_dot (N m)


def attitude_from_acc(acc, yaw, params: QuadParams):
    """Rotation whose body z-axis points along the required thrust vector."""
    force = params.mass * (acc + np.array([0.0, 0.0, params.gravity]))
    thrust = np.linalg.norm(force)
    b3 = force / thrust
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = cross(b3, heading)
    b2 /= np.linalg.norm(b2)
    b1 = cross(b2, b3)
    return np.column_stack([b1, b2, b3]), thrust


def _attitude(traj: Trajectory, t, params):
    ref = traj(t)
    R, _ = attitude_from_acc(ref.acc, ref.yaw, params)
    return R


def _body_rate(traj, t, params, h):
    R = _attitude(traj, t, params)
    R_dot = (_attitude(traj, t + h, params) - _attitude(traj, t - h, params)) / (2 * h)
    m = R.T @ R_dot
    return vee(0.5 * (m - m.T))


def flat_state(traj: Trajectory, t: float, params: QuadParams, h: float = 1e-3,
               with_omega_dot: bool = True) -> FlatState:
    """`with_omega_dot=False` skips the angular acceleration (and its torque) to save time."""
    ref = traj(t)
    R, thrust = attitude_from_acc(ref.acc, ref.yaw, params)
    omega = _body_rate(traj, t, params, h)
    if with_omega_dot:
        omega_dot = (_body_rate(traj, t + h, params, h) - _body_rate(traj, t - h, params, h)) / (2 * h)
    else:
        omega_dot = np.zeros(3)
    J = params.J
    torque = J @ omega_dot + cross(omega, J @ omega)
    return FlatState(
        pos=ref.pos, vel=ref.vel, acc=ref.acc, R=R, euler=rotmat_to_euler(R),
        omega=omega, omega_dot=omega_dot, thrust=thrust, torque=torque,
    )
