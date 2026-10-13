"""Rigid-body quadrotor dynamics, integrated with fixed-step RK4.

State layout (17,):
    p      [0:3]    position in world frame (m)
    v      [3:6]    velocity in world frame (m/s)
    q      [6:10]   attitude quaternion, scalar first, body -> world
    omega  [10:13]  body angular velocity (rad/s)
    f      [13:17]  actual rotor thrusts (N); differs from the command when motors lag

Equations of motion (z-up world frame):
    p_dot     = v
    v_dot     = -g e3 + (R e3 T + F_ext - c_d v) / m
    q_dot     = 0.5 q * [0, omega]
    omega_dot = J^-1 (tau - omega x J omega)
    f_dot     = (f_cmd - f) / tau_motor
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .params import QuadParams
from .rotations import cross, euler_to_quat, quat_multiply, quat_to_rotmat

P, V, Q, W, F = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 17)
STATE_DIM = 17


@dataclass
class PlantConfig:
    """Effects present in the simulated plant but not in the controllers' models."""
    motor_tau: float = 0.0               # s; 0 means rotors reach the commanded thrust instantly
    linear_drag: float = 0.0             # N per m/s, isotropic
    wind_force: Optional[Callable[[float], np.ndarray]] = None  # world-frame force (N) vs time


def make_state(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0),
               omega=(0.0, 0.0, 0.0), params: Optional[QuadParams] = None):
    x = np.zeros(STATE_DIM)
    x[P] = position
    x[V] = velocity
    x[Q] = euler_to_quat(*euler)
    x[W] = omega
    if params is not None:
        x[F] = params.hover_thrust / 4.0
    return x


class Quadrotor:
    def __init__(self, params: QuadParams, config: Optional[PlantConfig] = None):
        self.params = params
        self.config = config or PlantConfig()
        self._J = params.J
        self._J_inv = np.linalg.inv(self._J)

    def saturate(self, f_cmd):
        return np.clip(f_cmd, 0.0, self.params.max_rotor_thrust)

    def derivative(self, t, x, f_cmd):
        p = self.params
        cfg = self.config
        f = x[F]
        T, tau_x, tau_y, tau_z = p.rotor_thrusts_to_wrench(f)
        tau = np.array([tau_x, tau_y, tau_z])

        R = quat_to_rotmat(x[Q])
        force = R[:, 2] * T - cfg.linear_drag * x[V]
        if cfg.wind_force is not None:
            force = force + cfg.wind_force(t)

        omega = x[W]
        dx = np.empty(STATE_DIM)
        dx[P] = x[V]
        dx[V] = force / p.mass
        dx[V][2] -= p.gravity
        dx[Q] = 0.5 * quat_multiply(x[Q], np.array([0.0, *omega]))
        dx[W] = self._J_inv @ (tau - cross(omega, self._J @ omega))
        if cfg.motor_tau > 0.0:
            dx[F] = (f_cmd - f) / cfg.motor_tau
        else:
            dx[F] = 0.0
        return dx

    def step(self, t, x, f_cmd, dt):
        """Advance one RK4 step with the rotor command held constant (zero-order hold)."""
        f_cmd = self.saturate(f_cmd)
        if self.config.motor_tau <= 0.0:
            x = x.copy()
            x[F] = f_cmd
        k1 = self.derivative(t, x, f_cmd)
        k2 = self.derivative(t + dt / 2, x + dt / 2 * k1, f_cmd)
        k3 = self.derivative(t + dt / 2, x + dt / 2 * k2, f_cmd)
        k4 = self.derivative(t + dt, x + dt * k3, f_cmd)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[Q] /= np.linalg.norm(x_next[Q])
        return x_next
