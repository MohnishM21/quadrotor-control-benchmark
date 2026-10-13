"""Infinite-horizon LQR on the hover linearization, with flatness feedforward.

    u = u_ref - K (x - x_ref)

x_ref and u_ref come from differential flatness, so the controller only
corrects deviations from the reference. K is computed once for hover, so
tracking degrades as the vehicle tilts away from the linearization point.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from ..flatness import flat_state
from ..linear_model import hover_model, reduced_state, reference_state
from ..rotations import wrap_angle
from .base import Controller, default_bandwidths


def bryson(max_values):
    """Bryson's rule: weight = 1 / (largest acceptable value)^2."""
    return np.diag(1.0 / np.square(max_values))


def z_weights_matching(kp, kd, accel_max):
    """Bryson limits for (z, v_z) whose LQR gains equal the PD gains kp, kd.

    The vertical axis is a double integrator in acceleration units. Its LQR gains are
    kp = sqrt(q_p / r) and kd = sqrt(q_v / r + 2 kp), with r = 1 / accel_max^2.
    """
    r = 1.0 / accel_max ** 2
    return 1.0 / np.sqrt(kp ** 2 * r), 1.0 / np.sqrt((kd ** 2 - 2 * kp) * r)


def default_weights(params):
    """Bryson's rule, calibrated so LQR is as aggressive as the PID and geometric controllers.

    - Vertical: z and v_z limits are solved in closed form, so the thrust gains
      equal the geometric controller's k_x and k_v exactly.
    - Horizontal: the x/y limits were picked by grid search so a 0.3 m x-step
      matches the geometric controller (rise 0.79 s vs 0.74 s, no overshoot).
      tests/test_controllers.py checks both.
    """
    bw = default_bandwidths()
    kp, kd = bw["pos_wn"] ** 2, 2 * bw["pos_zeta"] * bw["pos_wn"]
    thrust_max = 0.5 * params.hover_thrust
    z_max, vz_max = z_weights_matching(kp, kd, thrust_max / params.mass)
    #        x     y     z      vx   vy   vz      roll, pitch, yaw    body rates
    q_max = [0.12, 0.12, z_max, 0.5, 0.5, vz_max, 0.35, 0.35, 0.35, 3.0, 3.0, 3.0]
    #        collective thrust   roll/pitch/yaw torque (N m)
    u_max = [thrust_max, 1.5e-3, 1.5e-3, 3e-4]
    return bryson(q_max), bryson(u_max)


class LQR(Controller):
    name = "LQR"

    def __init__(self, params, Q=None, R=None):
        super().__init__(params)
        Qd, Rd = default_weights(params)
        self.Q = Qd if Q is None else Q
        self.R = Rd if R is None else R
        A, B = hover_model(params)
        S = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, B.T @ S)
        self.closed_loop_poles = np.linalg.eigvals(A - B @ self.K)

    def compute(self, t, x, traj, dt):
        p = self.params
        ref = flat_state(traj, t, p)
        xs = reduced_state(x)
        xr = reference_state(ref)
        err = xs - xr
        err[8] = wrap_angle(err[8])
        u_ref = np.array([ref.thrust - p.hover_thrust, *ref.torque])
        u = u_ref - self.K @ err
        return np.array([p.hover_thrust + u[0], *u[1:]])
