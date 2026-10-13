"""Linearization about hover, shared by the LQR and MPC controllers.

12-state model, small-angle and zero-yaw assumptions:
    x = [p (3), v (3), roll, pitch, yaw, omega (3)]
    u = [T - m g, tau_x, tau_y, tau_z]

    p_dot      = v
    v_dot      = [g * pitch, -g * roll, (T - m g) / m]
    euler_dot  = omega            (body rates ~ Euler rates near hover)
    omega_dot  = J^-1 tau         (gyroscopic term is second order)
"""

import numpy as np
from scipy.linalg import expm

from .dynamics import P, Q, V, W
from .params import QuadParams
from .rotations import quat_to_rotmat, rotmat_to_euler

NX, NU = 12, 4


def hover_model(params: QuadParams):
    g = params.gravity
    A = np.zeros((NX, NX))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7] = g
    A[4, 6] = -g
    A[6:9, 9:12] = np.eye(3)

    B = np.zeros((NX, NU))
    B[5, 0] = 1.0 / params.mass
    B[9:12, 1:4] = np.linalg.inv(params.J)
    return A, B


def discretize(A, B, dt):
    """Exact zero-order-hold discretization."""
    n, m = B.shape
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A
    block[:n, n:] = B
    phi = expm(block * dt)
    return phi[:n, :n], phi[:n, n:]


def reduced_state(x):
    """Plant state (17,) -> linear-model state (12,)."""
    euler = rotmat_to_euler(quat_to_rotmat(x[Q]))
    return np.concatenate([x[P], x[V], euler, x[W]])


def reference_state(flat):
    """FlatState -> linear-model reference state (12,)."""
    return np.concatenate([flat.pos, flat.vel, flat.euler, flat.omega])
