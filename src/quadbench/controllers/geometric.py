"""Geometric tracking control on SE(3) (Lee, Leok & McClamroch, CDC 2010).

The attitude error is defined directly on the rotation group:
    e_R     = 0.5 vee(R_c^T R - R^T R_c)
    e_Omega = Omega - R^T R_c Omega_c
so there are no Euler-angle singularities. The paper proves almost-global
exponential attitude stability: the controller recovers from any attitude
except an exactly inverted one. Lee et al. use a z-down frame; this version
is rewritten for z-up.
"""

import numpy as np

from ..dynamics import P, Q, V, W
from ..flatness import flat_state
from ..rotations import cross, hat, quat_to_rotmat, vee
from .base import Controller, default_bandwidths


class GeometricSE3(Controller):
    name = "Geometric"

    def __init__(self, params, bandwidths=None, use_feedforward=True):
        super().__init__(params)
        bw = bandwidths or default_bandwidths()
        m, J = params.mass, params.J
        self.kx = m * bw["pos_wn"] ** 2
        self.kv = m * 2 * bw["pos_zeta"] * bw["pos_wn"]
        self.kR = J * bw["att_wn"] ** 2
        self.kW = J * 2 * bw["att_zeta"] * bw["att_wn"]
        self.use_feedforward = use_feedforward

    def compute(self, t, x, traj, dt):
        p = self.params
        ref = flat_state(traj, t, p) if self.use_feedforward else None
        r = traj(t)
        e3 = np.array([0.0, 0.0, 1.0])

        R = quat_to_rotmat(x[Q])
        omega = x[W]
        e_x = x[P] - r.pos
        e_v = x[V] - r.vel

        f_des = -self.kx * e_x - self.kv * e_v + p.mass * (p.gravity * e3 + r.acc)
        thrust = f_des @ R[:, 2]

        # Commanded attitude: b3 along f_des, heading from the reference yaw.
        b3c = f_des / np.linalg.norm(f_des)
        b1d = np.array([np.cos(r.yaw), np.sin(r.yaw), 0.0])
        b2c = cross(b3c, b1d)
        b2c /= np.linalg.norm(b2c)
        Rc = np.column_stack([cross(b2c, b3c), b2c, b3c])

        # Angular velocity and acceleration of the reference attitude (feedforward).
        omega_c = ref.omega if ref is not None else np.zeros(3)
        omega_c_dot = ref.omega_dot if ref is not None else np.zeros(3)

        e_R = 0.5 * vee(Rc.T @ R - R.T @ Rc)
        RtRc = R.T @ Rc
        e_W = omega - RtRc @ omega_c

        J = p.J
        torque = (-self.kR @ e_R - self.kW @ e_W + cross(omega, J @ omega)
                  - J @ (hat(omega) @ RtRc @ omega_c - RtRc @ omega_c_dot))
        return np.array([thrust, *torque])
