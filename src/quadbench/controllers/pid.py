"""Cascaded PID, the structure used by most flight stacks (PX4, Crazyflie firmware).

    position PID  ->  desired acceleration  ->  desired roll, pitch, thrust
    attitude P    ->  desired body rates
    rate P        ->  body torque

The attitude loops act on Euler angles, which is simple but has a
singularity at 90 degrees of pitch and treats roll, pitch and yaw as
independent. The geometric controller removes both limitations.
"""

import numpy as np

from ..dynamics import P, Q, V, W
from ..rotations import quat_to_rotmat, rotmat_to_euler, wrap_angle
from .base import Controller, default_bandwidths


class CascadedPID(Controller):
    name = "PID"

    def __init__(self, params, max_tilt=np.radians(65), max_integral_acc=5.0,
                 integral_zone=0.25, integral_pole_ratio=0.4, bandwidths=None):
        super().__init__(params)
        bw = bandwidths or default_bandwidths()
        wn, zeta = bw["pos_wn"], bw["pos_zeta"]
        # Pole placement: the same (wn, zeta) pair as the geometric controller,
        # plus a real integrator pole at a = 0.4 wn.
        # (s^2 + 2 zeta wn s + wn^2)(s + a) = s^3 + kd s^2 + kp s + ki
        a = integral_pole_ratio * wn
        self.kd = 2 * zeta * wn + a
        self.kp = wn ** 2 + 2 * zeta * wn * a
        self.ki = wn ** 2 * a
        # Anti-windup: integrate only within `integral_zone` metres of the
        # reference, and cap the integral term's contribution (m/s^2).
        self.integral_zone = integral_zone
        self.integral_limit = max_integral_acc / self.ki
        # Attitude P on angle, then P on rate: theta_ddot = k_rate (k_att e - theta_dot)
        # gives wn^2 = k_rate k_att and 2 zeta wn = k_rate.
        a_wn, a_zeta = bw["att_wn"], bw["att_zeta"]
        self.k_rate = 2 * a_zeta * a_wn
        self.k_att = a_wn ** 2 / self.k_rate
        self.max_tilt = max_tilt
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)

    def compute(self, t, x, traj, dt):
        p = self.params
        ref = traj(t)
        e_p = ref.pos - x[P]
        e_v = ref.vel - x[V]
        active = np.abs(e_p) < self.integral_zone
        self.integral = np.clip(self.integral + active * e_p * dt, -self.integral_limit, self.integral_limit)
        a_cmd = ref.acc + self.kp * e_p + self.kd * e_v + self.ki * self.integral

        R = quat_to_rotmat(x[Q])
        roll, pitch, yaw = rotmat_to_euler(R)

        # Invert R e3 ~ (a_x, a_y, g + a_z) in the heading frame.
        ax = a_cmd[0] * np.cos(yaw) + a_cmd[1] * np.sin(yaw)
        ay = -a_cmd[0] * np.sin(yaw) + a_cmd[1] * np.cos(yaw)
        az = max(p.gravity + a_cmd[2], 0.1 * p.gravity)
        pitch_d = np.clip(np.arctan2(ax, az), -self.max_tilt, self.max_tilt)
        roll_d = np.clip(np.arctan2(-ay, np.hypot(ax, az)), -self.max_tilt, self.max_tilt)
        thrust = p.mass * az / max(np.cos(roll) * np.cos(pitch), 0.3)

        e_att = np.array([roll_d - roll, pitch_d - pitch, wrap_angle(ref.yaw - yaw)])
        rate_d = self.k_att * e_att
        torque = p.J @ (self.k_rate * (rate_d - x[W]))
        return np.array([thrust, *torque])
