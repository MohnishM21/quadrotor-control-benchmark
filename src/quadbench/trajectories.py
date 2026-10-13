"""Reference trajectories with exact derivatives up to snap.

Each trajectory returns a Reference at time t. Controllers that use
differential flatness need acceleration and jerk, so derivatives are
analytic rather than numerically differentiated.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray
    yaw: float = 0.0


class Trajectory:
    def __call__(self, t: float) -> Reference:
        raise NotImplementedError


class Hover(Trajectory):
    def __init__(self, point, yaw=0.0):
        self.point = np.asarray(point, dtype=float)
        self.yaw = yaw

    def __call__(self, t):
        z = np.zeros(3)
        return Reference(self.point.copy(), z, z, z, z, self.yaw)


class Step(Trajectory):
    """Holds `start`, then jumps to `end` at `t_step`. Derivatives are zero."""

    def __init__(self, start, end, t_step=0.5, yaw=0.0):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.t_step = t_step
        self.yaw = yaw

    def __call__(self, t):
        z = np.zeros(3)
        pos = self.start if t < self.t_step else self.end
        return Reference(pos.copy(), z, z, z, z, self.yaw)


class Sinusoid(Trajectory):
    """Each axis follows p_k(t) = c_k + a_k sin(w_k t + phi_k)."""

    def __init__(self, center, amplitude, omega, phase, yaw=0.0):
        self.c = np.asarray(center, dtype=float)
        self.a = np.asarray(amplitude, dtype=float)
        self.w = np.asarray(omega, dtype=float)
        self.phi = np.asarray(phase, dtype=float)
        self.yaw = yaw

    def _derivative(self, t, n):
        # d^n/dt^n sin(wt + phi) = w^n sin(wt + phi + n pi / 2)
        return self.a * self.w ** n * np.sin(self.w * t + self.phi + n * np.pi / 2)

    def __call__(self, t):
        return Reference(
            pos=self.c + self._derivative(t, 0),
            vel=self._derivative(t, 1),
            acc=self._derivative(t, 2),
            jerk=self._derivative(t, 3),
            snap=self._derivative(t, 4),
            yaw=self.yaw,
        )


def circle(radius=1.0, period=4.0, height=1.0):
    w = 2 * np.pi / period
    return Sinusoid(center=(0.0, 0.0, height), amplitude=(radius, radius, 0.0),
                    omega=(w, w, 0.0), phase=(np.pi / 2, 0.0, 0.0))


def figure_eight(x_amp=1.5, y_amp=0.75, period=4.0, height=1.0, z_amp=0.0):
    """Lemniscate of Gerono: x = A sin(wt), y = B sin(2wt), optional vertical bob."""
    w = 2 * np.pi / period
    return Sinusoid(center=(0.0, 0.0, height), amplitude=(x_amp, y_amp, z_amp),
                    omega=(w, 2 * w, 2 * w), phase=(0.0, 0.0, 0.0))
