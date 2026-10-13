import numpy as np

from ..params import QuadParams
from ..trajectories import Trajectory


class Controller:
    """All controllers return the body wrench [T, tau_x, tau_y, tau_z].

    The simulator converts the wrench to rotor thrusts and clips them to the
    motor limits, so every controller sees the same actuator saturation.
    """

    name = "base"

    def __init__(self, params: QuadParams):
        self.params = params

    def reset(self):
        pass

    def compute(self, t: float, x: np.ndarray, traj: Trajectory, dt: float) -> np.ndarray:
        raise NotImplementedError


def default_bandwidths():
    """Shared closed-loop targets so the comparison is about structure, not tuning effort.

    The cascaded PID and geometric controllers are both tuned to these natural
    frequencies. LQR and MPC weights are calibrated against the geometric
    controller (see lqr.default_weights).
    """
    return {
        "pos_wn": 4.0,     # rad/s, translational loop
        "pos_zeta": 0.85,
        "att_wn": 25.0,    # rad/s, rotational loop
        "att_zeta": 0.8,
    }
