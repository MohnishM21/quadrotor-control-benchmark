"""Benchmark scenarios. Each one isolates a single difficulty."""

import numpy as np

from .dynamics import F, Q, PlantConfig, make_state
from .flatness import flat_state
from .rotations import rotmat_to_quat
from .simulator import Scenario, SensorNoise
from .trajectories import Hover, Step, circle, figure_eight


def _start_at(position, euler=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
    return lambda params: make_state(position=position, euler=euler, omega=omega, params=params)


def _start_on(traj):
    """Initial state exactly on the reference, so the run measures tracking only."""
    def init(params):
        ref = flat_state(traj, 0.0, params)
        x = make_state(position=ref.pos, velocity=ref.vel, omega=ref.omega, params=params)
        x[Q] = rotmat_to_quat(ref.R)
        x[F] = ref.thrust / 4.0
        return x
    return init


def _gusty_wind(t):
    # ~8% of the vehicle's weight steady, plus slow gusts of similar size.
    return np.array([0.02 + 0.015 * np.sin(2 * np.pi * 0.5 * t), 0.012 * np.sin(2 * np.pi * 0.3 * t + 1.0), 0.0])


def build_scenarios():
    hover = Hover((0.0, 0.0, 1.0))
    fig8 = figure_eight(x_amp=1.5, y_amp=0.75, period=4.0)
    fig8_fast = figure_eight(x_amp=1.5, y_amp=0.75, period=3.0)
    circ = circle(radius=1.0, period=4.0)

    return [
        Scenario("hover_recovery", "Start 0.6 m off target, tilted 30/-20 deg and spinning",
                 hover, 4.0, _start_at((0.4, -0.3, 0.6), euler=np.radians((30, -20, 25)), omega=(3.0, -2.0, 1.5))),
        Scenario("step", "1 m step in x and y, 0.5 m in z at t = 0.5 s",
                 Step((0, 0, 1), (1, 1, 1.5), t_step=0.5), 4.0, _start_at((0, 0, 1)), settle_from=0.5),
        Scenario("long_step", "3 m lateral step: large tilt, far outside the hover linearization",
                 Step((0, 0, 1), (3, 0, 1), t_step=0.2), 4.0, _start_at((0, 0, 1)), settle_from=0.2),
        Scenario("circle", "r = 1 m circle, 4 s period (1.6 m/s)", circ, 8.0, _start_on(circ)),
        Scenario("figure8", "Figure-8, 4 s period, peak tilt 39 deg", fig8, 8.0, _start_on(fig8)),
        Scenario("figure8_fast", "Figure-8, 3 s period, peak tilt 55 deg, 77% of max thrust",
                 fig8_fast, 6.0, _start_on(fig8_fast)),
        Scenario("flip_recovery", "Start nearly upside down (roll 170 deg) at 3 m altitude",
                 Hover((0.0, 0.0, 3.0)), 5.0, _start_at((0, 0, 3.0), euler=np.radians((170, 0, 0)))),
        Scenario("mass_mismatch", "Figure-8 with the real vehicle 30% heavier than the model",
                 fig8, 8.0, _start_on(fig8), mass_factor=1.3),
        Scenario("motor_lag", "Figure-8 with 30 ms first-order motor lag (not modelled)",
                 fig8, 8.0, _start_on(fig8), plant=PlantConfig(motor_tau=0.03)),
        Scenario("wind", "Circle with steady wind and gusts plus aerodynamic drag",
                 circ, 8.0, _start_on(circ), plant=PlantConfig(linear_drag=0.01, wind_force=_gusty_wind)),
        Scenario("sensor_noise", "Figure-8 with noisy state estimates",
                 fig8, 8.0, _start_on(fig8),
                 noise=SensorNoise(pos_std=0.005, vel_std=0.02, att_std=np.radians(0.5), gyro_std=0.02, seed=1)),
    ]


SCENARIOS = {s.name: s for s in build_scenarios()}
