"""Closed-loop simulation: controller at `ctrl_dt`, physics at `sim_dt`.

The controller sees a (possibly noisy) copy of the state and designs with the
nominal parameters. The plant may use different parameters and extra effects
(motor lag, drag, wind) that the controllers do not model.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .dynamics import P, Q, STATE_DIM, V, W, PlantConfig, Quadrotor
from .params import QuadParams
from .rotations import quat_multiply
from .trajectories import Trajectory


@dataclass
class SensorNoise:
    pos_std: float = 0.0      # m
    vel_std: float = 0.0      # m/s
    att_std: float = 0.0      # rad, per axis
    gyro_std: float = 0.0     # rad/s
    seed: int = 0


@dataclass
class Scenario:
    name: str
    description: str
    trajectory: Trajectory
    duration: float
    initial_state: Callable[[QuadParams], np.ndarray]
    plant: PlantConfig = field(default_factory=PlantConfig)
    mass_factor: float = 1.0
    noise: Optional[SensorNoise] = None
    settle_from: float = 0.0  # time from which settling is measured (e.g. a step's start)


@dataclass
class SimResult:
    controller: str
    scenario: str
    t: np.ndarray
    x: np.ndarray
    ref_pos: np.ndarray
    rotor_cmd: np.ndarray
    saturated: np.ndarray
    compute_time: np.ndarray
    crashed: bool
    crash_reason: str = ""


def mix(params: QuadParams, wrench):
    """Wrench -> rotor thrusts, keeping attitude torques when the motors saturate.

    If the torque request alone needs more spread than the motors have, the
    torques are scaled down. The collective thrust is then shifted so every
    rotor stays in [0, f_max]. Flight stacks do the same (PX4 calls it
    desaturation), and it is applied identically to every controller.
    """
    f_max = params.max_rotor_thrust
    f_torque = params.allocation_inv @ np.array([0.0, *wrench[1:]])
    saturated = False
    spread = f_torque.max() - f_torque.min()
    if spread > f_max:
        f_torque *= f_max / spread
        saturated = True
    base = wrench[0] / 4.0
    lo, hi = -f_torque.min(), f_max - f_torque.max()
    if base < lo:
        base, saturated = lo, True
    elif base > hi:
        base, saturated = hi, True
    return base + f_torque, saturated


def _measure(x, noise: Optional[SensorNoise], rng):
    if noise is None:
        return x
    y = x.copy()
    y[P] += rng.normal(0.0, noise.pos_std, 3)
    y[V] += rng.normal(0.0, noise.vel_std, 3)
    dtheta = rng.normal(0.0, noise.att_std, 3)
    dq = np.array([1.0, *(0.5 * dtheta)])
    y[Q] = quat_multiply(y[Q], dq / np.linalg.norm(dq))
    y[W] += rng.normal(0.0, noise.gyro_std, 3)
    return y


def simulate(controller, scenario: Scenario, params: QuadParams,
             sim_dt=0.001, ctrl_dt=0.005, max_error=5.0) -> SimResult:
    true_params = params.scaled(mass_factor=scenario.mass_factor)
    plant = Quadrotor(true_params, scenario.plant)
    rng = np.random.default_rng(scenario.noise.seed if scenario.noise else 0)
    controller.reset()

    substeps = int(round(ctrl_dt / sim_dt))
    n = int(round(scenario.duration / ctrl_dt))
    x = scenario.initial_state(params)

    t_log = np.zeros(n)
    x_log = np.zeros((n, STATE_DIM))
    ref_log = np.zeros((n, 3))
    f_log = np.zeros((n, 4))
    sat_log = np.zeros(n, dtype=bool)
    ct_log = np.zeros(n)
    crashed, reason = False, ""

    t = 0.0
    for i in range(n):
        y = _measure(x, scenario.noise, rng)
        tic = time.perf_counter()
        wrench = controller.compute(t, y, scenario.trajectory, ctrl_dt)
        ct_log[i] = time.perf_counter() - tic
        f_cmd, sat_log[i] = mix(params, wrench)

        t_log[i], x_log[i], f_log[i] = t, x, f_cmd
        ref_log[i] = scenario.trajectory(t).pos

        for _ in range(substeps):
            x = plant.step(t, x, f_cmd, sim_dt)
            t += sim_dt

        if not np.all(np.isfinite(x)):
            crashed, reason = True, "diverged"
        elif x[2] < 0.0:
            crashed, reason = True, "hit the ground"
        elif np.linalg.norm(x[P] - scenario.trajectory(t).pos) > max_error:
            crashed, reason = True, f"error above {max_error} m"
        if crashed:
            n = i + 1
            break

    return SimResult(
        controller=controller.name, scenario=scenario.name,
        t=t_log[:n], x=x_log[:n], ref_pos=ref_log[:n], rotor_cmd=f_log[:n],
        saturated=sat_log[:n], compute_time=ct_log[:n], crashed=crashed, crash_reason=reason,
    )
