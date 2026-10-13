import numpy as np
import pytest

from quadbench.dynamics import F, P, Q, V, W, PlantConfig, Quadrotor, make_state
from quadbench.params import QuadParams
from quadbench.rotations import quat_to_rotmat
from quadbench.simulator import mix

params = QuadParams()
hover_cmd = np.full(4, params.hover_thrust / 4)


def test_allocation_round_trip():
    wrench = np.array([0.3, 1e-3, -2e-3, 5e-4])
    np.testing.assert_allclose(params.rotor_thrusts_to_wrench(params.wrench_to_rotor_thrusts(wrench)), wrench)


def test_hover_is_an_equilibrium():
    quad = Quadrotor(params)
    x = make_state(position=(0, 0, 1), params=params)
    for _ in range(1000):
        x = quad.step(0.0, x, hover_cmd, 1e-3)
    np.testing.assert_allclose(x[P], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(x[V], 0, atol=1e-9)


def test_free_fall():
    quad = Quadrotor(params)
    x = make_state(position=(0, 0, 10), params=params)
    t = 0.5
    for _ in range(500):
        x = quad.step(0.0, x, np.zeros(4), 1e-3)
    assert x[2] == pytest.approx(10 - 0.5 * params.gravity * t ** 2, abs=1e-9)


def test_roll_torque_gives_expected_angular_acceleration():
    quad = Quadrotor(params)
    tau_x = 1e-4
    f = params.wrench_to_rotor_thrusts(np.array([params.hover_thrust, tau_x, 0, 0]))
    x = make_state(position=(0, 0, 1), params=params)
    x[F] = f
    dx = quad.derivative(0.0, x, f)
    assert dx[10] == pytest.approx(tau_x / params.inertia[0])
    assert dx[11] == pytest.approx(0.0, abs=1e-12)


def test_torque_free_rotation_conserves_angular_momentum():
    # Zero thrust, spinning about all three axes: |J omega| in the world frame is constant.
    quad = Quadrotor(params)
    x = make_state(omega=(5.0, -3.0, 8.0), params=params)
    x[F] = 0.0
    J = params.J

    def momentum(x):
        return quat_to_rotmat(x[Q]) @ (J @ x[W])

    h0 = momentum(x)
    for _ in range(2000):
        x = quad.step(0.0, x, np.zeros(4), 1e-3)
    np.testing.assert_allclose(momentum(x), h0, rtol=1e-6, atol=1e-12)
    assert np.linalg.norm(x[Q]) == pytest.approx(1.0)


def test_motor_lag_is_first_order():
    tau = 0.03
    quad = Quadrotor(params, PlantConfig(motor_tau=tau))
    x = make_state(position=(0, 0, 1), params=params)
    x[F] = 0.0
    cmd = np.full(4, 0.1)
    for _ in range(int(tau / 1e-4)):
        x = quad.step(0.0, x, cmd, 1e-4)
    # After one time constant a first-order lag reaches 1 - 1/e of the command.
    np.testing.assert_allclose(x[F], 0.1 * (1 - np.exp(-1)), rtol=1e-4)


@pytest.mark.parametrize("wrench", [
    [params.hover_thrust, 0.0, 0.0, 0.0],
    [3 * params.hover_thrust, 0.0, 0.0, 0.0],
    [-0.1, 1e-3, 0.0, 0.0],
    [params.hover_thrust, 0.05, -0.05, 0.01],
])
def test_mixer_keeps_rotors_in_bounds(wrench):
    f, _ = mix(params, np.array(wrench))
    assert np.all(f >= -1e-12)
    assert np.all(f <= params.max_rotor_thrust + 1e-12)


def test_mixer_preserves_torque_direction_when_saturated():
    request = np.array([params.hover_thrust, 0.05, -0.02, 0.0])
    f, saturated = mix(params, request)
    got = params.rotor_thrusts_to_wrench(f)[1:3]
    assert saturated
    cos = got @ request[1:3] / (np.linalg.norm(got) * np.linalg.norm(request[1:3]))
    assert cos == pytest.approx(1.0, abs=1e-9)
