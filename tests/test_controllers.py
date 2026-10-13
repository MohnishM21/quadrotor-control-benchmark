import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from quadbench.controllers import ALL_CONTROLLERS, LinearMPC
from quadbench.dynamics import P, make_state
from quadbench.linear_model import NU, NX, discretize, hover_model
from quadbench.metrics import position_error
from quadbench.params import QuadParams
from quadbench.scenarios import SCENARIOS
from quadbench.simulator import Scenario, simulate
from quadbench.trajectories import Hover, circle

params = QuadParams()


def offset_hover(duration=5.0):
    return Scenario("offset", "", Hover((0, 0, 1)), duration,
                    lambda p: make_state(position=(0.2, -0.1, 0.85), params=p))


@pytest.mark.parametrize("name", ALL_CONTROLLERS)
def test_converges_to_hover(name):
    r = simulate(ALL_CONTROLLERS[name](params), offset_hover(), params)
    assert not r.crashed
    assert position_error(r)[-1] < 1e-3


@pytest.mark.parametrize("name, tol_cm", [("PID", 2.0), ("LQR", 0.05), ("Geometric", 0.05), ("MPC", 0.05)])
def test_tracks_circle(name, tol_cm):
    scenario = SCENARIOS["circle"]
    short = Scenario("circle", "", scenario.trajectory, 3.0, scenario.initial_state)
    r = simulate(ALL_CONTROLLERS[name](params), short, params)
    assert 100 * position_error(r).max() < tol_cm


def test_geometric_recovers_from_near_inverted_start():
    r = simulate(ALL_CONTROLLERS["Geometric"](params), SCENARIOS["flip_recovery"], params)
    assert not r.crashed
    assert position_error(r)[-1] < 1e-2
    assert r.x[-1, 2] > 2.9


def test_mpc_plan_respects_rotor_limits():
    mpc = LinearMPC(params)
    x = make_state(position=(1.5, -1.0, 0.4), params=params)   # far away: constraints must bind
    mpc.compute(0.0, x, Hover((0, 0, 1)), 0.005)
    assert mpc.last_status == "solved"
    du = mpc.last_solution[mpc._iu: mpc._iu + NU * mpc.N].reshape(mpc.N, NU)
    wrench = du + np.array([params.hover_thrust, 0, 0, 0])
    f = (params.allocation_inv @ wrench.T).T
    assert f.min() > -1e-4 and f.max() < params.max_rotor_thrust + 1e-4
    assert np.isclose(f, params.max_rotor_thrust, atol=1e-3).any() or np.isclose(f, 0, atol=1e-3).any()


def test_tilt_limit_is_what_saves_mpc_on_long_step():
    from quadbench.metrics import max_tilt_deg
    scenario = SCENARIOS["long_step"]
    lqr = simulate(ALL_CONTROLLERS["LQR"](params), scenario, params)
    mpc = simulate(LinearMPC(params), scenario, params)
    assert lqr.crashed
    assert not mpc.crashed
    assert max_tilt_deg(mpc) < 65


def test_unconstrained_mpc_matches_discrete_lqr():
    mpc = LinearMPC(params)
    A, B = discretize(*hover_model(params), mpc.dt_pred)
    S = solve_discrete_are(A, B, mpc.Q, mpc.R)
    K = np.linalg.solve(mpc.R + B.T @ S @ B, B.T @ S @ A)
    x = make_state(position=(0.01, -0.005, 1.002), params=params)  # tiny error: no constraint active
    wrench = mpc.compute(0.0, x, Hover((0, 0, 1)), 0.005)
    dx = np.zeros(NX)
    dx[0:3] = x[P] - np.array([0, 0, 1])
    expected = -K @ dx
    np.testing.assert_allclose(wrench - [params.hover_thrust, 0, 0, 0], expected, rtol=2e-3, atol=1e-7)


def test_lqr_vertical_gains_match_geometric():
    lqr, geo = ALL_CONTROLLERS["LQR"](params), ALL_CONTROLLERS["Geometric"](params)
    assert lqr.K[0, 2] == pytest.approx(geo.kx, rel=1e-6)   # thrust per metre of z error
    assert lqr.K[0, 5] == pytest.approx(geo.kv, rel=1e-6)   # thrust per m/s of v_z error


def test_lqr_horizontal_step_matches_geometric():
    from quadbench.metrics import settling_time
    from quadbench.trajectories import Step
    step = Scenario("xstep", "", Step((0, 0, 1), (0.3, 0, 1), 0.2), 3.0,
                    lambda p: make_state(position=(0, 0, 1), params=p), settle_from=0.2)
    times = [settling_time(simulate(ALL_CONTROLLERS[n](params), step, params), 0.2) for n in ("LQR", "Geometric")]
    assert abs(times[0] - times[1]) < 0.1


def test_all_benchmark_scenarios_construct():
    for s in SCENARIOS.values():
        x0 = s.initial_state(params)
        assert x0.shape == (17,)
        assert np.all(np.isfinite(x0))
    assert circle()(0.0).pos[0] == pytest.approx(1.0)
