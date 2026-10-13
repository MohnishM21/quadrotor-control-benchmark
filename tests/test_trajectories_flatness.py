import numpy as np
import pytest

from quadbench.flatness import flat_state
from quadbench.params import QuadParams
from quadbench.trajectories import Hover, circle, figure_eight

params = QuadParams()
TRAJECTORIES = [circle(), figure_eight(), figure_eight(period=3.0, z_amp=0.3)]


@pytest.mark.parametrize("traj", TRAJECTORIES)
@pytest.mark.parametrize("t", [0.0, 0.7, 2.3])
def test_derivatives_match_finite_differences(traj, t):
    h = 1e-5
    chain = ["pos", "vel", "acc", "jerk", "snap"]
    for lower, higher in zip(chain, chain[1:]):
        numeric = (getattr(traj(t + h), lower) - getattr(traj(t - h), lower)) / (2 * h)
        np.testing.assert_allclose(numeric, getattr(traj(t), higher), atol=1e-4)


def test_hover_flat_state_is_level_and_at_rest():
    s = flat_state(Hover((1, 2, 3)), 0.0, params)
    np.testing.assert_allclose(s.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(s.omega, 0, atol=1e-9)
    assert s.thrust == pytest.approx(params.hover_thrust)


@pytest.mark.parametrize("traj", TRAJECTORIES)
@pytest.mark.parametrize("t", [0.3, 1.1, 2.9])
def test_thrust_axis_produces_reference_acceleration(traj, t):
    s = flat_state(traj, t, params)
    acc = s.R[:, 2] * s.thrust / params.mass - np.array([0, 0, params.gravity])
    np.testing.assert_allclose(acc, traj(t).acc, atol=1e-10)


@pytest.mark.parametrize("traj", TRAJECTORIES)
@pytest.mark.parametrize("t", [0.3, 1.1, 2.9])
def test_roll_pitch_rates_match_closed_form(traj, t):
    """Mellinger & Kumar (2011): b3_dot = (m / T)(j - (b3 . j) b3); p = -b3_dot . b2, q = b3_dot . b1."""
    s = flat_state(traj, t, params)
    b1, b2, b3 = s.R.T
    j = traj(t).jerk
    b3_dot = params.mass / s.thrust * (j - (b3 @ j) * b3)
    # Central differences with h = 1e-3 are accurate to about 1e-5 relative.
    np.testing.assert_allclose(s.omega[:2], [-b3_dot @ b2, b3_dot @ b1], rtol=1e-4, atol=1e-6)
