import numpy as np
import pytest

from quadbench.rotations import (cross, euler_to_quat, euler_to_rotmat, hat, quat_to_rotmat,
                                 rotmat_to_euler, rotmat_to_quat, vee)

rng = np.random.default_rng(0)


def random_euler():
    return rng.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])


@pytest.mark.parametrize("_", range(20))
def test_euler_rotmat_round_trip(_):
    e = random_euler()
    np.testing.assert_allclose(rotmat_to_euler(euler_to_rotmat(*e)), e, atol=1e-10)


@pytest.mark.parametrize("_", range(20))
def test_quat_rotmat_round_trip(_):
    R = euler_to_rotmat(*random_euler())
    np.testing.assert_allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-12)


def test_rotation_matrices_are_orthonormal():
    R = quat_to_rotmat(euler_to_quat(0.3, -0.7, 2.0))
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_hat_vee_and_cross():
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(hat(a) @ b, np.cross(a, b))
    np.testing.assert_allclose(cross(a, b), np.cross(a, b))
    np.testing.assert_allclose(vee(hat(a)), a)
