"""Rotation helpers.

Conventions used everywhere in this package:
- World frame is z-up. Gravity points along -z.
- Quaternions are scalar-first: q = [w, x, y, z], and rotate body vectors into the world frame.
- Euler angles are ZYX (yaw, then pitch, then roll): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

import numpy as np


def cross(a, b):
    """3-vector cross product. About 10x faster than np.cross for single vectors."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def hat(v):
    """Skew-symmetric matrix such that hat(a) @ b == cross(a, b)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def vee(m):
    """Inverse of hat() for a skew-symmetric matrix."""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def quat_multiply(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(r):
    """Shepperd's method: picks the numerically safest branch."""
    trace = np.trace(r)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def euler_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotmat_to_euler(r):
    """Returns (roll, pitch, yaw) for the ZYX convention."""
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll, pitch, yaw):
    return rotmat_to_quat(euler_to_rotmat(roll, pitch, yaw))


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
