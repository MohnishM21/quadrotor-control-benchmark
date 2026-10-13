import numpy as np

from .dynamics import P, Q
from .rotations import quat_to_rotmat


def position_error(result):
    return np.linalg.norm(result.x[:, P] - result.ref_pos, axis=1)


def settling_time(result, t_start=0.0, tol=0.05):
    """Time after t_start until the position error stays below `tol` (m). NaN if never."""
    err = position_error(result)
    mask = result.t >= t_start
    t, err = result.t[mask], err[mask]
    outside = np.nonzero(err > tol)[0]
    if len(outside) == 0:
        return 0.0
    if outside[-1] == len(err) - 1:
        return np.nan
    return t[outside[-1] + 1] - t_start


def max_tilt_deg(result):
    """Largest angle between the body z-axis and vertical."""
    cos_tilt = [quat_to_rotmat(q)[2, 2] for q in result.x[:, Q]]
    return float(np.degrees(np.arccos(np.clip(min(cos_tilt), -1.0, 1.0))))


def summarize(result, scenario):
    err = position_error(result)
    return {
        "scenario": result.scenario,
        "controller": result.controller,
        "status": "crashed: " + result.crash_reason if result.crashed else "ok",
        "rmse_cm": 100 * float(np.sqrt(np.mean(err ** 2))),
        "max_err_cm": 100 * float(np.max(err)),
        "settle_s": float(settling_time(result, scenario.settle_from)),
        "max_tilt_deg": max_tilt_deg(result),
        "saturated_pct": 100 * float(np.mean(result.saturated)),
        "compute_ms": 1000 * float(np.mean(result.compute_time)),
    }
