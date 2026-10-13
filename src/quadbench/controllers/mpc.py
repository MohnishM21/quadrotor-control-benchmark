"""Linear tracking MPC with rotor and tilt constraints, solved as a sparse QP by OSQP.

The MPC works on the error from the flatness reference, like the LQR:
    dx = x - x_ref,   du = u - u_ref

At every control step it solves

    minimise   sum_k |dx_k|_Q^2 + |du_k|_R^2 + |dx_N|_P^2 + sum_k (rho s_k^2 + mu s_k)
    subject to dx_0 = measured error
               dx_{k+1} = A_d dx_k + B_d du_k
               0 <= M^-1 (w_ref,k + du_k) <= f_max            each rotor thrust
               |roll_k|, |pitch_k| <= max_tilt + s_k,  s_k >= 0   soft tilt limit, k = 1..N

where w_ref,k is the reference wrench k steps ahead. The model is the hover
linearization, so MPC shares LQR's small-angle limits. The terminal weight P
solves the discrete Riccati equation: when no constraint is active, the first
move equals the discrete-time LQR law for the same weights.

The tilt limit is what separates MPC from LQR here. The linear model
believes lateral acceleration grows without bound as the vehicle tilts
(a_x = g * pitch), so on a long step LQR asks for more than 90 degrees and
falls. MPC keeps its plan inside the region where the model is roughly valid.
The limit is soft (slack s_k), so a start outside it, as in the flip test,
leaves the QP feasible.

A first version predicted absolute states x_k and penalised x_k - x_ref,k.
That forces the linear model to reproduce the nonlinear reference
(a_x = g * pitch instead of g * tan(pitch)). At 39 degrees of tilt the
mismatch cost 5 cm of tracking error. The error formulation avoids it.
"""

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are

from ..flatness import attitude_from_acc, flat_state
from ..linear_model import NU, NX, discretize, hover_model, reduced_state, reference_state
from ..rotations import rotmat_to_euler, wrap_angle
from .base import Controller
from .lqr import default_weights

ROLL, PITCH = 6, 7


class LinearMPC(Controller):
    name = "MPC"

    def __init__(self, params, horizon=20, dt_pred=0.02, Q=None, R=None,
                 max_tilt=np.radians(60), slack_weight=1e4, slack_linear=1e3):
        super().__init__(params)
        self.N = horizon
        self.dt_pred = dt_pred
        Qd, Rd = default_weights(params)
        self.Q = Qd if Q is None else Q
        self.R = Rd if R is None else R
        A, B = hover_model(params)
        self.Ad, self.Bd = discretize(A, B, dt_pred)
        self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.f_max = params.max_rotor_thrust
        self.max_tilt = max_tilt
        self.slack_weight = slack_weight
        self.slack_linear = slack_linear
        self.last_status = None
        self.last_solution = None
        self._build()

    # Decision vector z = [dx_0 .. dx_N, du_0 .. du_{N-1}, s_1 .. s_N]
    def _build(self):
        N = self.N
        nx_tot, nu_tot = NX * (N + 1), NU * N
        self._iu = nx_tot                      # first input index
        self._is = nx_tot + nu_tot             # first slack index
        nz = nx_tot + nu_tot + N

        H = sp.block_diag([sp.kron(sp.eye(N), self.Q), self.P, sp.kron(sp.eye(N), self.R),
                           self.slack_weight * sp.eye(N)], format="csc")
        q = np.zeros(nz)
        q[self._is:] = self.slack_linear
        no_slack = sp.csc_matrix((nx_tot, N))

        # Dynamics: dx_0 = measured error, and -dx_{k+1} + A dx_k + B du_k = 0
        Ax = sp.kron(sp.eye(N + 1), -sp.eye(NX)) + sp.kron(sp.eye(N + 1, k=-1), self.Ad)
        Bu = sp.kron(sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]), self.Bd)
        A_eq = sp.hstack([Ax, Bu, no_slack])

        # Rotor limits on M^-1 du_k; the bounds move with the reference wrench.
        A_rotor = sp.hstack([sp.csc_matrix((nu_tot, nx_tot)), sp.kron(sp.eye(N), self.params.allocation_inv),
                             sp.csc_matrix((nu_tot, N))])

        # Tilt: for k = 1..N and each of roll, pitch:
        #   d_angle_k - s_k <= max_tilt - ref_angle_k
        #   d_angle_k + s_k >= -max_tilt - ref_angle_k
        pick = np.zeros((2, NX))
        pick[0, ROLL] = pick[1, PITCH] = 1.0
        select = sp.hstack([sp.csc_matrix((2 * N, NX)), sp.kron(sp.eye(N), pick)])   # skips dx_0
        slack_pairs = sp.kron(sp.eye(N), np.ones((2, 1)))
        A_tilt_hi = sp.hstack([select, sp.csc_matrix((2 * N, nu_tot)), -slack_pairs])
        A_tilt_lo = sp.hstack([select, sp.csc_matrix((2 * N, nu_tot)), slack_pairs])
        A_slack = sp.hstack([sp.csc_matrix((N, nx_tot + nu_tot)), sp.eye(N)])

        self._A = sp.vstack([A_eq, A_rotor, A_tilt_hi, A_tilt_lo, A_slack], format="csc")
        self._n_eq = nx_tot
        self.solver = osqp.OSQP()
        hover_w = np.tile([self.params.hover_thrust, 0.0, 0.0, 0.0], (N, 1))
        l, u = self._bounds(np.zeros(NX), hover_w, np.zeros((N, 2)))
        self.solver.setup(H, q, self._A, l, u, verbose=False, warm_starting=True,
                          polishing=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)

    def _bounds(self, dx0, wrench_ref, tilt_ref):
        """tilt_ref: reference (roll, pitch) for k = 1..N, shape (N, 2)."""
        N = self.N
        eq = np.zeros(self._n_eq)
        eq[:NX] = -dx0
        f_ref = (self.params.allocation_inv @ wrench_ref.T).T.ravel()
        inf = np.full(2 * N, np.inf)
        tilt_ref = tilt_ref.ravel()
        lower = np.concatenate([eq, -f_ref, -inf, -self.max_tilt - tilt_ref, np.zeros(N)])
        upper = np.concatenate([eq, self.f_max - f_ref, self.max_tilt - tilt_ref, inf, np.full(N, np.inf)])
        return lower, upper

    def _horizon_reference(self, t, traj, first):
        """Reference wrench for k = 0..N-1 and roll/pitch for k = 1..N.

        Thrust and attitude are exact. The small torque feedforward is only
        computed for k = 0, where it is applied.
        """
        p = self.params
        wrench = np.zeros((self.N, NU))
        wrench[0] = [first.thrust, *first.torque]
        tilt = np.zeros((self.N, 2))
        for k in range(1, self.N + 1):
            ref = traj(t + k * self.dt_pred)
            R, thrust = attitude_from_acc(ref.acc, ref.yaw, p)
            if k < self.N:
                wrench[k, 0] = thrust
            tilt[k - 1] = rotmat_to_euler(R)[:2]
        return wrench, tilt

    def compute(self, t, x, traj, dt):
        p = self.params
        ref = flat_state(traj, t, p)
        dx0 = reduced_state(x) - reference_state(ref)
        dx0[8] = wrap_angle(dx0[8])
        wrench_ref, tilt_ref = self._horizon_reference(t, traj, ref)

        l, u = self._bounds(dx0, wrench_ref, tilt_ref)
        self.solver.update(l=l, u=u)
        res = self.solver.solve(raise_error=False)
        self.last_status = res.info.status
        if res.info.status in ("solved", "solved inaccurate"):
            self.last_solution = res.x
            du0 = res.x[self._iu: self._iu + NU]
        else:
            du0 = np.zeros(NU)                        # fall back to pure feedforward
        return wrench_ref[0] + du0
