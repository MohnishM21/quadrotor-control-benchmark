"""Physical parameters of the simulated vehicle.

Defaults approximate a Crazyflie 2.x nano-quadrotor. The values follow the
cf2x model shipped with gym-pybullet-drones (Panerati et al., 2021).
"""

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class QuadParams:
    mass: float = 0.027                  # kg
    inertia: tuple = (1.4e-5, 1.4e-5, 2.17e-5)  # kg m^2, principal axes
    arm_length: float = 0.0397           # m, centre to rotor
    thrust_to_weight: float = 2.25
    kf: float = 3.16e-10                 # thrust per (RPM^2)
    km: float = 7.94e-12                 # rotor drag torque per (RPM^2)
    gravity: float = GRAVITY
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def J(self):
        return np.diag(self.inertia)

    @property
    def hover_thrust(self):
        return self.mass * self.gravity

    @property
    def max_rotor_thrust(self):
        return self.thrust_to_weight * self.hover_thrust / 4.0

    @property
    def yaw_moment_coeff(self):
        """Reaction torque about body z per newton of rotor thrust."""
        return self.km / self.kf

    @property
    def allocation(self):
        """Maps rotor thrusts f (4,) to the body wrench [T, tau_x, tau_y, tau_z].

        X configuration. Rotor i sits at (x_i, y_i) and spins with sign s_i.
        A rotor's thrust f produces tau_x = y f, tau_y = -x f, and a reaction
        yaw torque s c f. Diagonal rotors spin the same way.
        """
        if "allocation" not in self._cache:
            d = self.arm_length / np.sqrt(2.0)
            x = np.array([d, -d, -d, d])
            y = np.array([-d, -d, d, d])
            s = np.array([1.0, -1.0, 1.0, -1.0])
            c = self.yaw_moment_coeff
            self._cache["allocation"] = np.vstack([np.ones(4), y, -x, s * c])
        return self._cache["allocation"]

    @property
    def allocation_inv(self):
        if "allocation_inv" not in self._cache:
            self._cache["allocation_inv"] = np.linalg.inv(self.allocation)
        return self._cache["allocation_inv"]

    def wrench_to_rotor_thrusts(self, wrench):
        return self.allocation_inv @ wrench

    def rotor_thrusts_to_wrench(self, f):
        return self.allocation @ f

    def scaled(self, mass_factor=1.0, inertia_factor=1.0):
        """A copy with a different mass or inertia. Used for model-mismatch tests."""
        return QuadParams(
            mass=self.mass * mass_factor,
            inertia=tuple(i * inertia_factor for i in self.inertia),
            arm_length=self.arm_length,
            # Keep the same motors: the absolute maximum rotor thrust does not change.
            thrust_to_weight=self.thrust_to_weight / mass_factor,
            kf=self.kf,
            km=self.km,
            gravity=self.gravity,
        )
