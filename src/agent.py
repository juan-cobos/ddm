"""
RLDDM agent and coupled trial runner.
"""

import numpy as np


class RLDDMAgent:
    """
    Single accumulator with RL value learning.

    Parameters
    ----------
    orientation : +1 or -1
        Maps the agent's body-relative upper bound to a world position.
        +1 → upper bound = world position 1 (default, egocentric frame).
        -1 → upper bound = world position 2 (mirrored agent, allocentric frame).
        The drift is multiplied by orientation so the agent always moves toward
        its preferred world position regardless of physical side.
    motor_cost_upper / motor_cost_lower : float
        Reward penalty subtracted when the corresponding bound is crossed.
        Encodes the effort cost of a physical action independent of reward outcome.
    """

    def __init__(
        self,
        lr=0.1,
        lr_kappa=0.0,
        bound_upper=0.5,
        bound_lower=0.5,
        motor_cost_upper=0.0,
        motor_cost_lower=0.0,
        orientation=1,
        drift_rate=1.0,
        decay_rate=0.0,
        noise_std=0.1,
        dt=1e-2,
    ):
        self.lr = lr
        self.lr_kappa = lr_kappa
        self.bound_upper = bound_upper
        self.bound_lower = bound_lower
        self.motor_cost_upper = motor_cost_upper
        self.motor_cost_lower = motor_cost_lower
        self.orientation = orientation
        self.drift_rate = drift_rate
        self.decay_rate = decay_rate
        self.noise_std = noise_std
        self.dt = dt

        self.v1 = 0.5
        self.v2 = 0.5
        self.kappa = 0.0
        self.X = 0.0
        self.t = 1
        self.last_rt = 0
        self._update_bounds()

    def _update_bounds(self):
        decay = np.exp(-self.decay_rate * (self.v1 + self.v2) * self.t * self.dt)
        self.ub = self.bound_upper * decay
        self.lb = -self.bound_lower * decay

    def _noise(self):
        return np.random.normal(0, self.noise_std) * np.sqrt(self.dt)

    def step(self, X_partner=0.0):
        """
        Single integration step.
        orientation scales the drift so that v1 > v2 always moves toward world pos 1,
        regardless of whether the agent faces +1 or -1 in the world.
        """
        coupling = self.kappa * (X_partner - self.X) * self.dt
        self.X += (
            self.drift_rate * (self.v1 - self.v2) * self.orientation * self.dt
            + coupling
            + self._noise()
        )
        self.t += 1
        self._update_bounds()

    def reset(self):
        self.X = 0.0
        self.t = 1
        self._update_bounds()

    def trial(self):
        """Run a single trial with no partner information."""
        self.reset()
        while self.lb < self.X < self.ub:
            self.step()
        self.last_rt = self.t - 1  # steps taken = t − 1 (t starts at 1)
        return np.sign(self.X)


def run_coupled_trial(a1, a2):
    """Both agents step simultaneously; each observes the other's live X."""
    a1.reset()
    a2.reset()
    m1_done = m2_done = False
    while not (m1_done and m2_done):
        x1, x2 = a1.X, a2.X  # snapshot before stepping
        if not m1_done:
            a1.step(X_partner=x2)
            if not (a1.lb < a1.X < a1.ub):
                m1_done = True
                a1.last_rt = a1.t - 1  # record when this agent crossed its bound
        if not m2_done:
            a2.step(X_partner=x1)
            if not (a2.lb < a2.X < a2.ub):
                m2_done = True
                a2.last_rt = a2.t - 1
    return np.sign(a1.X), np.sign(a2.X)
