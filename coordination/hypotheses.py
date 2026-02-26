"""
RLDDM Coordination — agent and composable hypothesis classes.

Four orthogonal axes:
  1. Partner mode    — FixedPartner | RotatingPartner
  2. Distance        — SymmetricDistance | AsymmetricBounds | AsymmetricMotorCost
  3. Reference frame — EgocentricCoordination | AllocentricCoordination
  4. Information     — NoCoupling | LiveCoupling

Reward rule: 1 if both agents reach the same bound (egocentric) or the same world
position (allocentric). No time window.
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
        decay_rate=0.1,
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

        # value of action 1 (upper in egocentric; world pos 1 in allocentric)
        self.v1 = 0.5
        # value of action 2 (lower in egocentric; world pos 2 in allocentric)
        self.v2 = 0.5
        self.kappa = 0.0
        self.X = 0.0
        self.t = 1
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
        """Run a sequential trial with no partner information."""
        self.reset()
        while self.lb < self.X < self.ub:
            self.step()
        return np.sign(self.X)  # +1 upper, -1 lower (body-relative)


# ---------------------------------------------------------------------------
# Simultaneous trial (shared by LiveCoupling)
# ---------------------------------------------------------------------------


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
        if not m2_done:
            a2.step(X_partner=x1)
            if not (a2.lb < a2.X < a2.ub):
                m2_done = True
    return np.sign(a1.X), np.sign(a2.X)


# ---------------------------------------------------------------------------
# Hypothesis 1 — Partner mode
# ---------------------------------------------------------------------------


class FixedPartner:
    """
    Agents are always paired with the same partner across all trials.
    Requires an even number of agents; pairs are formed by index order.
    """

    def make_pairs(self, agents, rng):
        return list(zip(agents[::2], agents[1::2]))


class RotatingPartner:
    """
    Agents are randomly re-paired from a pool each trial.
    Requires an even number of agents (minimum 4 for rotation to differ from fixed).
    """

    def make_pairs(self, agents, rng):
        idx = rng.permutation(len(agents))
        return [(agents[idx[i]], agents[idx[i + 1]]) for i in range(0, len(agents), 2)]


# ---------------------------------------------------------------------------
# Hypothesis 2 — Distance (bound asymmetry or motor cost)
# ---------------------------------------------------------------------------


class SymmetricDistance:
    """
    All agents have equal upper and lower bound distances.
    Both coordination equilibria (all-upper, all-lower) are equally likely.
    """

    def make_agents(self, n, **kwargs):
        return [RLDDMAgent(**kwargs) for _ in range(n)]


class AsymmetricBounds:
    """
    Paired agents have opposing bound asymmetries:
      Agent A (even index): bound_upper=a, bound_lower=b  →  lower bound closer
      Agent B (odd  index): bound_upper=b, bound_lower=a  →  upper bound closer
    """

    def __init__(self, bound_a_upper=1.0, bound_a_lower=0.25):
        self.bound_a_upper = bound_a_upper
        self.bound_a_lower = bound_a_lower

    def make_agents(self, n, **kwargs):
        return [
            RLDDMAgent(
                bound_upper=self.bound_a_upper,
                bound_lower=self.bound_a_lower,
                **kwargs,
            )
            if i % 2 == 0
            else RLDDMAgent(
                bound_upper=self.bound_a_lower,
                bound_lower=self.bound_a_upper,
                **kwargs,
            )
            for i in range(n)
        ]


class AsymmetricMotorCost:
    """
    Paired agents have opposing motor costs:
      Agent A (even index): crossing the upper bound incurs cost c
      Agent B (odd  index): crossing the lower bound incurs cost c
    Accumulation dynamics are identical; asymmetry appears only in the reward signal.
    """

    def __init__(self, cost=0.4):
        self.cost = cost

    def make_agents(self, n, **kwargs):
        # TODO: adjust cost upper and lower
        return [
            RLDDMAgent(motor_cost_upper=self.cost, motor_cost_lower=0.0, **kwargs)
            if i % 2 == 0
            else RLDDMAgent(motor_cost_upper=0.0, motor_cost_lower=self.cost, **kwargs)
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# Hypothesis 3 — Reference frame
# ---------------------------------------------------------------------------


# TODO: rewrite Ego and Allo agents
class EgocentricCoordination:
    """
    Agents learn the value of their own body-relative actions (go up / go down).
    Reward = 1 if both agents cross the same relative bound.
    """

    def prepare_agents(self, agents):
        for agent in agents:
            agent.orientation = 1

    def compute_reward(self, a1, a2, side1, side2):
        return float(side1 == side2)

    def update(self, agent, side, reward):
        eff = reward - (agent.motor_cost_upper if side > 0 else agent.motor_cost_lower)
        if side > 0:
            agent.v1 += agent.lr * (eff - agent.v1)
        else:
            agent.v2 += agent.lr * (eff - agent.v2)
        agent.kappa += agent.lr_kappa * (reward - agent.kappa)


class AllocentricCoordination:
    """
    Agents learn the value of reaching world positions (position 1 / position 2).
    Reward = 1 if both agents reach the same world position.
    Paired agents are assigned opposite orientations (+1 / -1).
    v1 always encodes world position 1 for both agents.
    """

    def prepare_agents(self, agents):
        for i, agent in enumerate(agents):
            agent.orientation = 1 if i % 2 == 0 else -1

    def compute_reward(self, a1, a2, side1, side2):
        world1 = a1.orientation * side1
        world2 = a2.orientation * side2
        return 1.0 if world1 == world2 else 0.0

    def update(self, agent, side, reward):
        eff = reward - (agent.motor_cost_upper if side > 0 else agent.motor_cost_lower)
        world_pos = agent.orientation * side
        if world_pos > 0:
            agent.v1 += agent.lr * (max(0.0, eff) - agent.v1)
        else:
            agent.v2 += agent.lr * (max(0.0, eff) - agent.v2)
        agent.kappa += agent.lr_kappa * (reward - agent.kappa)


# ---------------------------------------------------------------------------
# Hypothesis 4 — Information from the other accumulator
# ---------------------------------------------------------------------------


class NoCoupling:
    """
    Sequential trials: each agent runs to completion independently.
    Coordination can only emerge through the cross-trial RL signal.
    """

    def run_trial(self, a1, a2):
        side1 = a1.trial()
        side2 = a2.trial()
        return side1, side2


class LiveCoupling:
    """
    Simultaneous stepping: both agents step together, each observing the other's X.
    kappa (learned from reward) controls coupling strength.
    """

    def run_trial(self, a1, a2):
        return run_coupled_trial(a1, a2)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    partner,
    distance,
    frame,
    information,
    n_agents,
    n_trials=300,
    seed=0,
    agent_kwargs=None,
):
    """
    Run a single experiment with a specific combination of hypothesis classes.

    Parameters
    ----------
    partner       : FixedPartner | RotatingPartner
    distance      : SymmetricDistance | AsymmetricBounds | AsymmetricMotorCost
    frame         : EgocentricCoordination | AllocentricCoordination
    information   : NoCoupling | LiveCoupling
    agent_kwargs  : dict passed to distance.make_agents() → RLDDMAgent.__init__
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    agents = distance.make_agents(n_agents, **(agent_kwargs or {}))
    frame.prepare_agents(agents)

    reward_hist = np.zeros(n_trials)
    side_pref_hist = np.zeros((n_trials, n_agents))

    for trial in range(n_trials):
        pairs = partner.make_pairs(agents, rng)
        trial_reward = 0.0

        for a1, a2 in pairs:
            side1, side2 = information.run_trial(a1, a2)
            reward = frame.compute_reward(a1, a2, side1, side2)
            frame.update(a1, side1, reward)
            frame.update(a2, side2, reward)
            trial_reward += reward

        reward_hist[trial] = trial_reward / len(pairs)
        side_pref_hist[trial] = [a.v1 - a.v2 for a in agents]

    return reward_hist, side_pref_hist
