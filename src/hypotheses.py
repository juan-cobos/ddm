"""
Hypothesis configurations for RLDDM coordination.

Each class is a pure configuration object — no simulation logic.
All behavior is implemented in CoordinationEnv (env.py).

Four orthogonal axes:
  1. Partner mode    — FixedPartner | RotatingPartner
  2. Distance        — SymmetricDistance | AsymmetricBounds | MotorCost
  3. Reference frame — EgocentricCoordination | AllocentricCoordination
  4. Information     — NoCoupling | LiveCoupling
"""

from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Behavioral parameters shared by all agents in a simulation."""

    lr: float = 0.1
    lr_kappa: float = 0.0
    drift_rate: float = 1.0
    decay_rate: float = 0.0
    noise_std: float = 0.1
    dt: float = 1e-2

    def to_kwargs(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Hypothesis 1 — Partner mode
# ---------------------------------------------------------------------------


class FixedPartner:
    """Agents always paired with the same partner; pairs formed by index order."""


class RotatingPartner:
    """Agents randomly re-paired from the pool each trial."""


# ---------------------------------------------------------------------------
# Hypothesis 2 — Distance
# ---------------------------------------------------------------------------


class SymmetricDistance:
    """Equal upper and lower bounds; both coordination equilibria equally likely."""


@dataclass
class AsymmetricBounds:
    """
    Paired agents have opposing bound asymmetries.
    The shorter bound draws responses toward it, creating opposing biases within a pair.
    """

    bound_a: float = 1.0
    bound_b: float = 0.25


@dataclass
class MotorCost:
    """
    All agents share the same motor costs.
    Asymmetry enters only via the effective reward (reward − cost) used in value updates.
    """

    cost_upper: float = 0.0
    cost_lower: float = 0.1


# ---------------------------------------------------------------------------
# Hypothesis 3 — Reference frame
# ---------------------------------------------------------------------------


class EgocentricCoordination:
    """
    Agents learn the value of body-relative actions (go up / go down).
    Reward = 1 if both agents cross the same relative bound.
    """


class AllocentricCoordination:
    """
    Agents learn the value of world positions (position 1 / position 2).
    Reward = 1 if both agents reach the same world position.
    Paired agents are assigned opposite orientations (+1 / -1).
    """


# ---------------------------------------------------------------------------
# Hypothesis 4 — Information coupling
# ---------------------------------------------------------------------------


class NoCoupling:
    """Each agent runs to completion independently; coordination via RL signal only."""


class LiveCoupling:
    """Agents step simultaneously, each observing the other's accumulator."""
