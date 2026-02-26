"""
Coordination environment for RLDDM hypothesis testing.
"""

import dataclasses

import numpy as np
import polars as pl

from agent import RLDDMAgent, run_coupled_trial
from hypotheses import (
    AgentConfig,
    AllocentricCoordination,
    AsymmetricBounds,
    EgocentricCoordination,
    FixedPartner,
    LiveCoupling,
    MotorCost,
    NoCoupling,
    RotatingPartner,
    SymmetricDistance,
)


class CoordinationEnv:
    """
    RL-style environment for two-agent coordination via RLDDM.

    Parameters
    ----------
    n_agents : even integer; total number of agents in the pool
    partner  : FixedPartner | RotatingPartner
    distance : SymmetricDistance | AsymmetricBounds | MotorCost
    frame    : EgocentricCoordination | AllocentricCoordination
    coupling : NoCoupling | LiveCoupling
    config   : AgentConfig with lr, drift_rate, decay_rate, noise_std, dt
    """

    def __init__(self, n_agents, partner, distance, frame, coupling, config=None):
        self.n_agents = n_agents
        self.partner = partner
        self.distance = distance
        self.frame = frame
        self.coupling = coupling
        self._cfg = config or AgentConfig()

    @property
    def label(self):
        """Short descriptor used as CSV filename prefix."""
        def h(hyp):
            name = type(hyp).__name__
            if dataclasses.is_dataclass(hyp):
                vals = "_".join(str(v) for v in dataclasses.astuple(hyp))
                return f"{name}_{vals}"
            return name
        return "__".join([h(self.partner), h(self.distance), h(self.frame), h(self.coupling)])

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _make_agents(self):
        n, kw = self.n_agents, self._cfg.to_kwargs()
        match self.distance:
            case SymmetricDistance():
                return [RLDDMAgent(**kw) for _ in range(n)]
            case AsymmetricBounds(bound_a=a, bound_b=b):
                return [
                    RLDDMAgent(bound_upper=a, bound_lower=b, **kw)
                    if i % 2 == 0
                    else RLDDMAgent(bound_upper=b, bound_lower=a, **kw)
                    for i in range(n)
                ]
            case MotorCost(cost_upper=cu, cost_lower=cl):
                return [
                    RLDDMAgent(motor_cost_upper=cu, motor_cost_lower=cl, **kw)
                    for _ in range(n)
                ]
            case _:
                raise ValueError(f"Unknown distance hypothesis: {self.distance!r}")

    # ------------------------------------------------------------------
    # Per-pair helpers
    # ------------------------------------------------------------------

    def _set_orientations(self, a1, a2):
        """Assign orientations based on reference frame and pair order."""
        match self.frame:
            case EgocentricCoordination():
                a1.orientation = a2.orientation = 1
            case AllocentricCoordination():
                a1.orientation = 1
                a2.orientation = -1
            case _:
                raise ValueError(f"Unknown frame hypothesis: {self.frame!r}")

    def _make_pairs(self, agents):
        match self.partner:
            case FixedPartner():
                return list(zip(agents[::2], agents[1::2]))
            case RotatingPartner():
                idx = np.random.permutation(len(agents))
                return [
                    (agents[idx[i]], agents[idx[i + 1]])
                    for i in range(0, len(agents), 2)
                ]
            case _:
                raise ValueError(f"Unknown partner hypothesis: {self.partner!r}")

    def _run_pair(self, a1, a2):
        match self.coupling:
            case NoCoupling():
                return a1.trial(), a2.trial()
            case LiveCoupling():
                return run_coupled_trial(a1, a2)
            case _:
                raise ValueError(f"Unknown coupling hypothesis: {self.coupling!r}")

    def _compute_reward(self, a1, a2, side1, side2):
        match self.frame:
            case EgocentricCoordination():
                return float(side1 == side2)
            case AllocentricCoordination():
                return 1.0 if a1.orientation * side1 == a2.orientation * side2 else 0.0
            case _:
                raise ValueError(f"Unknown frame hypothesis: {self.frame!r}")

    def _update(self, agent, side, reward):
        eff = reward - (agent.motor_cost_upper if side > 0 else agent.motor_cost_lower)
        match self.frame:
            case EgocentricCoordination():
                target = side
            case AllocentricCoordination():
                target = agent.orientation * side
            case _:
                raise ValueError(f"Unknown frame hypothesis: {self.frame!r}")
        if target > 0:
            agent.v1 += agent.lr * (max(0.0, eff) - agent.v1)
        else:
            agent.v2 += agent.lr * (max(0.0, eff) - agent.v2)
        agent.kappa += agent.lr_kappa * (reward - agent.kappa)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def _to_dataframe(self, seed, rewards, side_pref, rt):
        """Pack one seed's trial arrays into a polars DataFrame."""
        n_trials = len(rewards)
        cfg = dataclasses.asdict(self._cfg)
        d = {
            "seed":   [seed]     * n_trials,
            "trial":  list(range(n_trials)),
            "reward": rewards.tolist(),
            **{f"rt_{i}":        rt[:, i].tolist()        for i in range(self.n_agents)},
            **{f"side_pref_{i}": side_pref[:, i].tolist() for i in range(self.n_agents)},
            **{k: [v] * n_trials for k, v in cfg.items()},
        }
        return pl.DataFrame(d)

    def run(self, n_trials, seed=0):
        """
        Run a single simulation with fresh agents.

        Returns
        -------
        dict with keys:
          rewards   : (n_trials,)          mean coordination rate per trial
          side_pref : (n_trials, n_agents)  v1 − v2 per agent per trial
          rt        : (n_trials, n_agents)  reaction time (steps) per agent per trial
        """
        np.random.seed(seed)
        agents = self._make_agents()
        rewards   = np.zeros(n_trials)
        side_pref = np.zeros((n_trials, self.n_agents))
        rt        = np.zeros((n_trials, self.n_agents), dtype=int)

        for t in range(n_trials):
            pairs = self._make_pairs(agents)
            trial_reward = 0.0

            for a1, a2 in pairs:
                self._set_orientations(a1, a2)
                side1, side2 = self._run_pair(a1, a2)
                reward = self._compute_reward(a1, a2, side1, side2)
                self._update(a1, side1, reward)
                self._update(a2, side2, reward)
                trial_reward += reward

            rewards[t]   = trial_reward / len(pairs)
            side_pref[t] = [a.v1 - a.v2 for a in agents]
            rt[t]        = [a.last_rt    for a in agents]

        return {"rewards": rewards, "side_pref": side_pref, "rt": rt}

    def run_seeds(self, n_trials, n_seeds, logger=None):
        """
        Run multiple independent seeds.

        Parameters
        ----------
        logger : RunLogger | None
            If provided, each seed's DataFrame is written to CSV before continuing.

        Returns
        -------
        polars DataFrame with one row per (seed, trial).
        Columns: seed, trial, reward, rt_0..rt_n, side_pref_0..side_pref_n, <AgentConfig fields>
        """
        frames = []
        for seed in range(n_seeds):
            data = self.run(n_trials, seed=seed)
            df = self._to_dataframe(seed, data["rewards"], data["side_pref"], data["rt"])
            frames.append(df)
            if logger is not None:
                logger.log(seed, df, self.label)
        return pl.concat(frames)
