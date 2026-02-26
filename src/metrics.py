"""
Metrics for RLDDM coordination experiments.

All functions operate on numpy arrays extracted from a logged DataFrame.
They accept either a single seed's data or the mean across seeds — the
caller decides how to aggregate.

Notation
--------
rewards   : (n_trials,)           mean coordination rate per trial  [0, 1]
side_pref : (n_trials, n_agents)  v1 − v2 per agent per trial
rt        : (n_trials, n_agents)  reaction time in steps per agent per trial

Extracting arrays from a polars DataFrame df
--------------------------------------------
    rewards   = df["reward"].to_numpy()
    rt        = df.select(cs.starts_with("rt_")).to_numpy()
    side_pref = df.select(cs.starts_with("side_pref_")).to_numpy()
"""

import numpy as np

# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def norm_reward(rewards):
    """
    Normalize coordination rate relative to chance (0.5).
    Returns (n_trials,).  0 = chance, 1 = perfect coordination.
    """
    return (rewards - 0.5) / 0.5


def final_accuracy(rewards, last_k=50):
    """Mean coordination rate over the last k trials."""
    return float(rewards[-last_k:].mean())


def convergence_trial(rewards, threshold=0.75, window=10):
    """
    First trial where the rolling-mean coordination rate exceeds threshold.
    Returns None if the threshold is never reached.
    """
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    above = np.where(smoothed > threshold)[0]
    return int(above[0]) if len(above) else None


# ---------------------------------------------------------------------------
# Side preference
# ---------------------------------------------------------------------------


def side_preference(side_pref, last_k=50):
    """
    Mean v1 − v2 per agent over the last k trials.
    Returns (n_agents,).  Positive = prefers upper bound, negative = lower bound.
    """
    return side_pref[-last_k:].mean(axis=0)


def preference_consensus(side_pref, last_k=50):
    """
    Variance of mean (v1 − v2) across agents over the last k trials.
    Low  → all agents converge to the same side (global consensus).
    High → agents split between sides.
    """
    return float(side_pref[-last_k:].mean(axis=0).var())


def pair_alignment(side_pref, last_k=50):
    """
    Mean Pearson correlation of (v1 − v2) between fixed partners (0-1, 2-3, …)
    over the last k trials.
    Positive → both partners prefer the same side  (egocentric pattern).
    Negative → partners prefer opposite sides       (allocentric pattern).
    """
    prefs = side_pref[-last_k:]
    n_agents = prefs.shape[1]
    corrs = [
        np.corrcoef(prefs[:, i], prefs[:, i + 1])[0, 1] for i in range(0, n_agents, 2)
    ]
    return float(np.nanmean(corrs))


# ---------------------------------------------------------------------------
# Reaction time
# ---------------------------------------------------------------------------


def mean_rt(rt):
    """
    Mean RT across agents per trial.
    Returns (n_trials,).
    """
    return rt.mean(axis=1)


def rt_asynchrony(rt):
    """
    |ti − tj| for fixed pairs (0-1, 2-3, …) per trial.
    Captures how asynchronously paired agents make their decisions.
    Returns (n_trials, n_pairs).
    """
    n_agents = rt.shape[1]
    return np.column_stack(
        [np.abs(rt[:, i] - rt[:, i + 1]) for i in range(0, n_agents, 2)]
    )


# ---------------------------------------------------------------------------
# Speed-accuracy tradeoffs
# ---------------------------------------------------------------------------


def speed_accuracy_rt(rewards, rt):
    """
    Pearson correlation between norm_reward and mean RT across trials.

    Interpretation: as learning progresses, value polarisation increases drift
    and reduces RT while also increasing accuracy — both driven by the same
    learning signal.  A negative correlation is therefore expected: faster
    trials are more accurate.
    """
    return float(np.corrcoef(norm_reward(rewards), mean_rt(rt))[0, 1])


def speed_accuracy_async(rewards, rt):
    """
    Pearson correlation between norm_reward and mean |ti − tj| across trials.

    A negative correlation means: trials where the two partners finish far
    apart in time are less likely to produce successful coordination.
    Most diagnostic for the LiveCoupling vs NoCoupling axis.
    """
    return float(
        np.corrcoef(norm_reward(rewards), rt_asynchrony(rt).mean(axis=1))[0, 1]
    )
