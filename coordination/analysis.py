"""
Experiment repetition across seeds and plotting utilities.
"""

import numpy as np
from hypotheses import run_experiment


def run_seeds(
    partner,
    distance,
    frame,
    information,
    n_agents=2,
    n_trials=300,
    n_seeds=10,
    agent_kwargs=None,
):
    all_rewards, all_side_pref = [], []
    for seed in range(n_seeds):
        rewards, side_pref = run_experiment(
            partner,
            distance,
            frame,
            information,
            n_agents=n_agents,
            n_trials=n_trials,
            seed=seed,
            agent_kwargs=agent_kwargs,
        )
        all_rewards.append(rewards)
        all_side_pref.append(side_pref)
    r = np.array(all_rewards)
    return {
        "mean": r.mean(0),
        "sem": r.std(0) / np.sqrt(n_seeds),
        "side_pref": np.array(all_side_pref).mean(0),
    }


def moving_avg(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_axis(ax, results, title, window=20):
    for label, res in results.items():
        x = moving_avg(res["mean"], window)
        sem = moving_avg(res["sem"], window)
        trials = np.arange(len(x))
        (line,) = ax.plot(trials, x, label=label, linewidth=1.8)
        ax.fill_between(trials, x - sem, x + sem, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Coordination rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
