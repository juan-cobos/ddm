"""
Run comparisons across the four hypothesis axes:
  1. Partner mode    — FixedPartner vs RotatingPartner
  2. Distance        — SymmetricDistance vs AsymmetricBounds vs AsymmetricMotorCost
  3. Reference frame — EgocentricCoordination vs AllocentricCoordination
  4. Information     — NoCoupling vs LiveCoupling

Each panel isolates one axis while holding the others fixed.
"""

import matplotlib.pyplot as plt
from analysis import plot_axis, run_seeds
from hypotheses import (
    AllocentricCoordination,
    AsymmetricBounds,
    AsymmetricMotorCost,
    EgocentricCoordination,
    FixedPartner,
    LiveCoupling,
    NoCoupling,
    RotatingPartner,
    SymmetricDistance,
)

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
N_AGENTS = 12
N_TRIALS = 100
N_SEEDS = 3
AGENT_KW = dict(lr=0.1, drift_rate=1.0, decay_rate=0.1, noise_std=0.1, dt=1e-2)

# ---------------------------------------------------------------------------
# Panel 1 — Partner mode  (Symmetric, Egocentric, NoCoupling)
# ---------------------------------------------------------------------------
panel1 = {
    "Fixed": run_seeds(
        FixedPartner(),
        SymmetricDistance(),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
    "Rotating": run_seeds(
        RotatingPartner(),
        SymmetricDistance(),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
}

# ---------------------------------------------------------------------------
# Panel 2 — Distance  (Fixed, Egocentric, NoCoupling)
# ---------------------------------------------------------------------------
panel2 = {
    "Symmetric": run_seeds(
        FixedPartner(),
        SymmetricDistance(),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
    "AsymmetricBounds": run_seeds(
        FixedPartner(),
        AsymmetricBounds(1.0, 0.6),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
    "AsymmetricCost": run_seeds(
        FixedPartner(),
        AsymmetricMotorCost(cost=0.4),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
}

# ---------------------------------------------------------------------------
# Panel 3 — Reference frame  (Fixed, AsymmetricBounds, NoCoupling)
# ---------------------------------------------------------------------------
panel3 = {
    "Egocentric": run_seeds(
        FixedPartner(),
        AsymmetricBounds(1.0, 0.6),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
    "Allocentric": run_seeds(
        FixedPartner(),
        AsymmetricBounds(1.0, 0.6),
        AllocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
}

# ---------------------------------------------------------------------------
# Panel 4 — Information  (Fixed, Symmetric, Egocentric)
# ---------------------------------------------------------------------------
panel4 = {
    "NoCoupling": run_seeds(
        FixedPartner(),
        SymmetricDistance(),
        EgocentricCoordination(),
        NoCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
    "LiveCoupling": run_seeds(
        FixedPartner(),
        SymmetricDistance(),
        EgocentricCoordination(),
        LiveCoupling(),
        n_agents=N_AGENTS,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        agent_kwargs=AGENT_KW,
    ),
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)

plot_axis(axes[0], panel1, "Partner mode\n(Symmetric · Ego · NoCoupling)")
plot_axis(axes[1], panel2, "Distance asymmetry\n(Fixed · Ego · NoCoupling)")
plot_axis(axes[2], panel3, "Reference frame\n(Fixed · AsymBounds · NoCoupling)")
plot_axis(axes[3], panel4, "Information coupling\n(Fixed · Symmetric · Ego)")

for ax in axes[1:]:
    ax.set_ylabel("")

fig.suptitle("RLDDM Coordination — Four Hypothesis Axes", fontsize=15, y=1.01)
fig.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
plt.show()
