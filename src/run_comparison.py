"""
Run all four hypothesis-axis comparisons and log results to outputs/.

Each condition produces one CSV per seed:
    outputs/<env_label>_seed<n>.csv

Load offline for metric computation and plotting.
"""

from env import CoordinationEnv
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
from logger import RunLogger

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
N_AGENTS = 12
N_TRIALS = 300
N_SEEDS = 3
CFG = AgentConfig(lr=0.1, drift_rate=1.0, decay_rate=0.1, noise_std=0.1)

logger = RunLogger("outputs")


def run(partner, distance, frame, coupling):
    return CoordinationEnv(
        N_AGENTS, partner, distance, frame, coupling, config=CFG
    ).run_seeds(N_TRIALS, N_SEEDS, logger=logger)


# ---------------------------------------------------------------------------
# Axis 1 — Partner mode  (Symmetric · Egocentric · NoCoupling)
# ---------------------------------------------------------------------------
run(FixedPartner(), SymmetricDistance(), EgocentricCoordination(), NoCoupling())
run(RotatingPartner(), SymmetricDistance(), EgocentricCoordination(), NoCoupling())

# ---------------------------------------------------------------------------
# Axis 2 — Distance  (Fixed · Egocentric · NoCoupling)
# ---------------------------------------------------------------------------
run(FixedPartner(), SymmetricDistance(), EgocentricCoordination(), NoCoupling())
run(FixedPartner(), AsymmetricBounds(1.0, 0.6), EgocentricCoordination(), NoCoupling())
run(FixedPartner(), MotorCost(cost_lower=0.4), EgocentricCoordination(), NoCoupling())

# ---------------------------------------------------------------------------
# Axis 3 — Reference frame  (Fixed · AsymmetricBounds · NoCoupling)
# ---------------------------------------------------------------------------
run(FixedPartner(), AsymmetricBounds(1.0, 0.6), EgocentricCoordination(), NoCoupling())
run(FixedPartner(), AsymmetricBounds(1.0, 0.6), AllocentricCoordination(), NoCoupling())

# ---------------------------------------------------------------------------
# Axis 4 — Information coupling  (Fixed · Symmetric · Egocentric)
# ---------------------------------------------------------------------------
run(FixedPartner(), SymmetricDistance(), EgocentricCoordination(), NoCoupling())
run(FixedPartner(), SymmetricDistance(), EgocentricCoordination(), LiveCoupling())
