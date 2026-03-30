"""ACB – Agent Coordination Bound core library."""

from acb.model import (
    acb_performance,
    marginal_gain,
    optimal_fleet_size,
    p_harm,
    p_harm_monte_carlo,
    rho_crit,
    topology_crossover,
)
from acb.cbi import compute_cbi, interpret_cbi

__version__ = "0.1.0"

__all__ = [
    "acb_performance",
    "marginal_gain",
    "optimal_fleet_size",
    "p_harm",
    "p_harm_monte_carlo",
    "rho_crit",
    "topology_crossover",
    "compute_cbi",
    "interpret_cbi",
]
