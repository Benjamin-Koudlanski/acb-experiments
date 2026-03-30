"""
Coordination Bottleneck Index (CBI) – Section 3.6.

CBI = n_deployed / n* = n · c / a

Interpretation:
  CBI = 1.0  → optimal
  CBI < 1.0  → under-provisioned (room to add agents)
  CBI > 1.0  → over-provisioned (remove agents)
  CBI > 2.0  → critical (likely negative net performance)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from acb.model import optimal_fleet_size


class CBIZone(Enum):
    UNDER_PROVISIONED = "under-provisioned"
    NEAR_OPTIMAL = "near-optimal"
    OVER_PROVISIONED = "over-provisioned"
    CRITICAL = "critical"


@dataclass
class CBIResult:
    """Result of a CBI computation."""

    n_deployed: int
    n_star: int
    cbi: float
    zone: CBIZone
    recommendation: str

    def __str__(self) -> str:
        return (
            f"CBI = {self.cbi:.2f} ({self.zone.value})\n"
            f"  Deployed: {self.n_deployed} agents | Optimal: {self.n_star} agents\n"
            f"  → {self.recommendation}"
        )


def compute_cbi(n_deployed: int, a: float, c: float) -> float:
    """Compute raw CBI value.

    CBI = n_deployed / n* = n_deployed * c / a
    """
    n_star = optimal_fleet_size(a, c)
    return n_deployed / n_star


def interpret_cbi(n_deployed: int, a: float, c: float) -> CBIResult:
    """Compute CBI and return a full diagnostic result.

    Parameters
    ----------
    n_deployed : int
        Current fleet size in production.
    a : float
        Per-agent accuracy gain (Pass@1 from benchmark).
    c : float
        Per-link coordination overhead.

    Returns
    -------
    CBIResult
        Diagnostic object with CBI value, zone, and recommendation.
    """
    n_star = optimal_fleet_size(a, c)
    cbi = n_deployed / n_star

    if cbi > 2.0:
        zone = CBIZone.CRITICAL
        recommendation = (
            f"CRITICAL: fleet is {cbi:.1f}x optimal. Net performance is likely negative. "
            f"Reduce from {n_deployed} to ~{n_star} agents immediately."
        )
    elif cbi > 1.2:
        zone = CBIZone.OVER_PROVISIONED
        recommendation = (
            f"Over-provisioned: {n_deployed - n_star} excess agents are degrading performance. "
            f"Reduce to {n_star} agents."
        )
    elif cbi >= 0.8:
        zone = CBIZone.NEAR_OPTIMAL
        recommendation = f"Near-optimal configuration. No changes needed."
    else:
        zone = CBIZone.UNDER_PROVISIONED
        recommendation = (
            f"Under-provisioned: room to add {n_star - n_deployed} agents for improved performance."
        )

    return CBIResult(
        n_deployed=n_deployed,
        n_star=n_star,
        cbi=cbi,
        zone=zone,
        recommendation=recommendation,
    )
