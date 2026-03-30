"""
Core ACB mathematical model.

Implements all analytical results from the paper:
  - I(n)  = n·a − c·n(n−1)/2                         (ACB model)
  - ΔI(n) = a − c·n                                   (marginal gain)
  - n*    = ⌈a/c⌉                                     (optimal fleet size)
  - P(harm|n) closed-form integral                     (Section 3.2)
  - ρ_crit = 1 − 2c/(λσ²)                             (Section 3.3)
  - n_crossover = ⌈2·c_s/c_a⌉ + 1                     (Section 3.5)
  - n*(β) = (a/(c·α·β))^{1/(β−1)}  for β > 1          (Section 3.4)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import integrate
from scipy.stats import norm


# ── Section 3.1: ACB Model ────────────────────────────────────────────


def acb_performance(n: int, a: float, c: float) -> float:
    """Aggregate network performance I(n) = n·a − c·n(n−1)/2.

    Parameters
    ----------
    n : int
        Fleet size (number of agents). Must be >= 1.
    a : float
        Per-agent accuracy gain (Pass@1).
    c : float
        Per-link token coordination overhead.

    Returns
    -------
    float
        Net performance I(n).
    """
    if n < 1:
        raise ValueError(f"Fleet size must be >= 1, got {n}")
    return n * a - c * n * (n - 1) / 2


def marginal_gain(n: int, a: float, c: float) -> float:
    """Marginal value of the (n+1)-th agent: ΔI(n) = a − c·n.

    Returns the performance change from adding one agent to a fleet of size n.
    Positive means the next agent helps; negative means it hurts.
    """
    return a - c * n


def optimal_fleet_size(a: float, c: float) -> int:
    """Optimal fleet size n* = ⌈a/c⌉.

    Parameters
    ----------
    a : float
        Per-agent accuracy gain.
    c : float
        Per-link coordination overhead.

    Returns
    -------
    int
        n*, the fleet size that maximizes I(n). Minimum 1.
    """
    if c <= 0:
        raise ValueError(f"Overhead c must be > 0, got {c}")
    if a <= 0:
        return 1
    return max(1, math.ceil(a / c))


def performance_curve(a: float, c: float, n_max: int = 20) -> list[tuple[int, float]]:
    """Compute I(n) for n = 1..n_max.

    Returns
    -------
    list of (n, I(n)) tuples.
    """
    return [(n, acb_performance(n, a, c)) for n in range(1, n_max + 1)]


# ── Section 3.2: Closed-Form Degradation Probability ─────────────────


def p_harm(n: int, mu_a: float, sigma_a: float, mu_c: float) -> float:
    """Closed-form probability that agent (n+1) degrades performance.

    P(ΔI(n) < 0) = ∫₀^∞ Φ((cn − μ_a)/σ_a) · (1/μ_c) · exp(−c/μ_c) dc

    where:
      - a ~ N(μ_a, σ_a²)   [CLT on aggregated pass rates]
      - c ~ Exp(1/μ_c)      [non-negative overhead with occasional spikes]
      - Φ is the standard normal CDF

    Parameters
    ----------
    n : int
        Current fleet size (we ask: does agent n+1 hurt?).
    mu_a : float
        Mean per-agent accuracy gain.
    sigma_a : float
        Std dev of per-agent accuracy gain.
    mu_c : float
        Mean per-link overhead (rate parameter = 1/mu_c).

    Returns
    -------
    float
        P(harm | n) in [0, 1].
    """
    if sigma_a <= 0:
        raise ValueError(f"sigma_a must be > 0, got {sigma_a}")
    if mu_c <= 0:
        raise ValueError(f"mu_c must be > 0, got {mu_c}")

    def integrand(c_val: float) -> float:
        z = (c_val * n - mu_a) / sigma_a
        phi = norm.cdf(z)
        exp_pdf = (1.0 / mu_c) * math.exp(-c_val / mu_c)
        return phi * exp_pdf

    result, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    return float(np.clip(result, 0.0, 1.0))


def p_harm_curve(
    n_values: list[int],
    mu_a: float,
    sigma_a: float,
    mu_c: float,
) -> list[tuple[int, float]]:
    """Compute P(harm|n) for a list of fleet sizes."""
    return [(n, p_harm(n, mu_a, sigma_a, mu_c)) for n in n_values]


def p_harm_monte_carlo(
    n: int,
    mu_a: float,
    sigma_a: float,
    mu_c: float,
    runs: int = 50_000,
    seed: int | None = 42,
) -> float:
    """Monte Carlo estimate of P(harm|n) for validation.

    Draws `runs` samples of (a, c) and counts how often ΔI(n) < 0.
    """
    rng = np.random.default_rng(seed)
    a_samples = rng.normal(mu_a, sigma_a, size=runs)
    c_samples = rng.exponential(mu_c, size=runs)
    delta = a_samples - c_samples * n
    return float(np.mean(delta < 0))


# ── Section 3.3: Diversity Collapse Threshold ─────────────────────────


def rho_crit(c: float, sigma_sq: float, lam: float = 1.0) -> float:
    """Diversity collapse threshold: ρ_crit = 1 − 2c/(λσ²).

    Above ρ_crit, no fleet size n improves over a single agent.

    Parameters
    ----------
    c : float
        Per-link coordination overhead.
    sigma_sq : float
        Variance of agent output quality.
    lam : float
        Normalizing constant (default 1.0).

    Returns
    -------
    float
        ρ_crit. If negative, aggregation is always beneficial.
    """
    if lam * sigma_sq == 0:
        return float("-inf")
    return 1.0 - 2.0 * c / (lam * sigma_sq)


# ── Section 3.4: Generalized Topology ────────────────────────────────


def optimal_fleet_topology(
    a: float, c: float, beta: float, alpha: float = 1.0
) -> float:
    """Generalized optimal fleet size for topology exponent β.

    n*(β) = (a / (c·α·β))^{1/(β−1)}   for β > 1

    For β = 1 (pure supervisor), returns inf (linear scaling).
    For β = 2 (all-to-all), reduces to a/c.
    """
    if beta <= 1.0:
        return float("inf")
    base = a / (c * alpha * beta)
    if base <= 0:
        return 1.0
    return base ** (1.0 / (beta - 1.0))


# ── Section 3.5: Topology Crossover Theorem ──────────────────────────


def topology_crossover(c_supervisor: float, c_all2all: float) -> int:
    """Fleet size at which supervisor routing first outperforms all-to-all.

    n_crossover = ⌈2·c_s/c_a⌉ + 1

    Parameters
    ----------
    c_supervisor : float
        Per-link overhead for supervisor topology.
    c_all2all : float
        Per-link overhead for all-to-all topology.

    Returns
    -------
    int
        n_crossover. Supervisor wins for all n >= this value.
    """
    if c_all2all <= 0:
        raise ValueError(f"c_all2all must be > 0, got {c_all2all}")
    return math.ceil(2.0 * c_supervisor / c_all2all) + 1


# ── Section 3.7: Submodular Performance Function ─────────────────────


def is_submodular_check(
    agents: list[tuple[str, float]],
    c: float,
    cross_penalty: float = 1.0,
) -> bool:
    """Empirically verify submodularity of I(S) on a small agent pool.

    Checks the diminishing returns property:
    I(S ∪ {j}) − I(S) >= I(T ∪ {j}) − I(T)  for all S ⊆ T, j ∉ T

    This is a sanity check on small pools; not a proof.
    """
    from itertools import combinations

    names = [name for name, _ in agents]
    n = len(agents)

    def I_set(indices: frozenset) -> float:
        k = len(indices)
        if k == 0:
            return 0.0
        total_a = sum(agents[i][1] for i in indices)
        links = k * (k - 1) / 2
        return total_a - c * cross_penalty * links

    for size_s in range(n):
        for s_combo in combinations(range(n), size_s):
            S = frozenset(s_combo)
            remaining = set(range(n)) - S
            for size_extra in range(1, len(remaining) + 1):
                for extra_combo in combinations(remaining, size_extra):
                    T = S | frozenset(extra_combo)
                    outside_T = set(range(n)) - T
                    for j in outside_T:
                        gain_S = I_set(S | {j}) - I_set(S)
                        gain_T = I_set(T | {j}) - I_set(T)
                        if gain_S < gain_T - 1e-10:
                            return False
    return True
