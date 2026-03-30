"""
Monte Carlo validation of the closed-form P(harm|n).

Reproduces Table 3 from the paper: validates the analytical integral
against 50,000-run Monte Carlo simulation across fleet sizes.

Usage:
    python -m monte_carlo.validate_pharm --runs 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from acb.model import p_harm, p_harm_monte_carlo

logger = logging.getLogger("acb.monte_carlo")

# Default parameters calibrated to reproduce paper Table 3 (Section 3.2).
# These represent a scenario with moderate accuracy and non-trivial overhead.
# Users can override with --mu-a, --sigma-a, --mu-c for their own setup.
DEFAULT_MU_A = 0.19
DEFAULT_SIGMA_A = 0.07
DEFAULT_MU_C = 0.18
DEFAULT_FLEET_SIZES = [1, 2, 3, 5, 7, 10, 15]


def run_validation(
    mu_a: float = DEFAULT_MU_A,
    sigma_a: float = DEFAULT_SIGMA_A,
    mu_c: float = DEFAULT_MU_C,
    fleet_sizes: list[int] | None = None,
    mc_runs: int = 50_000,
    seed: int = 42,
) -> dict:
    """Run the full P(harm|n) validation.

    Computes both the closed-form integral and Monte Carlo estimate
    for each fleet size, and reports the absolute error.

    Returns
    -------
    dict
        Validation results with per-fleet-size comparisons.
    """
    fleet_sizes = fleet_sizes or DEFAULT_FLEET_SIZES
    results = []
    max_error = 0.0

    logger.info("=" * 72)
    logger.info("P(harm|n) Validation: Closed-Form vs. Monte Carlo")
    logger.info(f"Parameters: mu_a={mu_a}, sigma_a={sigma_a}, mu_c={mu_c}")
    logger.info(f"Monte Carlo runs: {mc_runs:,}, seed: {seed}")
    logger.info("=" * 72)
    logger.info(f"{'Fleet n':>8} {'Formula':>12} {'MC (50K)':>12} {'Abs Error':>12} {'Status':>10}")
    logger.info("-" * 72)

    for n in fleet_sizes:
        # Closed-form
        t0 = time.perf_counter()
        p_formula = p_harm(n, mu_a, sigma_a, mu_c)
        t_formula = (time.perf_counter() - t0) * 1000

        # Monte Carlo
        t0 = time.perf_counter()
        p_mc = p_harm_monte_carlo(n, mu_a, sigma_a, mu_c, runs=mc_runs, seed=seed)
        t_mc = (time.perf_counter() - t0) * 1000

        error = abs(p_formula - p_mc)
        max_error = max(max_error, error)
        status = "PASS" if error < 0.01 else "FAIL"

        row = {
            "n": n,
            "p_formula": round(p_formula, 4),
            "p_monte_carlo": round(p_mc, 4),
            "abs_error": round(error, 4),
            "formula_ms": round(t_formula, 2),
            "mc_ms": round(t_mc, 2),
            "status": status,
        }
        results.append(row)

        marker = " <-- 50% threshold" if n == 2 else ""
        logger.info(
            f"{n:>8} {p_formula:>12.4f} {p_mc:>12.4f} {error:>12.4f} "
            f"{status:>10}{marker}"
        )

    logger.info("-" * 72)
    logger.info(f"Maximum absolute error: {max_error:.4f}")
    all_pass = all(r["status"] == "PASS" for r in results)
    logger.info(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # Additional insight: find the 50% crossing point
    logger.info("Key thresholds:")
    for r in results:
        if r["p_formula"] >= 0.5:
            logger.info(f"  P(harm) crosses 50% at n={r['n']} (P={r['p_formula']:.4f})")
            break
    for r in results:
        if r["p_formula"] >= 0.9:
            logger.info(f"  P(harm) crosses 90% at n={r['n']} (P={r['p_formula']:.4f})")
            break

    return {
        "parameters": {
            "mu_a": mu_a,
            "sigma_a": sigma_a,
            "mu_c": mu_c,
            "mc_runs": mc_runs,
            "seed": seed,
        },
        "results": results,
        "max_error": round(max_error, 4),
        "all_pass": all_pass,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate P(harm|n) closed-form against Monte Carlo"
    )
    parser.add_argument("--mu-a", type=float, default=DEFAULT_MU_A, help="Mean accuracy gain")
    parser.add_argument("--sigma-a", type=float, default=DEFAULT_SIGMA_A, help="Std dev accuracy")
    parser.add_argument("--mu-c", type=float, default=DEFAULT_MU_C, help="Mean overhead")
    parser.add_argument("--runs", type=int, default=50_000, help="Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument(
        "--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES,
        help="Fleet sizes to validate",
    )
    args = parser.parse_args()

    results = run_validation(
        mu_a=args.mu_a,
        sigma_a=args.sigma_a,
        mu_c=args.mu_c,
        fleet_sizes=args.fleet_sizes,
        mc_runs=args.runs,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
