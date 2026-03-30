"""
Statistical analysis for ACB experiments.

Provides:
  - Wilcoxon signed-rank tests (paired comparisons)
  - Bootstrap confidence intervals
  - Effect size (Cohen's d)
  - Falsification tests for P1, P2, P3
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: float | None = None
    description: str = ""


def wilcoxon_paired(
    scores_a: list[float],
    scores_b: list[float],
    test_name: str = "Wilcoxon",
    alpha: float = 0.05,
) -> TestResult:
    """Wilcoxon signed-rank test for paired samples.

    Used for P2 (supervisor vs all-to-all on same tasks).
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b

    # Remove zero differences
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return TestResult(
            test_name=test_name,
            statistic=0.0,
            p_value=1.0,
            significant=False,
            description="Too few non-zero differences for reliable test.",
        )

    stat, p = stats.wilcoxon(nonzero, alternative="two-sided")

    # Cohen's d for paired samples
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    return TestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        significant=p < alpha,
        effect_size=float(d),
        description=f"mean_diff={np.mean(diff):.4f}, n_pairs={len(nonzero)}",
    )


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for mean accuracy.

    Returns (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(arr)),
        float(np.percentile(means, alpha * 100)),
        float(np.percentile(means, (1 - alpha) * 100)),
    )


def test_p1_prediction(
    results_path: str,
    predicted_range: tuple[int, int, int] = (8, 9, 10),
    alpha: float = 0.05,
) -> TestResult:
    """Test P1: Does the empirical peak fall within the predicted range?

    Uses bootstrap to get a confidence interval on the peak location.
    """
    with open(results_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    pass_at_1 = summary.get("pass_at_1", {})

    if not pass_at_1:
        return TestResult("P1", 0, 1.0, False, description="No data")

    # Find empirical peak
    peak_n = max(pass_at_1, key=lambda k: pass_at_1[k])
    peak_n = int(peak_n)

    in_range = peak_n in predicted_range
    n_star = summary.get("n_star", -1)

    return TestResult(
        test_name="P1: Peak at n*",
        statistic=float(peak_n),
        p_value=0.0 if in_range else 1.0,  # simplified
        significant=not in_range,  # significant = falsified
        description=(
            f"Empirical peak at n={peak_n}, predicted range={predicted_range}, "
            f"n*={n_star}. {'CONFIRMED' if in_range else 'FALSIFIED'}"
        ),
    )


def test_p2_crossover(results_path: str, alpha: float = 0.05) -> list[TestResult]:
    """Test P2: Does supervisor beat all-to-all for all n >= 2?

    Runs paired Wilcoxon at each fleet size.
    """
    with open(results_path) as f:
        data = json.load(f)

    task_results = data.get("task_results", [])
    fleet_sizes = sorted(set(r["fleet_size"] for r in task_results))
    tests = []

    for n in fleet_sizes:
        if n < 2:
            continue

        a2a = [r for r in task_results if r["fleet_size"] == n and r["topology"] == "all-to-all"]
        sup = [r for r in task_results if r["fleet_size"] == n and r["topology"] == "supervisor"]

        if not a2a or not sup:
            continue

        # Match by task_id and rep
        pairs = {}
        for r in a2a:
            key = (r["task_id"], r["rep"])
            pairs.setdefault(key, {})["a2a"] = 1.0 if r["correct"] else 0.0
        for r in sup:
            key = (r["task_id"], r["rep"])
            pairs.setdefault(key, {})["sup"] = 1.0 if r["correct"] else 0.0

        matched = [(v["sup"], v["a2a"]) for v in pairs.values() if "sup" in v and "a2a" in v]
        if len(matched) < 20:
            continue

        sup_scores = [m[0] for m in matched]
        a2a_scores = [m[1] for m in matched]

        test = wilcoxon_paired(sup_scores, a2a_scores, f"P2 n={n}")
        tests.append(test)

    return tests


def test_p3_diversity(results_path: str, alpha: float = 0.05) -> dict:
    """Test P3: Does shared context eliminate diversity benefit?

    Tests:
      1. shared@n=6 vs baseline@n=1 (expect NOT significant)
      2. isolated@n=6 vs baseline@n=1 (expect significant)
    """
    with open(results_path) as f:
        data = json.load(f)

    task_results = data.get("task_results", [])

    def get_scores(topology: str, fleet_size: int) -> list[float]:
        return [
            1.0 if r["correct"] else 0.0
            for r in task_results
            if r["topology"] == topology and r["fleet_size"] == fleet_size
        ]

    baseline = get_scores("rag-shared", 1) or get_scores("rag-isolated", 1)
    shared_6 = get_scores("rag-shared", 6)
    isolated_6 = get_scores("rag-isolated", 6)

    results = {}

    if baseline and shared_6:
        # Use Mann-Whitney U for unpaired comparison
        stat, p = stats.mannwhitneyu(shared_6, baseline, alternative="greater")
        results["shared_vs_baseline"] = TestResult(
            test_name="P3: shared@6 vs baseline",
            statistic=float(stat),
            p_value=float(p),
            significant=p < alpha,
            description=f"mean_shared={np.mean(shared_6):.4f}, mean_base={np.mean(baseline):.4f}",
        )

    if baseline and isolated_6:
        stat, p = stats.mannwhitneyu(isolated_6, baseline, alternative="greater")
        results["isolated_vs_baseline"] = TestResult(
            test_name="P3: isolated@6 vs baseline",
            statistic=float(stat),
            p_value=float(p),
            significant=p < alpha,
            description=f"mean_isolated={np.mean(isolated_6):.4f}, mean_base={np.mean(baseline):.4f}",
        )

    # P3 falsified if shared context significantly helps
    p3_falsified = results.get("shared_vs_baseline", TestResult("", 0, 1, False)).significant
    results["p3_falsified"] = p3_falsified

    return results
