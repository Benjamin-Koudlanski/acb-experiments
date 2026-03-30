"""
Publication-quality plots for ACB experiments.

Generates figures matching the paper's style:
  - Figure 1: I(n) performance curves + phase diagram + retrospective
  - Figure 2: Scaffold overhead + CBI diagnostic
  - Figure 3: P(harm|n) + greedy selection
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from acb.model import acb_performance, p_harm, p_harm_monte_carlo, optimal_fleet_size
from acb.cbi import compute_cbi

# Paper-consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_performance_curves(
    configs: list[tuple[str, float, float]],
    n_max: int = 20,
    output: str | None = None,
):
    """Plot I(n) for multiple (label, a, c) configurations.

    Reproduces Figure 1A.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = np.arange(1, n_max + 1)

    for label, a, c in configs:
        I_vals = [acb_performance(n, a, c) for n in ns]
        n_star = optimal_fleet_size(a, c)
        ax.plot(ns, I_vals, "-o", markersize=4, label=f"{label} (n*={n_star})")
        ax.axvline(n_star, linestyle=":", alpha=0.3)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Fleet size n")
    ax.set_ylabel("Aggregate performance I(n)")
    ax.set_title("ACB Performance Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output:
        fig.savefig(output)
        print(f"Saved: {output}")
    return fig


def plot_pharm_validation(
    mu_a: float = 0.72,
    sigma_a: float = 0.15,
    mu_c: float = 0.082,
    mc_runs: int = 50_000,
    output: str | None = None,
):
    """Plot P(harm|n): formula vs Monte Carlo.

    Reproduces Figure 3A.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = list(range(1, 16))

    p_formula = [p_harm(n, mu_a, sigma_a, mu_c) for n in ns]
    p_mc = [p_harm_monte_carlo(n, mu_a, sigma_a, mu_c, runs=mc_runs) for n in ns]

    ax.plot(ns, p_formula, "b-", linewidth=2, label="Closed-form (analytical)")
    ax.scatter(ns, p_mc, color="red", s=40, zorder=5, label=f"Monte Carlo ({mc_runs:,} runs)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="90% threshold")

    ax.set_xlabel("Fleet size n")
    ax.set_ylabel("P(harm | n)")
    ax.set_title("Degradation Probability: Analytical vs. Monte Carlo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    if output:
        fig.savefig(output)
        print(f"Saved: {output}")
    return fig


def plot_scaffold_overhead(
    gamma: float = 620,
    context_window: int = 128_000,
    output: str | None = None,
):
    """Plot context window consumption by scaffolding.

    Reproduces Figure 2A.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = np.arange(2, 21)
    channels = ns * (ns - 1) / 2
    overhead_pct = (gamma * channels / context_window) * 100

    colors = ["#2ecc71" if p < 20 else "#f39c12" if p < 40 else "#e74c3c" for p in overhead_pct]

    ax.bar(ns, overhead_pct, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(40, color="red", linestyle="--", linewidth=1.5, label="40% threshold (Liu et al.)")
    ax.set_xlabel("Fleet size n")
    ax.set_ylabel("Context window consumed by scaffolding (%)")
    ax.set_title("Inter-Agent Scaffold Overhead (All-to-All)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if output:
        fig.savefig(output)
        print(f"Saved: {output}")
    return fig


def plot_cbi_diagnostic(
    deployments: list[tuple[str, int, int]],
    output: str | None = None,
):
    """Plot CBI for a set of deployments.

    Reproduces Figure 2B.

    Parameters
    ----------
    deployments : list of (label, n_deployed, n_star)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels = [d[0] for d in deployments]
    cbis = [d[1] / d[2] for d in deployments]

    colors = []
    for cbi in cbis:
        if cbi > 2.0:
            colors.append("#e74c3c")
        elif cbi > 1.2:
            colors.append("#f39c12")
        elif cbi >= 0.8:
            colors.append("#2ecc71")
        else:
            colors.append("#3498db")

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, cbis, color=colors, edgecolor="white")
    ax.axvline(1.0, color="green", linestyle="-", linewidth=2, alpha=0.7, label="Optimal (CBI=1)")
    ax.axvline(2.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Critical (CBI=2)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("CBI (n_deployed / n*)")
    ax.set_title("Coordination Bottleneck Index – Deployment Diagnostic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    for i, cbi in enumerate(cbis):
        ax.text(cbi + 0.05, i, f"{cbi:.2f}", va="center", fontsize=9)

    if output:
        fig.savefig(output)
        print(f"Saved: {output}")
    return fig


def plot_experiment_results(results_path: str, output_dir: str = "figures/"):
    """Generate all plots from an experiment results JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    experiment = data.get("experiment_name", "")
    summary = data.get("summary", {})

    if "pass_at_1" in summary:
        # P1-style: plot accuracy vs fleet size
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pass_at_1 = summary["pass_at_1"]
        ns = sorted(int(k) for k in pass_at_1.keys())
        accs = [pass_at_1[str(n)] for n in ns]

        ax.plot(ns, accs, "bo-", markersize=6, label="Empirical Pass@1")
        if "n_star" in summary:
            ax.axvline(summary["n_star"], color="red", linestyle="--",
                       label=f"n* = {summary['n_star']}")
        if "empirical_peak" in summary:
            peak = summary["empirical_peak"]
            ax.axvline(peak, color="green", linestyle=":",
                       label=f"Empirical peak = {peak}")

        ax.set_xlabel("Fleet size n")
        ax.set_ylabel("Pass@1 Accuracy")
        ax.set_title(f"{experiment}: Accuracy vs. Fleet Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / f"{experiment}_accuracy.png")

    if "all_to_all" in summary and "supervisor" in summary:
        # P2-style: compare topologies
        fig, ax = plt.subplots(figsize=(7, 4.5))
        a2a = summary["all_to_all"]
        sup = summary["supervisor"]
        ns = sorted(int(k) for k in a2a.keys())

        ax.plot(ns, [a2a[str(n)] for n in ns], "ro-", label="All-to-all")
        ax.plot(ns, [sup[str(n)] for n in ns], "bs-", label="Supervisor")
        ax.set_xlabel("Fleet size n")
        ax.set_ylabel("Pass@1 Accuracy")
        ax.set_title(f"{experiment}: Topology Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / f"{experiment}_topology.png")

    if "shared" in summary and "isolated" in summary:
        # P3-style: compare context modes
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shared = summary["shared"]
        isolated = summary["isolated"]
        ns = sorted(int(k) for k in shared.keys())

        ax.plot(ns, [shared[str(n)] for n in ns], "ro-", label="Shared context")
        ax.plot(ns, [isolated[str(n)] for n in ns], "bs-", label="Isolated context")
        ax.set_xlabel("Fleet size n")
        ax.set_ylabel("Pass@1 Accuracy")
        ax.set_title(f"{experiment}: Shared vs. Isolated Context")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / f"{experiment}_context.png")

    print(f"Plots saved to {out}/")
