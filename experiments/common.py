"""
Shared experiment utilities.

Common infrastructure for P1, P2, P3 experiments:
  - Configuration loading
  - Result saving with metadata
  - Logging
  - Reproducibility (seed management)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger("acb.experiments")


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""

    experiment_name: str = ""
    fleet_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 10, 15])
    n_reps: int = 50
    max_tasks: int | None = None  # None = all tasks
    max_rounds: int = 3
    seed: int = 42
    output_dir: str = "results/"
    temperature: float = 0.0
    model: str | None = None
    backend: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskResult:
    """Result of running one task with one fleet configuration."""

    task_id: str
    fleet_size: int
    rep: int
    correct: bool
    answer: str = ""
    total_tokens: int = 0
    latency_ms: float = 0.0
    topology: str = ""


@dataclass
class ExperimentResult:
    """Full result of an experiment run."""

    experiment_name: str
    config: dict
    started_at: str = ""
    finished_at: str = ""
    task_results: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def setup_logging(level: str = "INFO"):
    """Configure logging for experiments."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_results(result: ExperimentResult, output_dir: str = "results/"):
    """Save experiment results to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{result.experiment_name}_{timestamp}.json"
    filepath = out_path / filename

    with open(filepath, "w") as f:
        json.dump(asdict(result) if hasattr(result, "__dataclass_fields__") else result.__dict__, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


def compute_pass_at_1(results: list[TaskResult], fleet_size: int) -> float:
    """Compute Pass@1 for a given fleet size across all tasks and reps.

    Pass@1 = mean accuracy across all (task, rep) pairs at this fleet size.
    """
    relevant = [r for r in results if r.fleet_size == fleet_size]
    if not relevant:
        return 0.0
    return sum(1 for r in relevant if r.correct) / len(relevant)


def compute_token_overhead(results: list[TaskResult], n1: int = 1, n2: int = 2) -> float:
    """Compute per-link coordination overhead c from experimental data.

    c = [mean_tokens(n=2) − 2 · mean_tokens(n=1)] / 2

    This follows the measurement protocol in Section 5, Rule 1.
    """
    tok_n1 = [r.total_tokens for r in results if r.fleet_size == n1]
    tok_n2 = [r.total_tokens for r in results if r.fleet_size == n2]

    if not tok_n1 or not tok_n2:
        return 0.0

    mean_n1 = sum(tok_n1) / len(tok_n1)
    mean_n2 = sum(tok_n2) / len(tok_n2)

    # Normalize to [0, 1] scale relative to context window
    # Using 128K context window as reference
    c_tokens = (mean_n2 - 2 * mean_n1) / 2
    c_normalized = c_tokens / 128_000
    return max(0.0, c_normalized)


def compute_acb_metrics(results: list[TaskResult], fleet_sizes: list[int]) -> dict:
    """Compute all ACB metrics from experimental results.

    Returns a dict with:
      - pass_at_1: {fleet_size: accuracy} for each fleet size
      - a: estimated per-agent accuracy gain
      - c: estimated per-link overhead
      - n_star: predicted optimal fleet size
      - I_n: {fleet_size: I(n)} performance curve
      - empirical_peak: fleet size with highest accuracy
    """
    from acb.model import acb_performance, optimal_fleet_size

    pass_rates = {n: compute_pass_at_1(results, n) for n in fleet_sizes}
    a = pass_rates.get(1, 0.0)
    c = compute_token_overhead(results)

    n_star = optimal_fleet_size(a, c) if c > 0 else 1

    I_n = {}
    for n in fleet_sizes:
        I_n[n] = acb_performance(n, a, c)

    empirical_peak = max(pass_rates, key=pass_rates.get) if pass_rates else 1

    return {
        "pass_at_1": pass_rates,
        "a": a,
        "c": c,
        "n_star": n_star,
        "I_n": I_n,
        "empirical_peak": empirical_peak,
        "n_star_matches": abs(empirical_peak - n_star) <= 1,
    }
