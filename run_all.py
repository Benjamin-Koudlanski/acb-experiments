#!/usr/bin/env python3
"""
Run all ACB experiments in sequence.

Usage:
    python run_all.py                              # default configs
    python run_all.py --config configs/full_replication.yaml
    python run_all.py --quick                      # fast test with 5 tasks, 2 reps
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from experiments.common import ExperimentConfig, save_results, setup_logging
from experiments.p1_fleet_sizing import run_p1
from experiments.p2_topology_crossover import run_p2
from experiments.p3_rag_diversity import run_p3
from monte_carlo.validate_pharm import run_validation as run_mc_validation

logger = logging.getLogger("acb.run_all")


async def run_all(config: ExperimentConfig, skip_mc: bool = False):
    """Run all experiments and Monte Carlo validation."""

    setup_logging()
    start = time.time()
    output_dir = config.output_dir

    # ── Step 0: Monte Carlo validation (no API needed) ──
    if not skip_mc:
        logger.info("=" * 60)
        logger.info("STEP 0: Monte Carlo P(harm|n) validation")
        logger.info("=" * 60)
        mc_results = run_mc_validation(mc_runs=50_000)
        if not mc_results["all_pass"]:
            logger.error("Monte Carlo validation FAILED. Aborting.")
            sys.exit(1)
        logger.info("Monte Carlo validation PASSED.\n")

    # ── Step 1: P1 — Fleet sizing ──
    logger.info("=" * 60)
    logger.info("STEP 1: P1 — All-to-all fleet sizing on HumanEval")
    logger.info("=" * 60)
    p1_config = ExperimentConfig(
        experiment_name="p1_fleet_sizing",
        fleet_sizes=config.fleet_sizes,
        n_reps=config.n_reps,
        max_tasks=config.max_tasks,
        max_rounds=config.max_rounds,
        seed=config.seed,
        output_dir=output_dir,
        temperature=config.temperature,
        model=config.model,
        backend=config.backend,
    )
    p1_result = await run_p1(p1_config)
    save_results(p1_result, output_dir)

    # ── Step 2: P2 — Topology crossover ──
    logger.info("=" * 60)
    logger.info("STEP 2: P2 — Supervisor vs. all-to-all crossover on MATH")
    logger.info("=" * 60)
    p2_config = ExperimentConfig(
        experiment_name="p2_topology_crossover",
        fleet_sizes=config.fleet_sizes,
        n_reps=config.n_reps,
        max_tasks=config.max_tasks or 200,
        max_rounds=config.max_rounds,
        seed=config.seed,
        output_dir=output_dir,
        temperature=config.temperature,
        model=config.model,
        backend=config.backend,
    )
    p2_result = await run_p2(p2_config)
    save_results(p2_result, output_dir)

    # ── Step 3: P3 — RAG diversity ──
    logger.info("=" * 60)
    logger.info("STEP 3: P3 — Shared vs. isolated RAG context")
    logger.info("=" * 60)
    p3_config = ExperimentConfig(
        experiment_name="p3_rag_diversity",
        fleet_sizes=[1, 3, 6, 9],
        n_reps=config.n_reps,
        max_tasks=config.max_tasks or 200,
        max_rounds=1,
        seed=config.seed,
        output_dir=output_dir,
        temperature=config.temperature,
        model=config.model,
        backend=config.backend,
    )
    p3_result = await run_p3(p3_config)
    save_results(p3_result, output_dir)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("REPLICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"P1 — n* predicted: {p1_result.summary.get('n_star', '?')}, "
        f"empirical peak: {p1_result.summary.get('empirical_peak', '?')}, "
        f"match: {p1_result.summary.get('n_star_matches', '?')}"
    )
    logger.info(f"P2 — P2 falsified: {p2_result.summary.get('p2_falsified', '?')}")
    logger.info(f"P3 — P3 falsified: {p3_result.summary.get('p3_falsified', '?')}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Run all ACB experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 tasks, 2 reps")
    parser.add_argument("--skip-mc", action="store_true", help="Skip Monte Carlo validation")
    parser.add_argument("--output-dir", type=str, default="results/")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    elif args.quick:
        config = ExperimentConfig(
            experiment_name="quick_test",
            fleet_sizes=[1, 2, 3, 5],
            n_reps=2,
            max_tasks=5,
            max_rounds=2,
            output_dir=args.output_dir,
        )
    else:
        config = ExperimentConfig(
            experiment_name="full_replication",
            output_dir=args.output_dir,
        )

    asyncio.run(run_all(config, skip_mc=args.skip_mc))


if __name__ == "__main__":
    main()
