"""
Experiment P1 – Performance peak at n* = ⌈a/c⌉.

Prediction: On HumanEval with gpt-4o-mini (a≈0.72, c≈0.082, n*≈9),
the performance composite I(n) peaks within {8, 9, 10} in an all-to-all
topology. Falsified if peak is outside this range at p < 0.05.

Protocol:
  1. For each fleet size n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15}:
     a. For each rep ∈ {1..50}:
        i.  For each HumanEval task:
            - Deploy n agents in all-to-all topology
            - Run 3 rounds of coordination
            - Record: final answer, correctness, total tokens
  2. Compute Pass@1(n) for each fleet size
  3. Measure a = Pass@1(n=1), c from token overhead
  4. Compute n* = ⌈a/c⌉, compare to empirical peak
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from acb.llm_backend import LLMBackend, make_llm_fn
from acb.topologies import AllToAll
from benchmarks.humaneval import (
    evaluate_solution,
    format_task_prompt,
    load_tasks,
)
from experiments.common import (
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
    compute_acb_metrics,
    save_results,
    setup_logging,
)

logger = logging.getLogger("acb.p1")

DEFAULT_FLEET_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]


async def run_p1(config: ExperimentConfig) -> ExperimentResult:
    """Execute experiment P1.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    ExperimentResult
        Full results with per-task data and summary metrics.
    """
    setup_logging()

    fleet_sizes = config.fleet_sizes or DEFAULT_FLEET_SIZES
    n_reps = config.n_reps
    topology = AllToAll()

    # Load benchmark
    logger.info("Loading HumanEval tasks...")
    tasks = load_tasks(max_tasks=config.max_tasks)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Initialize LLM backend
    llm = LLMBackend(
        backend=config.backend,
        model=config.model,
        temperature=config.temperature,
    )
    llm_fn = await make_llm_fn(llm)

    result = ExperimentResult(
        experiment_name="p1_fleet_sizing",
        config=vars(config) if hasattr(config, "__dict__") else {},
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    total_runs = len(fleet_sizes) * n_reps * len(tasks)
    completed = 0

    for n in fleet_sizes:
        agents = [f"agent_{i}" for i in range(n)]
        logger.info(f"Fleet size n={n} ({len(agents)} agents, {topology.channel_count(n)} channels)")

        for rep in range(1, n_reps + 1):
            for task in tasks:
                prompt = format_task_prompt(task)

                try:
                    round_result = await topology.run(
                        task=prompt,
                        agents=agents,
                        llm_fn=llm_fn,
                        max_rounds=config.max_rounds,
                    )

                    correct = evaluate_solution(task, round_result.final_answer)

                    task_result = TaskResult(
                        task_id=task.task_id,
                        fleet_size=n,
                        rep=rep,
                        correct=correct,
                        answer=round_result.final_answer[:500],
                        total_tokens=round_result.total_tokens,
                        topology="all-to-all",
                    )
                except Exception as e:
                    logger.warning(f"Error on {task.task_id} n={n} rep={rep}: {e}")
                    task_result = TaskResult(
                        task_id=task.task_id,
                        fleet_size=n,
                        rep=rep,
                        correct=False,
                        answer=f"ERROR: {e}",
                        topology="all-to-all",
                    )

                result.task_results.append(vars(task_result))
                completed += 1

                if completed % 100 == 0:
                    logger.info(f"Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")

    # Compute summary metrics
    all_task_results = [TaskResult(**r) for r in result.task_results]
    result.summary = compute_acb_metrics(all_task_results, fleet_sizes)
    result.summary["llm_usage"] = llm.usage_summary()
    result.finished_at = datetime.now(timezone.utc).isoformat()

    # Log key findings
    logger.info("=" * 60)
    logger.info("P1 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  a (Pass@1, n=1) = {result.summary['a']:.4f}")
    logger.info(f"  c (overhead)    = {result.summary['c']:.4f}")
    logger.info(f"  n* (predicted)  = {result.summary['n_star']}")
    logger.info(f"  Empirical peak  = {result.summary['empirical_peak']}")
    logger.info(f"  Match (±1)?     = {result.summary['n_star_matches']}")
    logger.info(f"  Pass@1 by n: {result.summary['pass_at_1']}")

    await llm.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="P1: All-to-all fleet sizing on HumanEval")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            experiment_name="p1_fleet_sizing",
            fleet_sizes=args.fleet_sizes,
            n_reps=args.reps,
            max_tasks=args.max_tasks,
            output_dir=args.output_dir,
        )

    result = asyncio.run(run_p1(config))
    save_results(result, config.output_dir)


if __name__ == "__main__":
    main()
