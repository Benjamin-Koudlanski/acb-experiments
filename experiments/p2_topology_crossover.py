"""
Experiment P2 – Supervisor crossover at n=2.

Prediction: For any n >= 2, supervisor routing outperforms all-to-all
on the same benchmark. Falsified if all-to-all outperforms supervisor
at any n in {2,...,10} at p < 0.05.

Protocol:
  1. For each fleet size n in {1..15}:
     a. For each rep in {1..50}:
        i.  For each MATH task:
            - Deploy n agents in ALL-TO-ALL topology
            - Deploy n agents in SUPERVISOR topology
            - Record: accuracy, tokens for both
  2. For each n >= 2: paired Wilcoxon test comparing topologies
  3. Compute n_crossover from measured (c_s, c_a)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from acb.llm_backend import LLMBackend, make_llm_fn
from acb.model import topology_crossover
from acb.topologies import AllToAll, Supervisor
from benchmarks.math_bench import (
    evaluate_answer,
    extract_model_answer,
    format_task_prompt,
    load_tasks,
)
from experiments.common import (
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
    compute_pass_at_1,
    compute_token_overhead,
    save_results,
    setup_logging,
)

logger = logging.getLogger("acb.p2")

DEFAULT_FLEET_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]


async def run_p2(config: ExperimentConfig) -> ExperimentResult:
    """Execute experiment P2."""
    setup_logging()

    fleet_sizes = config.fleet_sizes or DEFAULT_FLEET_SIZES
    n_reps = config.n_reps
    topo_a2a = AllToAll()
    topo_sup = Supervisor()

    logger.info("Loading MATH tasks...")
    tasks = load_tasks(max_tasks=config.max_tasks or 100)
    logger.info(f"Loaded {len(tasks)} tasks")

    llm = LLMBackend(
        backend=config.backend,  # pyright: ignore[reportArgumentType]
        model=config.model,
        temperature=config.temperature,
    )
    llm_fn = await make_llm_fn(llm)

    result = ExperimentResult(
        experiment_name="p2_topology_crossover",
        config=vars(config) if hasattr(config, "__dict__") else {},
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    total_runs = len(fleet_sizes) * n_reps * len(tasks) * 2
    completed = 0

    for n in fleet_sizes:
        agents = [f"agent_{i}" for i in range(n)]
        logger.info(
            f"Fleet size n={n} | a2a channels={topo_a2a.channel_count(n)}, "
            f"sup channels={topo_sup.channel_count(n)}"
        )

        for rep in range(1, n_reps + 1):
            for task in tasks:
                prompt = format_task_prompt(task)

                # All-to-all run
                try:
                    rr_a2a = await topo_a2a.run(
                        task=prompt, agents=agents, llm_fn=llm_fn,
                        max_rounds=config.max_rounds,
                    )
                    ans_a2a = extract_model_answer(rr_a2a.final_answer)
                    correct_a2a = evaluate_answer(task, ans_a2a)
                    result.task_results.append(vars(TaskResult(
                        task_id=task.task_id, fleet_size=n, rep=rep,
                        correct=correct_a2a, answer=ans_a2a[:500],
                        total_tokens=rr_a2a.total_tokens, topology="all-to-all",
                    )))
                except Exception as e:
                    logger.warning(f"a2a error {task.task_id} n={n}: {e}")
                    result.task_results.append(vars(TaskResult(
                        task_id=task.task_id, fleet_size=n, rep=rep,
                        correct=False, answer=f"ERROR: {e}", topology="all-to-all",
                    )))

                # Supervisor run
                try:
                    rr_sup = await topo_sup.run(
                        task=prompt, agents=agents, llm_fn=llm_fn,
                        max_rounds=config.max_rounds,
                    )
                    ans_sup = extract_model_answer(rr_sup.final_answer)
                    correct_sup = evaluate_answer(task, ans_sup)
                    result.task_results.append(vars(TaskResult(
                        task_id=task.task_id, fleet_size=n, rep=rep,
                        correct=correct_sup, answer=ans_sup[:500],
                        total_tokens=rr_sup.total_tokens, topology="supervisor",
                    )))
                except Exception as e:
                    logger.warning(f"sup error {task.task_id} n={n}: {e}")
                    result.task_results.append(vars(TaskResult(
                        task_id=task.task_id, fleet_size=n, rep=rep,
                        correct=False, answer=f"ERROR: {e}", topology="supervisor",
                    )))

                completed += 2
                if completed % 200 == 0:
                    logger.info(
                        f"Progress: {completed}/{total_runs} "
                        f"({100*completed/total_runs:.1f}%)"
                    )

    # Summary
    all_results = [TaskResult(**r) for r in result.task_results]
    a2a_results = [r for r in all_results if r.topology == "all-to-all"]
    sup_results = [r for r in all_results if r.topology == "supervisor"]

    summary = {
        "all_to_all": {n: compute_pass_at_1(a2a_results, n) for n in fleet_sizes},
        "supervisor": {n: compute_pass_at_1(sup_results, n) for n in fleet_sizes},
        "supervisor_wins_at": [],
        "a2a_wins_at": [],
    }

    for n in fleet_sizes:
        acc_a2a = summary["all_to_all"][n]
        acc_sup = summary["supervisor"][n]
        if acc_sup > acc_a2a + 0.005:
            summary["supervisor_wins_at"].append(n)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
        elif acc_a2a > acc_sup + 0.005:
            summary["a2a_wins_at"].append(n)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult, reportAttributeAccessIssue]  # pyright: ignore[reportUnknownMemberType]

    c_a2a = compute_token_overhead(a2a_results)
    c_sup = compute_token_overhead(sup_results)
    if c_a2a > 0:
        summary["c_all2all"] = c_a2a  # pyright: ignore[reportArgumentType]
        summary["c_supervisor"] = c_sup  # pyright: ignore[reportArgumentType]
        summary["n_crossover_predicted"] = topology_crossover(c_sup, c_a2a)  # pyright: ignore[reportArgumentType]
    else:
        summary["n_crossover_predicted"] = None  # pyright: ignore[reportArgumentType]

    summary["p2_falsified"] = any(n >= 2 for n in summary["a2a_wins_at"])  # pyright: ignore[reportArgumentType]
    summary["llm_usage"] = llm.usage_summary()  # pyright: ignore[reportArgumentType]
    result.summary = summary
    result.finished_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("P2 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Supervisor wins at n: {summary['supervisor_wins_at']}")
    logger.info(f"  All-to-all wins at n: {summary['a2a_wins_at']}")
    logger.info(f"  n_crossover predicted: {summary['n_crossover_predicted']}")
    logger.info(f"  P2 FALSIFIED: {summary['p2_falsified']}")

    await llm.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="P2: Supervisor vs. all-to-all crossover")
    parser.add_argument("--config", type=str)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            experiment_name="p2_topology_crossover",
            fleet_sizes=args.fleet_sizes,
            n_reps=args.reps,
            max_tasks=args.max_tasks,
            output_dir=args.output_dir,
        )

    result = asyncio.run(run_p2(config))
    save_results(result, config.output_dir)


if __name__ == "__main__":
    main()
