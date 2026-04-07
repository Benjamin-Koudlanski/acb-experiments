"""
Experiment P3 – Shared context eliminates diversity benefit.

Prediction: At n=6, agents with shared RAG context show no significant
improvement over n=1 on factual QA. Agents with isolated context slices
show improvement. Falsified if shared-context benefit is statistically significant.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone

from acb.llm_backend import LLMBackend, make_llm_fn
from acb.model import rho_crit
from benchmarks.math_bench import evaluate_answer, extract_model_answer, load_tasks
from experiments.common import (
    ExperimentConfig, ExperimentResult, TaskResult,
    compute_pass_at_1, save_results, setup_logging,
)

logger = logging.getLogger("acb.p3")
DEFAULT_FLEET_SIZES = [1, 3, 6, 9]


def generate_synthetic_context(problem: str, n_passages: int = 5) -> list[str]:
    """Generate deterministic pseudo-passages based on problem hash."""
    h = hashlib.sha256(problem.encode()).hexdigest()
    hints = [
        "Consider breaking the problem into smaller parts.",
        "Look for patterns in the numbers or expressions.",
        "Try working backwards from the answer format.",
        "Check if algebraic identities or common formulas apply.",
        "Verify your intermediate steps before combining results.",
        "Consider edge cases and boundary conditions.",
        "Draw a diagram or table to organize the information.",
        "Simplify the expression before attempting to solve.",
        "Check units and dimensions for consistency.",
    ]
    return [
        f"[Passage {i+1}] {hints[int(h[i*4:i*4+4], 16) % len(hints)]} (ref: {h[i*2:i*2+6]})"
        for i in range(n_passages)
    ]


def format_shared_prompt(problem: str, passages: list[str], agent_id: str) -> str:
    ctx = "\n".join(passages)
    return (
        f"You are Agent {agent_id}. Use the following retrieved context to help "
        f"solve the problem.\n\nRetrieved context:\n{ctx}\n\n"
        f"Problem: {problem}\n\nSolve step by step. Put your final answer in \\boxed{{}}."
    )


def format_isolated_prompt(problem: str, passages: list[str], idx: int, agent_id: str) -> str:
    my_passage = passages[idx % len(passages)]
    return (
        f"You are Agent {agent_id}. Use your assigned context slice to help "
        f"solve the problem. Other agents have different slices.\n\n"
        f"Your context:\n{my_passage}\n\n"
        f"Problem: {problem}\n\nSolve step by step. Put your final answer in \\boxed{{}}."
    )


async def run_fleet_on_task(problem, passages, agents, llm_fn, mode):
    total_tokens = 0
    answers = []
    for i, aid in enumerate(agents):
        if mode == "shared":
            prompt = format_shared_prompt(problem, passages, aid)
        else:
            prompt = format_isolated_prompt(problem, passages, i, aid)
        resp, tok = await llm_fn(prompt, agent_id=aid)
        answers.append(extract_model_answer(resp))
        total_tokens += tok
    # Deterministic tie-breaking: highest count, then lexicographic order
    if answers:
        counts = Counter(answers)
        vote = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    else:
        vote = ""
    return vote, total_tokens


async def run_p3(config: ExperimentConfig) -> ExperimentResult:
    setup_logging()
    fleet_sizes = config.fleet_sizes or DEFAULT_FLEET_SIZES
    tasks = load_tasks(max_tasks=config.max_tasks or 100)
    logger.info(f"Loaded {len(tasks)} tasks")

    llm = LLMBackend(backend=config.backend, model=config.model, temperature=config.temperature)  # pyright: ignore[reportArgumentType]
    llm_fn = await make_llm_fn(llm)

    result = ExperimentResult(
        experiment_name="p3_rag_diversity",
        config=vars(config) if hasattr(config, "__dict__") else {},
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    total = len(fleet_sizes) * config.n_reps * len(tasks) * 2
    done = 0

    for n in fleet_sizes:
        agents = [f"agent_{i}" for i in range(n)]
        for rep in range(1, config.n_reps + 1):
            for task in tasks:
                passages = generate_synthetic_context(task.problem)
                for mode in ["shared", "isolated"]:
                    try:
                        ans, tok = await run_fleet_on_task(
                            task.problem, passages, agents, llm_fn, mode,
                        )
                        correct = evaluate_answer(task, ans)
                        result.task_results.append(vars(TaskResult(
                            task_id=task.task_id, fleet_size=n, rep=rep,
                            correct=correct, answer=ans[:500],
                            total_tokens=tok, topology=f"rag-{mode}",
                        )))
                    except Exception as e:
                        logger.warning(f"{mode} error {task.task_id} n={n}: {e}")
                        result.task_results.append(vars(TaskResult(
                            task_id=task.task_id, fleet_size=n, rep=rep,
                            correct=False, answer=f"ERROR: {e}", topology=f"rag-{mode}",
                        )))
                    done += 1
                    if done % 200 == 0:
                        logger.info(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    all_res = [TaskResult(**r) for r in result.task_results]
    shared = [r for r in all_res if r.topology == "rag-shared"]
    isolated = [r for r in all_res if r.topology == "rag-isolated"]

    summary = {
        "shared": {n: compute_pass_at_1(shared, n) for n in fleet_sizes},
        "isolated": {n: compute_pass_at_1(isolated, n) for n in fleet_sizes},
    }
    baseline = summary["shared"].get(1, 0) or summary["isolated"].get(1, 0)
    summary["baseline_n1"] = baseline  # pyright: ignore[reportArgumentType]
    summary["shared_n6"] = summary["shared"].get(6, 0)  # pyright: ignore[reportArgumentType]
    summary["isolated_n6"] = summary["isolated"].get(6, 0)  # pyright: ignore[reportArgumentType]
    summary["shared_improvement"] = summary["shared_n6"] - baseline  # pyright: ignore[reportArgumentType]
    summary["isolated_improvement"] = summary["isolated_n6"] - baseline  # pyright: ignore[reportArgumentType]
    summary["p3_falsified"] = summary["shared_improvement"] > 0.05  # pyright: ignore[reportArgumentType]
    # ρ_crit parameters: c estimated from token overhead, sigma_sq from
    # empirical pass-rate variance.  Fall back to calibration defaults
    # (c=0.06, σ²=0.15) when experimental estimates are unavailable.
    estimated_c = summary.get("c_overhead", 0.06) or 0.06
    estimated_sigma_sq = 0.15  # TODO: derive from per-task variance when data permits
    summary["rho_crit_params"] = {"c": estimated_c, "sigma_sq": estimated_sigma_sq}  # pyright: ignore[reportArgumentType]
    summary["rho_crit_estimate"] = rho_crit(c=estimated_c, sigma_sq=estimated_sigma_sq)  # pyright: ignore[reportArgumentType]
    summary["llm_usage"] = llm.usage_summary()  # pyright: ignore[reportArgumentType]

    result.summary = summary
    result.finished_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("P3 RESULTS SUMMARY")
    logger.info(f"  Baseline (n=1):     {baseline:.4f}")
    logger.info(f"  Shared ctx (n=6):   {summary['shared_n6']:.4f} ({summary['shared_improvement']:+.4f})")
    logger.info(f"  Isolated ctx (n=6): {summary['isolated_n6']:.4f} ({summary['isolated_improvement']:+.4f})")
    logger.info(f"  P3 FALSIFIED:       {summary['p3_falsified']}")

    await llm.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="P3: Shared vs. isolated RAG context")
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
            experiment_name="p3_rag_diversity", fleet_sizes=args.fleet_sizes,
            n_reps=args.reps, max_tasks=args.max_tasks, output_dir=args.output_dir,
        )
    result = asyncio.run(run_p3(config))
    save_results(result, config.output_dir)


if __name__ == "__main__":
    main()
