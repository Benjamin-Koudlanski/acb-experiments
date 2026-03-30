"""
MATH benchmark loader and evaluator.

Uses the MATH dataset (Hendrycks et al., 2021) for mathematical
problem-solving evaluation.

Reference: https://github.com/hendrycks/math
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class MathTask:
    """A single MATH problem."""

    task_id: str
    problem: str
    solution: str
    answer: str  # extracted from \\boxed{}
    level: str
    subject: str


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in a MATH solution.

    Handles nested braces.
    """
    # Find the last \\boxed{...}
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start : i - 1].strip()


def load_tasks(
    data_dir: Path | None = None,
    split: str = "test",
    max_tasks: int | None = None,
    subjects: list[str] | None = None,
) -> list[MathTask]:
    """Load MATH tasks from disk.

    Expects the dataset to be at `data_dir/MATH/{split}/`.
    Download from: https://github.com/hendrycks/math

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    split : str
        "test" or "train".
    max_tasks : int, optional
        Limit total tasks.
    subjects : list[str], optional
        Filter by subject (e.g., ["algebra", "geometry"]).

    Returns
    -------
    list of MathTask
    """
    data_dir = data_dir or DATA_DIR
    math_dir = data_dir / "MATH" / split
    if not math_dir.exists():
        raise FileNotFoundError(
            f"MATH dataset not found at {math_dir}. "
            f"Download from https://github.com/hendrycks/math and extract to {data_dir}/MATH/"
        )

    target_subjects = subjects or MATH_SUBJECTS
    tasks = []

    for subject in target_subjects:
        subject_dir = math_dir / subject
        if not subject_dir.exists():
            continue

        for json_file in sorted(subject_dir.glob("*.json")):
            with open(json_file) as f:
                obj = json.load(f)

            answer = extract_boxed_answer(obj.get("solution", ""))
            tasks.append(
                MathTask(
                    task_id=f"{subject}/{json_file.stem}",
                    problem=obj["problem"],
                    solution=obj["solution"],
                    answer=answer,
                    level=obj.get("level", ""),
                    subject=subject,
                )
            )

            if max_tasks and len(tasks) >= max_tasks:
                return tasks

    return tasks


def normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace, removes trailing periods, lowercases,
    and normalizes common LaTeX patterns.
    """
    s = answer.strip().rstrip(".").lower()
    # Remove \\text{}, \\mathrm{}, etc.
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    # Remove $ delimiters
    s = s.replace("$", "")
    # Normalize fractions: \\frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    # Remove whitespace
    s = re.sub(r"\s+", "", s)
    return s


def evaluate_answer(task: MathTask, model_answer: str) -> bool:
    """Check if a model's answer matches the ground truth.

    Uses normalized string comparison. This is a heuristic —
    some equivalent mathematical expressions may not match.
    """
    gold = normalize_answer(task.answer)
    pred = normalize_answer(model_answer)

    if not gold or not pred:
        return False

    # Direct match
    if gold == pred:
        return True

    # Try numeric comparison for simple numbers
    try:
        g_val = float(gold)
        p_val = float(pred)
        return abs(g_val - p_val) < 1e-6
    except ValueError:
        pass

    return False


def extract_model_answer(response: str) -> str:
    """Extract the final answer from a model's response.

    Looks for \\boxed{}, "The answer is", or takes the last line.
    """
    # Check for \\boxed{}
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed

    # Check for "the answer is X"
    match = re.search(r"(?:the\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Check for "= X" at end
    match = re.search(r"=\s*(.+?)$", response.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def format_task_prompt(task: MathTask) -> str:
    """Format a MATH task as an LLM prompt."""
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {task.problem}"
    )
