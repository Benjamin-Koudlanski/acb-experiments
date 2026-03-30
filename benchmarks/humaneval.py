"""
HumanEval benchmark loader and evaluator.

Downloads the HumanEval dataset (Chen et al., 2021) and provides
task iteration and code execution-based evaluation.

Reference: https://github.com/openai/human-eval
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class HumanEvalTask:
    """A single HumanEval coding task."""

    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str


def download_humaneval(data_dir: Path | None = None) -> Path:
    """Download HumanEval dataset if not already present.

    Returns path to the JSONL file.
    """
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / "HumanEval.jsonl"
    gz_path = data_dir / "HumanEval.jsonl.gz"

    if jsonl_path.exists():
        return jsonl_path

    import urllib.request
    import gzip

    print(f"Downloading HumanEval to {gz_path}...")
    urllib.request.urlretrieve(HUMANEVAL_URL, gz_path)

    print("Extracting...")
    with gzip.open(gz_path, "rt") as fin, open(jsonl_path, "w") as fout:
        fout.write(fin.read())

    gz_path.unlink(missing_ok=True)
    return jsonl_path


def load_tasks(data_dir: Path | None = None, max_tasks: int | None = None) -> list[HumanEvalTask]:
    """Load HumanEval tasks from disk (downloading if necessary).

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing HumanEval.jsonl.
    max_tasks : int, optional
        Limit the number of tasks loaded (for quick testing).

    Returns
    -------
    list of HumanEvalTask
    """
    jsonl_path = download_humaneval(data_dir)
    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            tasks.append(
                HumanEvalTask(
                    task_id=obj["task_id"],
                    prompt=obj["prompt"],
                    entry_point=obj["entry_point"],
                    canonical_solution=obj["canonical_solution"],
                    test=obj["test"],
                )
            )
            if max_tasks and len(tasks) >= max_tasks:
                break
    return tasks


def evaluate_solution(task: HumanEvalTask, completion: str, timeout: float = 10.0) -> bool:
    """Execute a completion against HumanEval test cases.

    Parameters
    ----------
    task : HumanEvalTask
        The task being evaluated.
    completion : str
        The model's code completion (appended to the prompt).
    timeout : float
        Maximum execution time in seconds.

    Returns
    -------
    bool
        True if all test cases pass, False otherwise.
    """
    # Build the full program: prompt + completion + tests
    full_code = task.prompt + completion + "\n\n" + task.test + f"\n\ncheck({task.entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        os.unlink(tmp_path)


def format_task_prompt(task: HumanEvalTask) -> str:
    """Format a HumanEval task as an LLM prompt.

    Returns a prompt asking the model to complete the function.
    """
    return (
        "Complete the following Python function. "
        "Return ONLY the function body (no explanation, no markdown).\n\n"
        f"{task.prompt}"
    )
