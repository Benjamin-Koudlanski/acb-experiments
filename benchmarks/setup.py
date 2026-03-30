#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for ACB experiments.

Usage:
    python benchmarks/setup.py              # Download all benchmarks
    python benchmarks/setup.py --humaneval  # HumanEval only
    python benchmarks/setup.py --math       # MATH only
    python benchmarks/setup.py --verify     # Check what's installed

Datasets are saved to benchmarks/data/.
"""

from __future__ import annotations

import importlib.util
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)

# Multiple sources for MATH dataset (repos change structure over time)
MATH_SOURCES = [
    {
        "name": "HuggingFace (lighteval)",
        "type": "hf_json",
        "url": "https://huggingface.co/datasets/lighteval/MATH/resolve/main/data/test-00000-of-00001.parquet",
        "alt_url": "https://raw.githubusercontent.com/hendrycks/math/main/MATH/test",
    },
    {
        "name": "GitHub (hendrycks/math)",
        "type": "git",
        "url": "https://github.com/hendrycks/math.git",
    },
]


def download_humaneval():
    """Download HumanEval dataset (164 coding tasks, ~1 MB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / "HumanEval.jsonl"

    if jsonl_path.exists():
        n_tasks = sum(1 for _ in open(jsonl_path))
        print(f"✓ HumanEval already present: {jsonl_path} ({n_tasks} tasks)")
        return

    gz_path = DATA_DIR / "HumanEval.jsonl.gz"
    print(f"Downloading HumanEval from {HUMANEVAL_URL}...")
    urllib.request.urlretrieve(HUMANEVAL_URL, gz_path)

    print("Extracting...")
    with gzip.open(gz_path, "rt") as fin, open(jsonl_path, "w") as fout:
        fout.write(fin.read())
    gz_path.unlink(missing_ok=True)

    n_tasks = sum(1 for _ in open(jsonl_path))
    print(f"✓ HumanEval ready: {jsonl_path} ({n_tasks} tasks)")


def _count_math_tasks(math_dir: Path) -> int:
    """Count .json tasks in a MATH test directory."""
    test_dir = math_dir / "test"
    if not test_dir.exists():
        return 0
    count = 0
    for subdir in test_dir.iterdir():
        if subdir.is_dir():
            count += len(list(subdir.glob("*.json")))
    return count


def _find_test_dir(root: Path) -> Path | None:
    """Recursively find the 'test' directory containing MATH subject folders.

    The hendrycks/math repo has changed structure multiple times:
      - Old: math/test/algebra/...
      - New: math/MATH/test/algebra/...
      - Alt: math/data/test/algebra/...

    This function finds whichever structure exists.
    """
    expected_subjects = {"algebra", "geometry", "number_theory", "prealgebra"}

    # Direct children first
    for candidate in [root / "test", root / "MATH" / "test", root / "data" / "test"]:
        if candidate.is_dir():
            children = {c.name for c in candidate.iterdir() if c.is_dir()}
            if children & expected_subjects:
                return candidate

    # Broader search (max depth 4)
    for depth in range(1, 5):
        for p in root.glob("/".join(["*"] * depth) + "/test"):
            if p.is_dir():
                children = {c.name for c in p.iterdir() if c.is_dir()}
                if children & expected_subjects:
                    return p

    return None


def download_math():
    """Download MATH dataset.

    Tries multiple sources in order:
    1. Git clone from hendrycks/math (handles multiple repo structures)
    2. Falls back to manual instructions if all sources fail
    """
    math_dir = DATA_DIR / "MATH"

    if math_dir.exists() and (math_dir / "test").exists():
        n = _count_math_tasks(math_dir)
        if n > 0:
            subjects = [d.name for d in (math_dir / "test").iterdir() if d.is_dir()]
            print(f"✓ MATH already present: {math_dir} ({n} tasks across {len(subjects)} subjects)")
            return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_DIR / "_math_tmp"

    # Clean up any previous failed attempt
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Try git clone
    print("Downloading MATH dataset...")
    print("  Trying: git clone from hendrycks/math")

    git_success = False
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/hendrycks/math.git", str(tmp_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            git_success = True
            print("  ✓ Clone successful")
        else:
            print(f"  ✗ Clone failed: {result.stderr.strip()[:200]}")
    except FileNotFoundError:
        print("  ✗ git not installed")
    except subprocess.TimeoutExpired:
        print("  ✗ Clone timed out")
    except Exception as e:
        print(f"  ✗ Clone error: {e}")

    if git_success:
        # Find the test directory regardless of repo structure
        test_dir = _find_test_dir(tmp_dir)

        if test_dir is None:
            print("  ✗ Could not find test/ directory with MATH subjects in cloned repo")
            print(f"  Repo contents: {[p.name for p in tmp_dir.iterdir()]}")

            # Show deeper structure for debugging
            for p in tmp_dir.rglob("*"):
                depth = len(p.relative_to(tmp_dir).parts)
                if depth <= 3 and p.is_dir():
                    print(f"    {'  ' * depth}{p.name}/")
        else:
            print(f"  Found test data at: {test_dir.relative_to(tmp_dir)}")

            # Copy into canonical structure
            if math_dir.exists():
                shutil.rmtree(math_dir)
            math_dir.mkdir(parents=True)

            # Copy test directory
            shutil.copytree(str(test_dir), str(math_dir / "test"))

            # Also copy train if it exists (sibling of test)
            train_candidate = test_dir.parent / "train"
            if train_candidate.is_dir():
                shutil.copytree(str(train_candidate), str(math_dir / "train"))

            n = _count_math_tasks(math_dir)
            subjects = [d.name for d in (math_dir / "test").iterdir() if d.is_dir()]
            print(f"  ✓ MATH ready: {math_dir} ({n} tasks across {len(subjects)} subjects)")

            # Clean up
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

    # Clean up failed clone
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Try alternative: download via pip (datasets library)
    print("\n  Trying: HuggingFace datasets library")
    try:
        import importlib
        if importlib.util.find_spec("datasets") is None:
            print("  Installing datasets library...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "datasets", "-q"],
                check=True,
                capture_output=True,
            )

        from datasets import load_dataset
        print("  Loading MATH from HuggingFace...")
        ds = load_dataset("hendrycks/competition_math", split="test")

        # Save in our expected format
        if math_dir.exists():
            shutil.rmtree(math_dir)

        test_dir = math_dir / "test"
        counters: dict[str, int] = {}

        for item in ds:
            subject = item.get("type", "unknown").lower().replace(" ", "_")
            subject_dir = test_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)

            idx = counters.get(subject, 0)
            counters[subject] = idx + 1

            with open(subject_dir / f"{idx}.json", "w") as f:
                json.dump({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "level": item.get("level", ""),
                    "type": item.get("type", ""),
                }, f)

        n = _count_math_tasks(math_dir)
        subjects = [d.name for d in test_dir.iterdir() if d.is_dir()]
        print(f"  ✓ MATH ready: {math_dir} ({n} tasks across {len(subjects)} subjects)")
        return

    except Exception as e:
        print(f"  ✗ HuggingFace download failed: {e}")

    # All methods failed — give manual instructions
    print("\n" + "=" * 50)
    print("MATH dataset could not be downloaded automatically.")
    print("Please download manually:")
    print("=" * 50)
    print()
    print("Option A (pip):")
    print("  pip install datasets")
    print("  python -c \"")
    print("    from datasets import load_dataset")
    print("    ds = load_dataset('hendrycks/competition_math', split='test')")
    print("    print(f'Loaded {len(ds)} problems')\"")
    print()
    print("Option B (git):")
    print(f"  git clone https://github.com/hendrycks/math.git /tmp/math")
    print(f"  # Then find the test/ directory and copy it to:")
    print(f"  # {math_dir}/test/")
    print()
    print("Option C (manual):")
    print(f"  1. Go to https://github.com/hendrycks/math")
    print(f"  2. Download and extract to: {math_dir}/")
    print(f"  3. Ensure {math_dir}/test/ contains subject folders")
    print(f"     (algebra/, geometry/, number_theory/, etc.)")
    print()
    print("After downloading, run: python benchmarks/setup.py --verify")


def verify():
    """Verify all datasets are ready."""
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    all_ok = True

    # HumanEval
    he_path = DATA_DIR / "HumanEval.jsonl"
    if he_path.exists():
        n = sum(1 for _ in open(he_path))
        print(f"  ✓ HumanEval: {n} tasks")
    else:
        print(f"  ✗ HumanEval: NOT FOUND at {he_path}")
        all_ok = False

    # MATH
    math_test = DATA_DIR / "MATH" / "test"
    if math_test.exists():
        n = _count_math_tasks(DATA_DIR / "MATH")
        subjects = [d.name for d in math_test.iterdir() if d.is_dir()]
        if n > 0:
            print(f"  ✓ MATH: {n} test tasks across {len(subjects)} subjects")
            print(f"    Subjects: {', '.join(sorted(subjects))}")
        else:
            print(f"  ✗ MATH: directory exists but no .json files found")
            all_ok = False
    else:
        print(f"  ✗ MATH: NOT FOUND at {math_test}")
        all_ok = False

    if all_ok:
        print("\nAll benchmarks ready. You can run experiments.")
    else:
        print("\nSome benchmarks missing. Run: python benchmarks/setup.py")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download ACB benchmark datasets")
    parser.add_argument("--humaneval", action="store_true", help="Download HumanEval only")
    parser.add_argument("--math", action="store_true", help="Download MATH only")
    parser.add_argument("--verify", action="store_true", help="Verify datasets only")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    if args.humaneval:
        download_humaneval()
    elif args.math:
        download_math()
    else:
        # Download all
        print("=" * 50)
        print("ACB Benchmark Setup")
        print("=" * 50)
        print()
        download_humaneval()
        print()
        download_math()

    print()
    verify()


if __name__ == "__main__":
    main()
