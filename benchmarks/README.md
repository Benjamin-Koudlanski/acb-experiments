# Benchmarks

This directory contains loaders and evaluators for the benchmark datasets
used in ACB experiments. **The data files themselves are not included in the
repository** — they are downloaded on first use.

## Quick Setup

```bash
python benchmarks/setup.py
```

This downloads:
- **HumanEval** (164 coding tasks, ~1 MB) — automatic from GitHub
- **MATH** (12,500 math problems, ~50 MB) — cloned from GitHub

## Manual Download

If the automatic setup fails:

### HumanEval
```bash
mkdir -p benchmarks/data
wget -O benchmarks/data/HumanEval.jsonl.gz \
  https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip benchmarks/data/HumanEval.jsonl.gz
```

### MATH
```bash
cd benchmarks/data
git clone --depth 1 https://github.com/hendrycks/math.git MATH_tmp
mv MATH_tmp/test MATH_tmp/train .
mkdir -p MATH && mv test train MATH/
rm -rf MATH_tmp
```

## Verify

```bash
python benchmarks/setup.py --verify
```

## Structure After Setup

```
benchmarks/
├── data/
│   ├── HumanEval.jsonl          # 164 coding tasks
│   └── MATH/
│       ├── test/
│       │   ├── algebra/         # ~1,187 problems
│       │   ├── counting_and_probability/
│       │   ├── geometry/
│       │   ├── intermediate_algebra/
│       │   ├── number_theory/
│       │   ├── prealgebra/
│       │   └── precalculus/
│       └── train/               # ~7,500 problems
├── humaneval.py                 # Loader + code execution evaluator
├── math_bench.py                # Loader + answer extraction/matching
├── setup.py                     # Download script
└── README.md
```
