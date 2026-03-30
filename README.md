# ACB – Agent Coordination Bound

**Experimental code and replication toolkit for the paper:**

> *The Agent Coordination Bound (ACB): A Framework for Multi-LLM Fleet Sizing, Topology Selection, and Deployment Diagnostics*
> — Benjamin Koudlanski, 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains:

| Directory | Contents |
|-----------|----------|
| `acb/` | Core library – ACB model, CBI calculator, P(harm) formula, greedy fleet selector |
| `experiments/` | P1, P2, P3 experiment runners as described in Section 6 of the paper |
| `monte_carlo/` | 50,000-run Monte Carlo validation of the closed-form P(harm\|n) |
| `analysis/` | Result analysis, plotting, and statistical tests |
| `benchmarks/` | Benchmark loaders for HumanEval and MATH |
| `configs/` | YAML configuration files for each experiment |
| `tests/` | Unit tests |

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/acb-experiments.git
cd acb-experiments
pip install -e ".[dev]"
```

### 2. Download benchmark datasets

```bash
python benchmarks/setup.py
```

This downloads HumanEval (~1 MB) and MATH (~50 MB). See `benchmarks/README.md` for manual alternatives.

### 3. Configure your LLM backend

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your API key(s) or local endpoint URL
```

**Supported backends:**
- **OpenAI API** (GPT-4o, GPT-4o-mini) — set `OPENAI_API_KEY`
- **Local / Ollama** (LLaMA-3-70B, Qwen2.5-72B) — set `LOCAL_MODEL_URL` (e.g., `http://localhost:11434`)
- **vLLM** — set `VLLM_URL` (e.g., `http://localhost:8000`)

### 4. Run a single experiment

```bash
# P1: All-to-all fleet sizing on HumanEval
python -m experiments.p1_fleet_sizing --config configs/p1_default.yaml

# P2: Supervisor vs. all-to-all topology comparison
python -m experiments.p2_topology_crossover --config configs/p2_default.yaml

# P3: Shared vs. isolated RAG context
python -m experiments.p3_rag_diversity --config configs/p3_default.yaml
```

### 5. Run the Monte Carlo validation (no API needed)

```bash
python -m monte_carlo.validate_pharm --runs 50000 --seed 42
```

### 6. Compute CBI for your own deployment

```bash
python -m acb.cli cbi --n-deployed 8 --pass-at-1 0.72 --overhead 0.082
# Output: CBI = 0.91 (near-optimal)
```

### 7. Run all experiments (full replication)

```bash
python run_all.py --config configs/full_replication.yaml
```

## Estimated Costs

| Experiment | Fleet sizes | Benchmark | Reps | Est. cost (API) | Zero-cost alternative |
|------------|-------------|-----------|------|------------------|-----------------------|
| P1 | n ∈ {1..15} | HumanEval (164 tasks) | 50 | ~$120 | LLaMA-3-70B via Ollama |
| P2 | n ∈ {1..15} | MATH (200 tasks) | 50 | ~$200 | Qwen2.5-72B via vLLM |
| P3 | n ∈ {1,3,6,9} | MATH retrieval | 50 | ~$60 | Any open model |
| **Total** | | | | **~$380** | **$0 with open models** |

## Repository Structure

```
acb-experiments/
├── acb/                          # Core library
│   ├── __init__.py
│   ├── model.py                  # I(n), ΔI(n), n*, P(harm|n), ρ_crit
│   ├── cbi.py                    # Coordination Bottleneck Index
│   ├── greedy_fleet.py           # Heterogeneous greedy fleet selection
│   ├── topologies.py             # All-to-all, supervisor, hybrid
│   ├── cli.py                    # Command-line interface
│   └── llm_backend.py            # Unified LLM API wrapper
├── experiments/
│   ├── __init__.py
│   ├── p1_fleet_sizing.py        # Prediction P1
│   ├── p2_topology_crossover.py  # Prediction P2
│   ├── p3_rag_diversity.py       # Prediction P3
│   └── common.py                 # Shared experiment utilities
├── monte_carlo/
│   ├── __init__.py
│   └── validate_pharm.py         # P(harm|n) Monte Carlo validation
├── benchmarks/
│   ├── __init__.py
│   ├── humaneval.py              # HumanEval loader & evaluator
│   ├── math_bench.py             # MATH loader & evaluator
│   ├── setup.py                  # Download datasets (run this first)
│   └── README.md                 # Manual download instructions
├── analysis/
│   ├── __init__.py
│   ├── plots.py                  # Publication-quality figures
│   └── statistics.py             # Statistical tests (Wilcoxon, bootstrap CI)
├── notebooks/
│   └── colab_replication.ipynb   # Free replication on Google Colab
├── configs/
│   ├── p1_default.yaml
│   ├── p2_default.yaml
│   ├── p3_default.yaml
│   └── full_replication.yaml
├── tests/
│   ├── test_model.py
│   ├── test_cbi.py
│   └── test_monte_carlo.py
├── .env.example
├── pyproject.toml
├── run_all.py
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@misc{koudlanski2026acb,
  title   = {The Agent Coordination Bound (ACB): A Framework for Multi-LLM
             Fleet Sizing, Topology Selection, and Deployment Diagnostics},
  author  = {Koudlanski, Benjamin},
  year    = {2026},
  note    = {Preprint, TechRxiv}
}
```

## License

MIT — see [LICENSE](LICENSE).
