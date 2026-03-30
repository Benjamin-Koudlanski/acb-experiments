# Contributing to ACB Experiments

Thank you for your interest in contributing to the ACB framework.

## How to Contribute

### Reporting Results

If you run P1, P2, or P3 (with any model/backend), please share your results:

1. Fork this repository
2. Run the experiments and save results to `results/`
3. Open a Pull Request with:
   - Your result JSON files
   - The model and backend you used
   - Any configuration changes from the defaults

We are collecting results across models and backends to build a comprehensive
validation dataset.

### Adding New Experiments

If you want to test the ACB model on a new benchmark or topology:

1. Create a new file in `experiments/`
2. Follow the pattern in `experiments/common.py` for config, logging, and result saving
3. Add a YAML config in `configs/`
4. Add tests in `tests/`

### Bug Reports

Open an issue on GitHub with:
- Python version and OS
- The command you ran
- The full error traceback

## Development Setup

```bash
git clone https://github.com/<your-username>/acb-experiments.git
cd acb-experiments
pip install numpy scipy pandas matplotlib seaborn pyyaml httpx python-dotenv pytest
PYTHONPATH=. pytest tests/ -v
```

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings (NumPy style) on all public functions
- Line length: 100 characters
