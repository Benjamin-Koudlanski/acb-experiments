"""Tests for monte_carlo.validate_pharm."""

import pytest
from monte_carlo.validate_pharm import run_validation


class TestMonteCarloValidation:
    def test_all_pass(self):
        """All fleet sizes should have formula-MC error < 0.01."""
        results = run_validation(mc_runs=10_000, seed=42)
        assert results["all_pass"] is True

    def test_max_error_small(self):
        results = run_validation(mc_runs=10_000, seed=42)
        assert results["max_error"] < 0.01

    def test_monotonic(self):
        """P(harm) should increase with n."""
        results = run_validation(mc_runs=10_000, seed=42)
        p_values = [r["p_formula"] for r in results["results"]]
        for i in range(1, len(p_values)):
            assert p_values[i] >= p_values[i - 1] - 0.001

    def test_custom_params(self):
        results = run_validation(mu_a=0.5, sigma_a=0.2, mu_c=0.1, mc_runs=5_000)
        assert len(results["results"]) == 7  # default fleet sizes
