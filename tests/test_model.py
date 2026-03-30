"""Tests for acb.model — core mathematical functions."""

import math
import pytest
from acb.model import (
    acb_performance,
    marginal_gain,
    optimal_fleet_size,
    p_harm,
    p_harm_monte_carlo,
    rho_crit,
    topology_crossover,
    performance_curve,
    optimal_fleet_topology,
)


class TestACBPerformance:
    def test_single_agent(self):
        assert acb_performance(1, 0.72, 0.08) == pytest.approx(0.72)

    def test_two_agents(self):
        # I(2) = 2*0.72 - 0.08*2*1/2 = 1.44 - 0.08 = 1.36
        assert acb_performance(2, 0.72, 0.08) == pytest.approx(1.36)

    def test_performance_at_nstar(self):
        a, c = 0.72, 0.08
        n_star = optimal_fleet_size(a, c)
        # Performance at n* should be >= performance at n*+1 and n*-1
        I_star = acb_performance(n_star, a, c)
        I_plus = acb_performance(n_star + 1, a, c)
        assert I_star >= I_plus

    def test_negative_performance(self):
        # At 2*n*, performance should be near zero or negative
        a, c = 0.72, 0.08
        n_star = optimal_fleet_size(a, c)
        I_2nstar = acb_performance(2 * n_star, a, c)
        assert I_2nstar < acb_performance(1, a, c)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            acb_performance(0, 0.72, 0.08)


class TestMarginalGain:
    def test_positive_at_small_n(self):
        assert marginal_gain(1, 0.72, 0.08) > 0

    def test_zero_at_nstar(self):
        a, c = 0.72, 0.08
        # At n = a/c, marginal gain should be ~0
        n_exact = a / c
        assert abs(marginal_gain(int(n_exact), a, c)) < c

    def test_negative_beyond_nstar(self):
        a, c = 0.72, 0.08
        n_star = optimal_fleet_size(a, c)
        assert marginal_gain(n_star + 5, a, c) < 0


class TestOptimalFleetSize:
    def test_paper_example(self):
        # a=0.72, c=0.082 -> n* = ceil(0.72/0.082) = ceil(8.78) = 9
        assert optimal_fleet_size(0.72, 0.082) == 9

    def test_small_ratio(self):
        # a/c < 1 -> n* = 1
        assert optimal_fleet_size(0.05, 0.10) == 1

    def test_exact_integer(self):
        # a/c = 5.0 exactly -> ceil(5.0) = 5
        assert optimal_fleet_size(0.5, 0.1) == 5

    def test_invalid_c(self):
        with pytest.raises(ValueError):
            optimal_fleet_size(0.72, 0)

    def test_negative_a(self):
        assert optimal_fleet_size(-0.1, 0.08) == 1


class TestPHarm:
    def test_increases_with_n(self):
        """P(harm|n) should be monotonically increasing in n."""
        params = (0.72, 0.15, 0.082)
        prev = 0.0
        for n in [1, 2, 3, 5, 10]:
            curr = p_harm(n, *params)
            assert curr >= prev, f"P(harm) decreased at n={n}"
            prev = curr

    def test_bounded_01(self):
        for n in [1, 5, 15]:
            p = p_harm(n, 0.72, 0.15, 0.082)
            assert 0 <= p <= 1

    def test_matches_monte_carlo(self):
        """Formula should match MC within 0.01."""
        params = (0.72, 0.15, 0.082)
        for n in [1, 3, 7, 10]:
            pf = p_harm(n, *params)
            pmc = p_harm_monte_carlo(n, *params, runs=50_000, seed=42)
            assert abs(pf - pmc) < 0.01, f"Mismatch at n={n}: {pf:.4f} vs {pmc:.4f}"

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            p_harm(1, 0.72, 0, 0.082)


class TestRhoCrit:
    def test_paper_example(self):
        # c=0.082, sigma_sq=0.20, lambda=1 -> rho_crit = 1 - 2*0.082/0.20 = 0.18
        assert rho_crit(0.082, 0.20) == pytest.approx(0.18)

    def test_zero_overhead(self):
        # c=0 -> rho_crit = 1 (aggregation always helps)
        assert rho_crit(0, 0.20) == pytest.approx(1.0)

    def test_high_overhead(self):
        # When c > lambda*sigma_sq/2, rho_crit < 0
        assert rho_crit(0.5, 0.20) < 0


class TestTopologyCrossover:
    def test_paper_example(self):
        # c_s/c_a = 0.18 -> n_crossover = ceil(0.36) + 1 = 2
        assert topology_crossover(0.18 * 0.082, 0.082) == 2

    def test_equal_costs(self):
        # c_s = c_a -> n_crossover = ceil(2) + 1 = 3
        assert topology_crossover(0.082, 0.082) == 3

    def test_invalid_c_a2a(self):
        with pytest.raises(ValueError):
            topology_crossover(0.1, 0)


class TestGeneralizedTopology:
    def test_beta2_reduces_to_standard(self):
        # For beta=2, n*(beta) should be close to a/c
        a, c = 0.72, 0.08
        n_gen = optimal_fleet_topology(a, c, beta=2.0, alpha=0.5)
        assert 5 < n_gen < 20  # sanity range

    def test_beta1_returns_inf(self):
        assert optimal_fleet_topology(0.72, 0.08, beta=1.0) == float("inf")


class TestPerformanceCurve:
    def test_length(self):
        curve = performance_curve(0.72, 0.08, n_max=15)
        assert len(curve) == 15

    def test_peak_exists(self):
        curve = performance_curve(0.72, 0.08, n_max=20)
        values = [v for _, v in curve]
        peak_idx = values.index(max(values))
        # Peak should not be at the endpoints
        assert 0 < peak_idx < len(values) - 1
