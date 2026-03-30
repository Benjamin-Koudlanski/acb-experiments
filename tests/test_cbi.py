"""Tests for acb.cbi — Coordination Bottleneck Index."""

import pytest
from acb.cbi import compute_cbi, interpret_cbi, CBIZone


class TestComputeCBI:
    def test_optimal(self):
        # n_deployed = n* = 9 for a=0.72, c=0.082
        cbi = compute_cbi(9, 0.72, 0.082)
        assert 0.9 <= cbi <= 1.1

    def test_under_provisioned(self):
        cbi = compute_cbi(3, 0.72, 0.082)
        assert cbi < 0.5

    def test_over_provisioned(self):
        cbi = compute_cbi(20, 0.72, 0.082)
        assert cbi > 2.0


class TestInterpretCBI:
    def test_near_optimal(self):
        result = interpret_cbi(9, 0.72, 0.082)
        assert result.zone == CBIZone.NEAR_OPTIMAL

    def test_critical(self):
        result = interpret_cbi(20, 0.51, 0.065)
        assert result.zone == CBIZone.CRITICAL
        assert "CRITICAL" in result.recommendation

    def test_under_provisioned(self):
        result = interpret_cbi(3, 0.72, 0.082)
        assert result.zone == CBIZone.UNDER_PROVISIONED

    def test_agentprune_before(self):
        # AgentPrune before: n=20, n*=8 -> CBI=2.5
        result = interpret_cbi(20, 0.51, 0.065)
        assert result.cbi > 2.0

    def test_agentprune_after(self):
        # AgentPrune after: n=8, n*=8 -> CBI=1.0
        result = interpret_cbi(8, 0.51, 0.065)
        assert result.zone in (CBIZone.NEAR_OPTIMAL, CBIZone.OVER_PROVISIONED)

    def test_str_output(self):
        result = interpret_cbi(5, 0.72, 0.082)
        s = str(result)
        assert "CBI" in s
        assert "agents" in s
