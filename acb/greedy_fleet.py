"""
Greedy Heterogeneous Fleet Selection – Section 3.7.

When deploying a pool of heterogeneous models, optimal selection is NP-hard.
The performance function I(S) is submodular, so a greedy algorithm achieves
within (1 − 1/e) ≈ 63.2% of the true optimum (Nemhauser et al., 1978).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GreedyStep:
    """One step of the greedy selection process."""

    step: int
    agent_name: str
    agent_a: float
    marginal_gain: float
    cumulative_I: float
    stop: bool = False


@dataclass
class GreedyFleetResult:
    """Result of greedy fleet selection."""

    selected: list[str] = field(default_factory=list)
    steps: list[GreedyStep] = field(default_factory=list)
    total_I: float = 0.0


def greedy_fleet_select(
    agents: list[tuple[str, float]],
    c: float,
    cross_model_penalty: float = 1.0,
    max_agents: int = 20,
) -> GreedyFleetResult:
    """Greedy heterogeneous fleet selection.

    Iteratively adds the agent with the highest marginal gain ΔI until
    ΔI becomes negative.

    Parameters
    ----------
    agents : list of (name, a) tuples
        Available agents with their per-agent accuracy gain.
    c : float
        Per-link coordination overhead (base, same-model).
    cross_model_penalty : float
        Multiplicative penalty for cross-model links (default 1.0 = no penalty).
        Paper uses 1.3 for heterogeneous pools.
    max_agents : int
        Safety cap on fleet size.

    Returns
    -------
    GreedyFleetResult
        Selected fleet, selection trace, and total performance.
    """
    result = GreedyFleetResult()
    selected_indices: list[int] = []
    remaining = set(range(len(agents)))

    for step_num in range(1, max_agents + 1):
        best_idx = -1
        best_gain = float("-inf")

        for idx in remaining:
            # Marginal gain: a_j − c · (number of new links)
            # New links = number of already-selected agents
            n_existing = len(selected_indices)
            effective_c = c * cross_model_penalty
            gain = agents[idx][1] - effective_c * n_existing
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0:
            break

        name, a_val = agents[best_idx]

        if best_gain < 0:
            # Record the stop step
            result.steps.append(
                GreedyStep(
                    step=step_num,
                    agent_name=name,
                    agent_a=a_val,
                    marginal_gain=best_gain,
                    cumulative_I=result.total_I,
                    stop=True,
                )
            )
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        result.total_I += best_gain
        result.selected.append(name)
        result.steps.append(
            GreedyStep(
                step=step_num,
                agent_name=name,
                agent_a=a_val,
                marginal_gain=best_gain,
                cumulative_I=result.total_I,
            )
        )

    return result
