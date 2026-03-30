"""
Agent communication topologies for ACB experiments.

Implements all-to-all and supervisor routing as used in P1 and P2.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    """A message between agents."""

    sender: str
    receiver: str
    content: str
    tokens_used: int = 0


@dataclass
class RoundResult:
    """Result of one coordination round."""

    final_answer: str
    messages: list[AgentMessage] = field(default_factory=list)
    total_tokens: int = 0
    rounds: int = 1


class Topology(ABC):
    """Abstract base for communication topologies."""

    @abstractmethod
    async def run(
        self,
        task: str,
        agents: list,
        llm_fn,
        max_rounds: int = 3,
    ) -> RoundResult:
        """Execute the topology on a task and return the result."""
        ...

    @abstractmethod
    def channel_count(self, n: int) -> int:
        """Number of active communication channels for n agents."""
        ...


class AllToAll(Topology):
    """All-to-all topology: every agent sees every other agent's output.

    Channel count: n(n−1)/2 (β = 2).
    """

    def channel_count(self, n: int) -> int:
        return n * (n - 1) // 2

    async def run(
        self,
        task: str,
        agents: list[str],
        llm_fn,
        max_rounds: int = 3,
    ) -> RoundResult:
        n = len(agents)
        messages: list[AgentMessage] = []
        total_tokens = 0

        # Round 1: Each agent answers independently
        responses = {}
        for i, agent_id in enumerate(agents):
            prompt = (
                f"You are Agent {agent_id}. Answer the following task.\n\n"
                f"Task: {task}\n\n"
                f"Provide your answer concisely."
            )
            resp, tok = await llm_fn(prompt, agent_id=agent_id)
            responses[agent_id] = resp
            total_tokens += tok

        # Subsequent rounds: share all responses, ask to refine
        for round_num in range(2, max_rounds + 1):
            context = "\n\n".join(
                f"Agent {aid}: {resp}" for aid, resp in responses.items()
            )
            new_responses = {}
            for agent_id in agents:
                prompt = (
                    f"You are Agent {agent_id}. Here are all agents' current answers:\n\n"
                    f"{context}\n\n"
                    f"Task: {task}\n\n"
                    f"Considering all perspectives, provide your refined answer."
                )
                resp, tok = await llm_fn(prompt, agent_id=agent_id)
                new_responses[agent_id] = resp
                total_tokens += tok

                # Record messages (each agent reads from n-1 others)
                for other_id in agents:
                    if other_id != agent_id:
                        messages.append(
                            AgentMessage(
                                sender=other_id,
                                receiver=agent_id,
                                content=responses[other_id][:200],
                                tokens_used=tok // max(1, n - 1),
                            )
                        )

            responses = new_responses

        # Majority vote or last round consensus
        final = _majority_vote(list(responses.values()))

        return RoundResult(
            final_answer=final,
            messages=messages,
            total_tokens=total_tokens,
            rounds=max_rounds,
        )


class Supervisor(Topology):
    """Supervisor routing topology: one supervisor dispatches to workers.

    Channel count: n−1 (β = 1).
    """

    def channel_count(self, n: int) -> int:
        return n - 1

    async def run(
        self,
        task: str,
        agents: list[str],
        llm_fn,
        max_rounds: int = 3,
    ) -> RoundResult:
        n = len(agents)
        messages: list[AgentMessage] = []
        total_tokens = 0

        supervisor = agents[0]
        workers = agents[1:]

        # Step 1: Supervisor dispatches task to all workers
        worker_responses = {}
        for worker_id in workers:
            prompt = (
                f"You are Worker {worker_id}. The Supervisor has assigned you this task.\n\n"
                f"Task: {task}\n\n"
                f"Provide your answer concisely."
            )
            resp, tok = await llm_fn(prompt, agent_id=worker_id)
            worker_responses[worker_id] = resp
            total_tokens += tok
            messages.append(
                AgentMessage(sender=supervisor, receiver=worker_id, content=task[:200], tokens_used=tok)
            )

        # Step 2: Supervisor aggregates
        summary = "\n\n".join(
            f"Worker {wid}: {resp}" for wid, resp in worker_responses.items()
        )
        agg_prompt = (
            f"You are the Supervisor. Your workers provided these answers:\n\n"
            f"{summary}\n\n"
            f"Task: {task}\n\n"
            f"Synthesize the best answer from the workers' responses."
        )
        final, tok = await llm_fn(agg_prompt, agent_id=supervisor)
        total_tokens += tok

        for worker_id in workers:
            messages.append(
                AgentMessage(
                    sender=worker_id,
                    receiver=supervisor,
                    content=worker_responses[worker_id][:200],
                    tokens_used=tok // max(1, len(workers)),
                )
            )

        return RoundResult(
            final_answer=final,
            messages=messages,
            total_tokens=total_tokens,
            rounds=1,
        )


def _majority_vote(answers: list[str]) -> str:
    """Deterministic majority vote — returns the most common answer.

    Tie-breaking: when multiple answers share the highest count,
    the lexicographically smallest (normalized) answer wins.  This
    guarantees reproducible results across runs and Python versions.

    Falls back to the first answer if the list is empty.
    """
    if not answers:
        return ""

    from collections import Counter

    # Normalize: strip, lowercase for comparison
    normalized = [a.strip().lower() for a in answers]
    counts = Counter(normalized)

    # Deterministic tie-breaking: highest count, then lexicographic order
    most_common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # Return the original-cased version (first occurrence)
    for a in answers:
        if a.strip().lower() == most_common:
            return a
    return answers[0]
