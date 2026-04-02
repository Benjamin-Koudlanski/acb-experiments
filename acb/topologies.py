"""
Agent communication topologies for ACB experiments.
Implements analytical all-to-all and supervisor routing for P1 and P2.
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
        """Execute the topology on a task."""
        ...

    @abstractmethod
    def channel_count(self, n: int) -> int:
        """Number of active communication channels for n agents."""
        ...


class AllToAll(Topology):
    """All-to-all topology: analytical reasoning and peer review."""

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

        # Round 1: Generation avec Chain-of-Thought (CoT)
        responses = {}
        for i, agent_id in enumerate(agents):
            prompt = (
                f"You are Agent {agent_id}. Solve the following task.\n\n"
                f"Task: {task}\n\n"
                "INSTRUCTION: First, analyze the problem carefully and think step-by-step. "
                "Identify potential edge cases. Then, provide your complete final answer."
            )
            resp, tok = await llm_fn(prompt, agent_id=agent_id)
            responses[agent_id] = resp
            total_tokens += tok

        # Subsequent rounds: Peer review and critical refinement
        for round_num in range(2, max_rounds + 1):
            context = "\n\n".join(
                f"Agent {aid}: {resp}" for aid, resp in responses.items()
            )
            new_responses = {}
            for agent_id in agents:
                prompt = (
                    f"You are Agent {agent_id}. Review the current answers from all agents:\n\n"
                    f"{context}\n\n"
                    f"Task: {task}\n\n"
                    "INSTRUCTION: Critically evaluate the answers above. Check for logic, syntax, "
                    "or mathematical errors. If you find a mistake, explain it clearly. "
                    "Considering this collective feedback, provide your refined and final corrected answer."
                )
                resp, tok = await llm_fn(prompt, agent_id=agent_id)
                new_responses[agent_id] = resp
                total_tokens += tok

                # Record messages
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

        # Majority vote on the final round
        final = _majority_vote(list(responses.values()))

        return RoundResult(
            final_answer=final,
            messages=messages,
            total_tokens=total_tokens,
            rounds=max_rounds,
        )


class Supervisor(Topology):
    """Supervisor routing topology: Worker analysis and Supervisor synthesis."""

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

        # Step 1: Workers analyze and solve
        worker_responses = {}
        for worker_id in workers:
            prompt = (
                f"You are Worker {worker_id}. The Supervisor has assigned you this task.\n\n"
                f"Task: {task}\n\n"
                "INSTRUCTION: Think step-by-step and provide a rigorous final answer."
            )
            resp, tok = await llm_fn(prompt, agent_id=worker_id)
            worker_responses[worker_id] = resp
            total_tokens += tok
            messages.append(
                AgentMessage(sender=supervisor, receiver=worker_id, content=task[:200], tokens_used=tok)
            )

        # Step 2: Supervisor synthesizes the best approach
        summary = "\n\n".join(
            f"Worker {wid}: {resp}" for wid, resp in worker_responses.items()
        )
        agg_prompt = (
            f"You are the Supervisor. Your workers provided these answers:\n\n"
            f"{summary}\n\n"
            f"Task: {task}\n\n"
            "INSTRUCTION: Compare the workers' reasoning and solutions. Identify the most "
            "correct logic and synthesize it into a final, perfect answer."
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
    """Deterministic majority vote."""
    if not answers:
        return ""

    from collections import Counter

    # Normalize: strip, lowercase for comparison
    normalized = [a.strip().lower() for a in answers]
    counts = Counter(normalized)

    # Tie-breaking: highest count, then lexicographic
    most_common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    for a in answers:
        if a.strip().lower() == most_common:
            return a
    return answers[0]