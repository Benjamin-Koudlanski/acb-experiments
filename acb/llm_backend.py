# pyright: ignore[reportMissingTypeArgument]
# pyright: ignore[reportUndefinedVariable]
# pyright: ignore[reportUnreachable]
"""
Unified LLM backend wrapper.
Supports OpenAI API, local Ollama, and vLLM servers through a single
async interface. Backend is selected based on environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("acb.llm_backend")

BackendType = Literal["openai", "ollama", "vllm"]


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion


class LLMBackend:
    """Unified async LLM client with hardened expert prompts."""

    def __init__(
        self,
        backend: BackendType | None = None,
        model: str | None = None,
        max_concurrent: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

        if backend:
            self.backend = backend
        elif os.getenv("OPENAI_API_KEY"):
            self.backend = "openai"
        elif os.getenv("VLLM_URL"):
            self.backend = "vllm"
        elif os.getenv("LOCAL_MODEL_URL"):
            self.backend = "ollama"
        else:
            raise RuntimeError(
                "No LLM backend configured. Set OPENAI_API_KEY, VLLM_URL, "
                "or LOCAL_MODEL_URL in your .env file."
            )

        if model:
            self.model = model
        elif self.backend == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.backend == "vllm":
            self.model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        elif self.backend == "ollama":
            self.model = os.getenv("LOCAL_MODEL_NAME", "llama3.1:8b")

        if self.backend == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        elif self.backend == "vllm":
            self.base_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
            self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        elif self.backend == "ollama":
            self.base_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:11434")
            self.api_key = ""

        max_conc = 1
        self._semaphore = asyncio.Semaphore(max_conc)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    async def generate(
        self,
        prompt: str,
        system: str = "You are a world-class expert in computer science and mathematics. You are rigorous, analytical, and provide bug-free, optimized solutions. You always double-check your logic step-by-step before providing a final answer.",
        agent_id: str = "",
        temperature: float | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature

        async with self._semaphore:
            if self.backend == "ollama":
                return await self._call_ollama(prompt, system, temp)
            else:
                return await self._call_openai_compatible(prompt, system, temp)

    async def _call_openai_compatible(
        self, prompt: str, system: str, temperature: float
    ) -> LLMResponse:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        t0 = time.monotonic()
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )
        latency = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        result = LLMResponse(
            text=data["choices"][0]["message"]["content"],
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )

        self._update_stats(result)
        return result

    async def _call_ollama(
        self, prompt: str, system: str, temperature: float
    ) -> LLMResponse:  # pyright: ignore[reportReturnType]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                resp = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json=body,
                    timeout=120.0,
                )
                latency = (time.monotonic() - t0) * 1000
                resp.raise_for_status()
                data = resp.json()

                result = LLMResponse(
                    text=data["message"]["content"],
                    tokens_prompt=data.get("prompt_eval_count", 0),
                    tokens_completion=data.get("eval_count", 0),
                    latency_ms=latency,
                )
                self._update_stats(result)
                return result
            except Exception as error:
                if attempt == max_retries - 1:
                    logger.error(
                        "Ollama request failed for model %s after %s attempts: %s",
                        self.model,
                        max_retries,
                        error,
                    )
                    raise

                wait_time = (attempt + 1) * 5
                logger.warning(
                    "Ollama request failed on attempt %s/%s. Retrying in %ss...",
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                await asyncio.sleep(wait_time)

    def _update_stats(self, result: LLMResponse):
        """Update global token and call statistics."""
        self.total_prompt_tokens += result.tokens_prompt
        self.total_completion_tokens += result.tokens_completion
        self.total_calls += 1

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def usage_summary(self) -> dict[str, int]:
        """Return token usage summary."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

# --- FIN DE LA CLASSE LLMBackend ---

async def make_llm_fn(backend: LLMBackend):
    """Factory to create a simple async function for agents to use."""
    async def llm_fn(prompt: str, agent_id: str = "") -> tuple[str, int]:
        resp = await backend.generate(prompt, agent_id=agent_id)
        return resp.text, resp.total_tokens

    return llm_fn
