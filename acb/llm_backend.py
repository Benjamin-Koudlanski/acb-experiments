"""
Unified LLM backend wrapper.

Supports OpenAI API, local Ollama, and vLLM servers through a single
async interface. Backend is selected based on environment variables.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

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
    """Unified async LLM client.

    Automatically detects backend from environment variables:
      - OPENAI_API_KEY → OpenAI
      - LOCAL_MODEL_URL → Ollama
      - VLLM_URL → vLLM

    Usage
    -----
    ```python
    llm = LLMBackend()
    resp = await llm.generate("What is 2+2?")
    print(resp.text, resp.total_tokens)
    ```
    """

    def __init__(
        self,
        backend: BackendType | None = None,
        model: str | None = None,
        max_concurrent: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

        # Auto-detect backend
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

        # Model name
        if model:
            self.model = model
        elif self.backend == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.backend == "vllm":
            self.model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        elif self.backend == "ollama":
            self.model = os.getenv("LOCAL_MODEL_NAME", "llama3:70b")

        # Base URL
        if self.backend == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        elif self.backend == "vllm":
            self.base_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
            self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        elif self.backend == "ollama":
            self.base_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:11434")
            self.api_key = ""

        # Rate limiting
        max_conc = max_concurrent or int(os.getenv("MAX_CONCURRENT", "5"))
        self._semaphore = asyncio.Semaphore(max_conc)

        # Token accounting
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    async def generate(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        agent_id: str = "",
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Parameters
        ----------
        prompt : str
            User message.
        system : str
            System message.
        agent_id : str
            Optional agent identifier for logging.
        temperature : float, optional
            Override instance temperature.

        Returns
        -------
        LLMResponse
        """
        temp = temperature if temperature is not None else self.temperature

        async with self._semaphore:
            if self.backend == "ollama":
                return await self._call_ollama(prompt, system, temp)
            else:
                return await self._call_openai_compatible(prompt, system, temp)

    async def _call_openai_compatible(
        self, prompt: str, system: str, temperature: float
    ) -> LLMResponse:
        """Call OpenAI or vLLM (both use the same API format)."""
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

        self.total_prompt_tokens += result.tokens_prompt
        self.total_completion_tokens += result.tokens_completion
        self.total_calls += 1
        return result

    async def _call_ollama(
        self, prompt: str, system: str, temperature: float
    ) -> LLMResponse:
        """Call local Ollama server."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }

        t0 = time.monotonic()
        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json=body,
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

        self.total_prompt_tokens += result.tokens_prompt
        self.total_completion_tokens += result.tokens_completion
        self.total_calls += 1
        return result

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def usage_summary(self) -> dict:
        """Return token usage summary."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


async def make_llm_fn(backend: LLMBackend):
    """Create a callable (prompt, agent_id) -> (text, tokens) for topologies."""

    async def llm_fn(prompt: str, agent_id: str = "") -> tuple[str, int]:
        resp = await backend.generate(prompt, agent_id=agent_id)
        return resp.text, resp.total_tokens

    return llm_fn
