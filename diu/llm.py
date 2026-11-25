"""LLM client for answer generation.

Kept behind a thin interface so the provider is a configuration choice rather
than something threaded through the answering logic. Groq is the default: it
serves the Llama family named in the project's architecture, and its free tier
is fast enough for interactive use without a GPU.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM is configured. Never answer without one."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    name = "groq"

    def __init__(self, cfg):
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set.\n"
                "  1. Get a free key at https://console.groq.com/keys\n"
                "  2. Create a file named .env in the project root containing:\n"
                "       GROQ_API_KEY=gsk_your_key_here"
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnavailable(f"the groq package is not installed: {exc}") from exc

        # The SDK retries rate-limit (429) and connection errors with backoff.
        # An evaluation run sends dozens of requests in a row, which exceeds
        # the free tier's per-minute limit with the default of 2 retries.
        self._client = Groq(api_key=api_key, max_retries=int(cfg.answer.get("max_retries", 6)))
        self.model = cfg.answer.get("model", "openai/gpt-oss-120b")
        self.temperature = float(cfg.answer.get("temperature", 0.1))
        self.max_tokens = int(cfg.answer.get("max_tokens", 2000))
        # Reasoning models (gpt-oss, Qwen3) think before answering. Only the
        # options set in config.yaml are sent, because non-reasoning models
        # reject them.
        self.extra: dict = {
            key: cfg.answer[key]
            for key in ("reasoning_effort", "include_reasoning", "reasoning_format")
            if cfg.answer.get(key) is not None
        }

    def complete(self, system: str, user: str) -> str:
        from groq import GroqError

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                # Counts hidden reasoning as well as the answer on reasoning models.
                max_completion_tokens=self.max_tokens,
                **self.extra,
            )
        except GroqError as exc:
            message = str(exc)
            if "decommissioned" in message or "does not exist" in message:
                raise LLMUnavailable(
                    f"Groq rejected model '{self.model}': {message}\n"
                    f"Run 'python cli.py models' to list what is currently served, "
                    f"then update answer.model in config.yaml."
                ) from exc
            raise LLMUnavailable(f"Groq request failed: {message}") from exc
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            # The token budget went on reasoning and nothing was answered.
            # Failing loudly beats returning an empty answer.
            raise LLMUnavailable(
                f"Model '{self.model}' used its whole token budget ({self.max_tokens}) "
                f"without answering. Raise answer.max_tokens or lower "
                f"answer.reasoning_effort in config.yaml."
            )
        return text

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list().data)


def get_client(cfg) -> LLMClient:
    provider = (cfg.answer.get("provider") or "groq").lower()
    if provider == "groq":
        return GroqClient(cfg)
    raise LLMUnavailable(
        f"Unknown answer.provider '{provider}'. Supported: groq."
    )
