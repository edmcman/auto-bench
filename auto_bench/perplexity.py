"""Compute perplexity via the OpenAI-compatible completions endpoint."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .backends.base import Backend
    from .config import PerplexityConfig


def _load_text(cfg: PerplexityConfig) -> str:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, cfg.dataset_name, split=cfg.split)
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    return text[: cfg.max_tokens * 4]


def compute_perplexity(backend: Backend, cfg: PerplexityConfig) -> float:
    text = _load_text(cfg)
    chunks = [text[i : i + cfg.chunk_chars] for i in range(0, len(text), cfg.chunk_chars)]
    total_nll, total_tokens = 0.0, 0
    with httpx.Client(timeout=120) as client:
        for chunk in chunks:
            if not chunk.strip():
                continue
            resp = client.post(
                f"{backend.base_url}/completions",
                json={
                    "model": backend.model_name,
                    "prompt": chunk,
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 1,
                },
            )
            resp.raise_for_status()
            logprobs = resp.json()["choices"][0]["logprobs"]
            if "token_logprobs" in logprobs:
                lps = logprobs["token_logprobs"]  # OpenAI / vLLM format
            else:
                lps = [item["logprob"] for item in logprobs["content"]]  # llama.cpp format
            valid = [lp for lp in lps if lp is not None]
            total_nll += -sum(valid)
            total_tokens += len(valid)
    return math.exp(total_nll / total_tokens) if total_tokens else float("inf")
