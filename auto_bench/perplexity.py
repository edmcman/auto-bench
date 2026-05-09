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
    return text[: cfg.max_chars]


def _extract_logprobs(logprobs_data: dict) -> tuple[list[float | None], list[int] | None]:
    if "token_logprobs" in logprobs_data:
        # OpenAI / vLLM: text_offset gives char offset of each token
        return logprobs_data["token_logprobs"], logprobs_data.get("text_offset")
    else:
        # llama.cpp: reconstruct offsets from token text
        items = logprobs_data["content"]
        lps = [item["logprob"] for item in items]
        offsets, pos = [], 0
        for item in items:
            offsets.append(pos)
            pos += len(item["token"])
        return lps, offsets


def compute_perplexity(backend: Backend, cfg: PerplexityConfig) -> tuple[float, list[float]]:
    """Return (perplexity, per-token log-probs) over the evaluation corpus."""
    text = _load_text(cfg)

    all_logprobs: list[float] = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(text), cfg.stride_chars):
            context_start = max(0, i - (cfg.chunk_chars - cfg.stride_chars))
            chunk_end = min(len(text), i + cfg.stride_chars)
            prompt = text[context_start:chunk_end]
            actual_context_chars = i - context_start

            if not prompt.strip():
                continue

            resp = client.post(
                f"{backend.base_url}/completions",
                json={
                    "model": backend.model_name,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 1,
                },
            )
            resp.raise_for_status()
            lps, offsets = _extract_logprobs(resp.json()["choices"][0]["logprobs"])

            if offsets is not None:
                start_idx = next((j for j, off in enumerate(offsets) if off >= actual_context_chars), len(lps))
            else:
                start_idx = round(len(lps) * actual_context_chars / len(prompt)) if prompt else 0

            all_logprobs.extend(lp for lp in lps[start_idx:] if lp is not None)

    if not all_logprobs:
        return float("inf"), []
    nll = -sum(all_logprobs) / len(all_logprobs)
    return math.exp(nll), all_logprobs


def compute_kl_divergence(ref: list[float], cand: list[float]) -> float:
    """D_KL(ref ∥ cand) ≈ mean(log_ref − log_cand) per token."""
    n = min(len(ref), len(cand))
    return sum(r - c for r, c in zip(ref[:n], cand[:n])) / n
