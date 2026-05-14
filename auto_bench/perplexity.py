"""Compute perplexity and KL divergence via llama-perplexity."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PerplexityConfig


def _load_text(cfg: PerplexityConfig) -> str:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, cfg.dataset_name, split=cfg.split)
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    return text[: cfg.max_chars]


def run_llama_perplexity(
    cmd_template: str,
    model_path: str,
    text: str,
    n_gpu_layers: int | str,
    logits_save: Path | None = None,
    logits_base: Path | None = None,
) -> tuple[float, float | None]:
    """Run llama-perplexity and return (ppl, kl). kl is None unless logits_base is given."""
    ngl = 999 if n_gpu_layers in ("auto", "all") else int(n_gpu_layers)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_file = Path(f.name)

    try:
        extra: list[str] = ["-ngl", str(ngl), "--no-mmap"]
        if logits_save:
            extra += ["--save-all-logits", str(logits_save)]
        if logits_base:
            extra += ["--kl-divergence", "--kl-divergence-base", str(logits_base)]

        cmd_str = cmd_template.format(
            model=model_path,
            file=str(text_file),
            args=" ".join(extra),
        )
        cmd = cmd_str.split()

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        output = proc.stderr + proc.stdout

        if proc.returncode != 0:
            raise RuntimeError(f"llama-perplexity failed (rc={proc.returncode}):\n{output[-2000:]}")

        ppl = _parse_ppl(output, kl_mode=logits_base is not None)
        kl = _parse_kl(output) if logits_base else None
        return ppl, kl
    finally:
        text_file.unlink(missing_ok=True)


def _parse_ppl(output: str, kl_mode: bool = False) -> float:
    if kl_mode:
        m = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", output)
    else:
        m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", output)
    if not m:
        raise ValueError(f"Could not parse PPL from output:\n{output[-1000:]}")
    return float(m.group(1))


def _parse_kl(output: str) -> float:
    m = re.search(r"Mean\s+KLD\s*:\s*([\d.]+)", output)
    if not m:
        raise ValueError(f"Could not parse KLD from output:\n{output[-1000:]}")
    return float(m.group(1))
