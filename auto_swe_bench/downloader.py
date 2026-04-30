"""Model download helpers using huggingface_hub."""
from __future__ import annotations

from rich.console import Console

console = Console()


def download_gguf(
    repo_id: str,
    filename: str,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """
    Download a single GGUF file from HuggingFace Hub.
    Returns the local file path (cached — re-runs are no-ops).
    """
    from huggingface_hub import hf_hub_download

    console.print(f"[cyan]Downloading GGUF:[/cyan] {repo_id}/{filename} (revision={revision})")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    console.print(f"[green]Model ready:[/green] {path}")
    return path


def download_hf_snapshot(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Download a full HF repo snapshot (for vLLM / safetensors models).
    Returns the local snapshot directory path.
    """
    from huggingface_hub import snapshot_download

    console.print(f"[cyan]Downloading HF snapshot:[/cyan] {repo_id} (revision={revision})")
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    console.print(f"[green]Snapshot ready:[/green] {path}")
    return path
