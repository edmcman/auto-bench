"""Model download helpers using huggingface_hub."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def cached_gguf_path(repo_id: str, filename: str, revision: str = "main") -> str | None:
    """Local path of a GGUF already in the HF cache, or None. Never downloads."""
    from huggingface_hub import try_to_load_from_cache

    result = try_to_load_from_cache(repo_id, filename, revision=revision)
    return result if isinstance(result, str) else None


def cached_snapshot_path(repo_id: str, revision: str = "main") -> str | None:
    """Local path of a fully cached HF snapshot, or None. Never downloads."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
    except Exception:
        return None


def remove_from_cache(model_path: str) -> None:
    """Remove a model from the HF cache given its snapshot path."""
    from huggingface_hub import scan_cache_dir

    # Extract revision hash from path like .../snapshots/<HASH>/...
    parts = Path(model_path).parts
    try:
        idx = parts.index("snapshots")
        revision_hash = parts[idx + 1]
    except (ValueError, IndexError):
        console.print(f"[yellow]Cannot parse revision hash from path: {model_path}[/yellow]")
        return

    try:
        strategy = scan_cache_dir().delete_revisions(revision_hash)
        console.print(f"[cyan]Removing downloaded model:[/cyan] {model_path}")
        strategy.execute()
        console.print(f"[green]Freed {strategy.expected_freed_size_str}[/green]")
        _prune_detached_revisions()
    except Exception as exc:
        console.print(f"[yellow]Failed to remove cached model {model_path}: {exc}[/yellow]")


def _prune_detached_revisions() -> None:
    """Prune detached revisions from cache (equivalent to `hf cache prune`)."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    detached = {
        revision.commit_hash
        for repo in cache.repos
        for revision in repo.revisions
        if len(revision.refs) == 0
    }
    if not detached:
        return

    strategy = cache.delete_revisions(*sorted(detached))
    strategy.execute()
    console.print(f"[dim]Pruned {len(detached)} detached revision(s), freed {strategy.expected_freed_size_str}[/dim]")


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


def download_gguf_shards(
    repo_id: str,
    filenames: list[str],
    revision: str = "main",
    token: str | None = None,
) -> str:
    """
    Download multiple GGUF shards from HuggingFace Hub (split GGUFs).
    Downloads all shards and returns the path of the first one.
    llama.cpp auto-discovers the remaining shards from the same directory.
    """
    from huggingface_hub import hf_hub_download

    console.print(f"[cyan]Downloading {len(filenames)} GGUF shards:[/cyan] {repo_id} (revision={revision})")
    first_path: str | None = None
    for filename in filenames:
        console.print(f"[dim]  → {filename}[/dim]")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
        if first_path is None:
            first_path = path
    assert first_path is not None
    console.print(f"[green]Model ready:[/green] {first_path}")
    return first_path


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
