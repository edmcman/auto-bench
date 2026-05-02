"""OpenAI-compatible API backend (connects to a pre-running remote server)."""
from __future__ import annotations

from pathlib import Path

from .base import Backend


class OpenAIBackend(Backend):
    """Backend that connects to a pre-running OpenAI-compatible server.

    No model download, server start, or server stop — the remote endpoint
    is assumed to be available when the run starts.
    """

    def download(self) -> str:
        return ""

    def start(self, model_path: str, output_dir: Path | None = None) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait_ready(self, timeout: int | None = None) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self.backend.openai.model or self.model.effective_name()
