"""Abstract backend base class."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import httpx

from ..config import BackendConfig, ModelConfig, SamplingConfig


class Backend(ABC):
    """Abstract interface for a local LLM server backend."""

    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig) -> None:
        self.model = model
        self.backend = backend
        self.sampling = sampling

    @abstractmethod
    def download(self) -> str:
        """
        Download / verify the model files.
        Returns the local path (GGUF file or directory) on disk.
        """

    @abstractmethod
    def start(self, model_path: str) -> None:
        """Launch the server process."""

    @abstractmethod
    def stop(self) -> None:
        """Terminate the server process."""

    @property
    def base_url(self) -> str:
        return self.backend.base_url()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name string to use in API requests."""

    def check_alive(self) -> None:
        """Raise if the backend process has died (e.g. due to startup error)."""
        pass

    def wait_ready(self, timeout: int | None = None) -> None:
        """Poll GET /health until 200 or timeout."""
        timeout = timeout or self.backend.startup_timeout
        health_url = f"http://{self.backend.host}:{self.backend.effective_port()}/health"
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            self.check_alive()
            try:
                r = httpx.get(health_url, timeout=5)
                if r.status_code == 200:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(2)
        raise TimeoutError(
            f"Backend did not become healthy within {timeout}s "
            f"(last error: {last_exc})"
        )
