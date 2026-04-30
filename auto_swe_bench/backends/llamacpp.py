"""llama.cpp server backend."""
from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, ModelConfig, SamplingConfig
from ..downloader import download_gguf
from .openai_backend import OpenAIBackend

console = Console()


class LlamaCppBackend(OpenAIBackend):
    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig) -> None:
        super().__init__(model, backend, sampling)
        self._process: subprocess.Popen | None = None
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    def download(self) -> str:
        if self.model.source == "local":
            path = self.model.local_path
            assert path, "local_path must be set for source='local'"
            console.print(f"[cyan]Using local model:[/cyan] {path}")
            return path

        assert self.model.repo_id, "model.repo_id is required for llamacpp backend"
        filename = self.model.filename
        assert filename, "model.filename (GGUF filename) is required for llamacpp backend"

        return download_gguf(
            repo_id=self.model.repo_id,
            filename=filename,
            revision=self.model.revision,
            token=self.model.hf_token,
        )

    # ------------------------------------------------------------------
    def start(self, model_path: str) -> None:
        self._model_path = model_path
        cfg = self.backend.llamacpp

        binary = cfg.binary
        if not Path(binary).is_absolute():
            resolved = shutil.which(binary)
            if resolved is None:
                raise FileNotFoundError(
                    f"llama-server binary '{binary}' not found in PATH. "
                    "Build llama.cpp or set backend.llamacpp.binary to the full path."
                )
            binary = resolved

        cmd: list[str] = [
            binary,
            "--model", model_path,
            "--host", self.backend.host,
            "--port", str(self.backend.effective_port()),
            "--ctx-size", str(cfg.ctx_size),
            "--parallel", str(cfg.parallel),
            "--batch-size", str(cfg.batch_size),
            "--ubatch-size", str(cfg.ubatch_size),
        ]

        # GPU layers
        ngl = cfg.n_gpu_layers
        if str(ngl).lower() == "auto":
            cmd += ["--n-gpu-layers", "999"]  # load as many as fit
        elif str(ngl).lower() == "all":
            cmd += ["--n-gpu-layers", "999"]
        else:
            cmd += ["--n-gpu-layers", str(ngl)]

        # Flash attention
        fa = str(cfg.flash_attn).lower()
        if fa in ("true", "1", "on", "auto"):
            cmd += ["--flash-attn", fa]
        elif fa not in ("false", "0", "off"):
            cmd += ["--flash-attn", fa]

        # Threads
        if cfg.threads > 0:
            cmd += ["--threads", str(cfg.threads)]

        # API key
        if cfg.api_key:
            cmd += ["--api-key", cfg.api_key]

        # Alias (model name used in API)
        cmd += ["--alias", self.model_name]

        if cfg.mlock:
            cmd.append("--mlock")
        if cfg.no_mmap:
            cmd.append("--no-mmap")

        cmd.extend(cfg.extra_args)

        console.print(f"[cyan]Starting llama-server:[/cyan] {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            console.print("[yellow]Stopping llama-server...[/yellow]")
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    # ------------------------------------------------------------------
    def check_alive(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            stdout, _ = self._process.communicate()
            raise RuntimeError(
                f"llama-server exited with code {self._process.returncode}:\n"
                f"{stdout.decode()[-3000:]}"
            )

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        if self.model.name:
            return self.model.name
        if self.model.filename:
            return Path(self.model.filename).stem
        return self.model.effective_name()
