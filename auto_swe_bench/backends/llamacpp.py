"""llama.cpp server backend."""
from __future__ import annotations

import shlex
import shutil
import signal
import subprocess
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, BackendOptions, ModelConfig, SamplingConfig
from ..downloader import download_gguf
from .openai_backend import OpenAIBackend

console = Console()


class LlamaCppBackend(OpenAIBackend):
    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig, backend_options: BackendOptions | None = None) -> None:
        super().__init__(model, backend, sampling, backend_options)
        self._process: subprocess.Popen | None = None
        self._model_path: str | None = None
        self._log_path: Path | None = None

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
    def build_start_command(self, model_path: str) -> list[str]:
        cfg = self.backend.llamacpp

        binary_parts = shlex.split(cfg.binary)
        binary_name = binary_parts[0]
        if not Path(binary_name).is_absolute():
            resolved = shutil.which(binary_name)
            if resolved is None:
                raise FileNotFoundError(
                    f"Command '{binary_name}' not found in PATH. "
                    "Set backend.llamacpp.binary to a full path or compound command."
                )
            binary_parts[0] = resolved

        cmd: list[str] = binary_parts + [
            "--model", model_path,
            "--host", self.backend.host,
            "--port", str(self.backend.effective_port()),
            "--parallel", str(cfg.parallel),
            "--alias", self.model_name,
        ]

        cmd += ["--ctx-size", str(self.backend_options.ctx_size or 0)]

        if cfg.auto_fit:
            cmd += ["--fit", "on"]
        else:
            cmd += ["--n-gpu-layers", str(cfg.n_gpu_layers)]

        cmd.extend(cfg.extra_args)
        return cmd

    # ------------------------------------------------------------------
    def start(self, model_path: str, output_dir: Path | None = None) -> None:
        self._model_path = model_path
        cmd = self.build_start_command(model_path)

        console.print(f"[cyan]Starting llama-server:[/cyan] {' '.join(cmd)}")
        if output_dir is not None:
            self._log_path = Path(output_dir) / "llama-server.log"
            log_file = open(self._log_path, "wb")
            console.print(f"[dim]llama-server log → {self._log_path}[/dim]")
        else:
            log_file = subprocess.PIPE  # type: ignore[assignment]
        self._process = subprocess.Popen(
            cmd,
            stdout=log_file,
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
            if self._log_path and self._log_path.exists():
                tail = self._log_path.read_bytes()[-3000:].decode(errors="replace")
            else:
                stdout, _ = self._process.communicate()
                tail = stdout.decode(errors="replace")[-3000:]
            raise RuntimeError(
                f"llama-server exited with code {self._process.returncode}:\n{tail}"
            )

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        if self.model.name:
            return self.model.name
        if self.model.filename:
            return Path(self.model.filename).stem
        return self.model.effective_name()
