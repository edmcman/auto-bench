"""llama.cpp server backend."""
from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, BackendOptions, ModelConfig, SamplingConfig
from ..downloader import download_gguf
from .subprocess_backend import SubprocessBackend

console = Console()


class LlamaCppBackend(SubprocessBackend):
    server_name = "llama-server"
    log_filename = "llama-server.log"

    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig, backend_options: BackendOptions | None = None) -> None:
        super().__init__(model, backend, sampling, backend_options)
        self._model_path: str | None = None

    def download(self) -> str:
        if (path := self._resolve_local_model()) is not None:
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

        if self.sampling.max_tokens > 0:
            cmd += ["--predict", str(self.sampling.max_tokens)]

        _flags = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p", "presence_penalty": "--presence-penalty", "repetition_penalty": "--repeat-penalty"}
        for key, val in self.sampling.non_defaults().items():
            cmd += [_flags[key], str(val)]

        if cfg.auto_fit:
            cmd += ["--fit", "on"]
        else:
            cmd += ["--n-gpu-layers", str(cfg.n_gpu_layers)]

        cmd.extend(cfg.extra_args)
        return cmd

    def start(self, model_path: str, output_dir: Path | None = None) -> None:
        self._model_path = model_path
        cmd = self.build_start_command(model_path)
        console.print(f"[cyan]Starting llama-server:[/cyan] {' '.join(cmd)}")
        if output_dir is not None:
            console.print(f"[dim]llama-server log → {Path(output_dir) / self.log_filename}[/dim]")
        self._launch(cmd, output_dir)

    @property
    def model_name(self) -> str:
        if self.model.filename:
            return Path(self.model.filename).stem
        if self.model.name:
            return self.model.name
        return self.model.effective_name()
