"""llama.cpp server backend."""
from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, BackendOptions, ModelConfig, SamplingConfig
from ..downloader import download_gguf, download_gguf_shards
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
        if self.model.filenames:
            return download_gguf_shards(
                repo_id=self.model.repo_id,
                filenames=self.model.filenames,
                revision=self.model.revision,
                token=self.model.hf_token,
            )
        filename = self.model.filename
        assert filename, "model.filename or model.filenames (GGUF filename(s)) is required for llamacpp backend"
        return download_gguf(
            repo_id=self.model.repo_id,
            filename=filename,
            revision=self.model.revision,
            token=self.model.hf_token,
        )

    def build_start_command(self, model_path: str) -> list[str]:
        cfg = self.backend.llamacpp
        model_file = Path(model_path)
        port = self.backend.effective_port()

        parallel = self.backend_options.parallel
        ctx_size = (self.backend_options.ctx_size or 0) * parallel
        inner_args: list[str] = [
            "--parallel", str(parallel),
            "--alias", self.model_name,
            "--ctx-size", str(ctx_size),
        ]
        if self.sampling.max_tokens > 0:
            inner_args += ["--predict", str(self.sampling.max_tokens)]
        _flags = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p", "presence_penalty": "--presence-penalty", "repetition_penalty": "--repeat-penalty"}
        for key, val in self.sampling.non_defaults().items():
            inner_args += [_flags[key], str(val)]
        if cfg.auto_fit:
            inner_args += ["--fit", "on"]
        else:
            inner_args += ["--n-gpu-layers", str(cfg.n_gpu_layers)]
        if self.backend_options.chat_template_kwargs:
            inner_args += ["--chat-template-kwargs", json.dumps(self.backend_options.chat_template_kwargs)]
        inner_args.extend(cfg.extra_args)

        parts = shlex.split(cfg.cmd_template.format(
            model=model_path,
            model_dir=str(model_file.parent),
            model_name=model_file.name,
            host=self.backend.host,
            port=str(port),
            args=shlex.join(inner_args),
        ))
        if not Path(parts[0]).is_absolute():
            resolved = shutil.which(parts[0])
            if resolved is None:
                raise FileNotFoundError(
                    f"Command '{parts[0]}' not found in PATH. "
                    "Set backend.llamacpp.cmd_template to a full path or template."
                )
            parts[0] = resolved
        return parts

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
        if self.model.filenames:
            return Path(self.model.filenames[0]).stem
        if self.model.name:
            return self.model.name
        return self.model.effective_name()
