"""vLLM backend."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, BackendOptions, ModelConfig, SamplingConfig
from ..downloader import download_hf_snapshot
from .subprocess_backend import SubprocessBackend

console = Console()


class VllmBackend(SubprocessBackend):
    server_name = "vllm"
    log_filename = "vllm.log"

    def download(self) -> str:
        if (path := self._resolve_local_model()) is not None:
            return path
        assert self.model.repo_id, "model.repo_id is required for vllm backend"
        return download_hf_snapshot(
            repo_id=self.model.repo_id,
            revision=self.model.revision,
            token=self.model.hf_token,
            allow_patterns=self.model.allow_patterns,
            ignore_patterns=self.model.ignore_patterns,
        )

    def build_start_command(self, model_path: str) -> list[str]:
        vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            raise FileNotFoundError(
                "vllm not found in PATH. Install it with: pip install vllm"
            )

        cfg = self.backend.vllm
        model_arg = self.model.repo_id if self.model.source == "huggingface" else model_path

        cmd: list[str] = [
            vllm_bin, "serve", model_arg,
            "--host", self.backend.host,
            "--port", str(self.backend.effective_port()),
            "--dtype", cfg.dtype,
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--pipeline-parallel-size", str(cfg.pipeline_parallel_size),
        ]

        if self.backend_options.ctx_size:
            cmd += ["--max-model-len", str(self.backend_options.ctx_size)]

        gen_override: dict[str, float | int] = dict(self.sampling.non_defaults())
        if self.sampling.max_tokens > 0:
            gen_override["max_new_tokens"] = self.sampling.max_tokens
        if gen_override:
            cmd += ["--override-generation-config", json.dumps(gen_override)]

        if cfg.quantization:
            cmd += ["--quantization", cfg.quantization]
        if cfg.enforce_eager:
            cmd.append("--enforce-eager")
        if cfg.max_num_seqs:
            cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]
        if cfg.api_key:
            cmd += ["--api-key", cfg.api_key]
        if cfg.chat_template:
            cmd += ["--chat-template", cfg.chat_template]

        cmd.extend(cfg.extra_args)
        return cmd

    def start(self, model_path: str, output_dir: Path | None = None) -> None:
        cmd = self.build_start_command(model_path)
        console.print(f"[cyan]Starting vLLM:[/cyan] {' '.join(cmd)}")
        if output_dir is not None:
            console.print(f"[dim]vLLM log → {Path(output_dir) / self.log_filename}[/dim]")
        self._launch(cmd, output_dir)

    @property
    def model_name(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return self.model.effective_name()
