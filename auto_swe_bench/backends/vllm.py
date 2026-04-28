"""vLLM backend."""
from __future__ import annotations

import shutil
import signal
import subprocess

from rich.console import Console

from ..config import BackendConfig, ModelConfig, SamplingConfig
from ..downloader import download_hf_snapshot
from .base import Backend
from .openai_backend import OpenAIBackend

console = Console()


class VllmBackend(OpenAIBackend):
    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig) -> None:
        super().__init__(model, backend, sampling)
        self._process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    def download(self) -> str:
        """Pre-cache the HF model snapshot so startup is fast."""
        if self.model.source == "local":
            path = self.model.local_path
            assert path, "local_path must be set for source='local'"
            console.print(f"[cyan]Using local model directory:[/cyan] {path}")
            return path

        assert self.model.repo_id, "model.repo_id is required for vllm backend"

        return download_hf_snapshot(
            repo_id=self.model.repo_id,
            revision=self.model.revision,
            token=self.model.hf_token,
            allow_patterns=self.model.allow_patterns,
            ignore_patterns=self.model.ignore_patterns,
        )

    # ------------------------------------------------------------------
    def start(self, model_path: str) -> None:
        vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            raise FileNotFoundError(
                "vllm not found in PATH. Install it with: pip install vllm"
            )

        cfg = self.backend.vllm
        # Use the repo_id (or local path) as the model argument — vLLM
        # will use the pre-cached HF snapshot automatically.
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

        if cfg.max_model_len:
            cmd += ["--max-model-len", str(cfg.max_model_len)]
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

        console.print(f"[cyan]Starting vLLM:[/cyan] {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            console.print("[yellow]Stopping vLLM...[/yellow]")
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    # ------------------------------------------------------------------
    def wait_ready(self, timeout: int | None = None) -> None:
        Backend.wait_ready(self, timeout)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        # vLLM expects the repo_id as the model name in API requests
        if self.model.repo_id:
            return self.model.repo_id
        return self.model.effective_name()
