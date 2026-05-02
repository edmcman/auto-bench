"""vLLM backend."""
from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, ModelConfig, SamplingConfig
from ..downloader import download_hf_snapshot
from .openai_backend import OpenAIBackend

console = Console()


class VllmBackend(OpenAIBackend):
    def __init__(self, model: ModelConfig, backend: BackendConfig, sampling: SamplingConfig) -> None:
        super().__init__(model, backend, sampling)
        self._process: subprocess.Popen | None = None
        self._log_path: Path | None = None

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
        return cmd

    # ------------------------------------------------------------------
    def start(self, model_path: str, output_dir: Path | None = None) -> None:
        cmd = self.build_start_command(model_path)

        console.print(f"[cyan]Starting vLLM:[/cyan] {' '.join(cmd)}")
        if output_dir is not None:
            self._log_path = Path(output_dir) / "vllm.log"
            log_file = open(self._log_path, "wb")
            console.print(f"[dim]vLLM log → {self._log_path}[/dim]")
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
            console.print("[yellow]Stopping vLLM...[/yellow]")
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
                f"vllm exited with code {self._process.returncode}:\n{tail}"
            )

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        # vLLM expects the repo_id as the model name in API requests
        if self.model.repo_id:
            return self.model.repo_id
        return self.model.effective_name()
