"""vLLM backend."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from ..config import BackendConfig, BackendOptions, ModelConfig, SamplingConfig
from ..downloader import download_hf_snapshot
from .subprocess_backend import SubprocessBackend

console = Console()


def _count_visible_gpus() -> int:
    """Return the number of GPUs available to this process."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible not in ("", "NoDeviceFiles"):
        return len(cuda_visible.split(","))
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            count = len(result.stdout.strip().splitlines())
            return count if count > 0 else 1
    except Exception:
        pass
    return 1


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
        cfg = self.backend.vllm
        model_file = Path(model_path)
        port = self.backend.effective_port()
        model_arg = self.model.repo_id if self.model.source == "huggingface" else model_path

        inner_args: list[str] = [
            "--dtype", cfg.dtype,
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        ]
        tp_size = _count_visible_gpus() if cfg.tensor_parallel_size == "auto" else cfg.tensor_parallel_size
        if tp_size > 1:
            inner_args += ["--tensor-parallel-size", str(tp_size)]
        if cfg.pipeline_parallel_size > 1:
            inner_args += ["--pipeline-parallel-size", str(cfg.pipeline_parallel_size)]
        if self.backend_options.ctx_size:
            inner_args += ["--max-model-len", str(self.backend_options.ctx_size)]
        gen_override: dict[str, float | int] = dict(self.sampling.non_defaults())
        if self.sampling.max_tokens > 0:
            gen_override["max_new_tokens"] = self.sampling.max_tokens
        if gen_override:
            inner_args += ["--override-generation-config", json.dumps(gen_override)]
        if cfg.quantization:
            inner_args += ["--quantization", cfg.quantization]
        if cfg.enforce_eager:
            inner_args.append("--enforce-eager")
        if self.backend_options.parallel > 1:
            inner_args += ["--max-num-seqs", str(self.backend_options.parallel)]
        if cfg.api_key:
            inner_args += ["--api-key", cfg.api_key]
        if cfg.chat_template:
            inner_args += ["--chat-template", cfg.chat_template]
        if cfg.tool_call_parser:
            inner_args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg.tool_call_parser]
        inner_args.extend(cfg.extra_args)

        parts = shlex.split(cfg.cmd_template.format(
            model=model_arg,
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
                    "Install vllm or set backend.vllm.cmd_template."
                )
            parts[0] = resolved
        return parts

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
