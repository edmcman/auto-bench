from __future__ import annotations

import subprocess

import pytest

from auto_bench.backends.llamacpp import LlamaCppBackend
from auto_bench.backends.vllm import VllmBackend
from auto_bench.config import BackendConfig, ModelConfig, SamplingConfig


def _backend(backend_type: str) -> LlamaCppBackend | VllmBackend:
    backend_config = BackendConfig.model_validate({"type": backend_type})
    backend_class = LlamaCppBackend if backend_type == "llamacpp" else VllmBackend
    return backend_class(
        ModelConfig(source="local", local_path="/models/test"),
        backend_config,
        SamplingConfig(),
    )


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_llamacpp_extracts_version_line(
    stream: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = (
        "ggml_cuda_init: failed to initialize CUDA\n"
        "version: 9009 (0754b7b6f)\n"
        "built with GNU 13.3.0 for Linux x86_64\n"
    )
    completed = subprocess.CompletedProcess(
        ["llama-server", "--version"],
        0,
        stdout=output if stream == "stdout" else "",
        stderr=output if stream == "stderr" else "",
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.shutil.which",
        lambda _: "/usr/bin/llama-server",
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert _backend("llamacpp").get_version() == "9009 (0754b7b6f)"


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (1, "version: 9009 (0754b7b6f)\n", ""),
        (0, "built with GNU 13.3.0 for Linux x86_64\n", ""),
        (0, "version:   \n", ""),
    ],
)
def test_llamacpp_version_returns_none_for_invalid_results(
    returncode: int, stdout: str, stderr: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess(
        ["llama-server", "--version"], returncode, stdout=stdout, stderr=stderr
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.shutil.which",
        lambda _: "/usr/bin/llama-server",
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert _backend("llamacpp").get_version() is None


def test_version_returns_none_when_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.shutil.which", lambda _: None
    )

    assert _backend("llamacpp").get_version() is None


def test_version_returns_none_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.shutil.which",
        lambda _: "/usr/bin/llama-server",
    )

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)

    monkeypatch.setattr("auto_bench.backends.subprocess_backend.subprocess.run", fail)

    assert _backend("llamacpp").get_version() is None


def test_vllm_version_output_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(
        ["vllm", "--version"],
        0,
        stdout="vllm 0.10.0\n",
        stderr="unrelated diagnostic\n",
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.shutil.which", lambda _: "/usr/bin/vllm"
    )
    monkeypatch.setattr(
        "auto_bench.backends.subprocess_backend.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert _backend("vllm").get_version() == "vllm 0.10.0"
