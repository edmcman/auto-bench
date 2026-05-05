"""Shared base for backends that manage a subprocess."""
from __future__ import annotations

import signal
import subprocess
import threading
from pathlib import Path

from .openai_backend import OpenAIBackend

_TAIL = 3000


class SubprocessBackend(OpenAIBackend):
    server_name: str = "server"
    log_filename: str = "server.log"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._stopping = False
        self._crash_error: str | None = None

    def _launch(self, cmd: list[str], output_dir: Path | None) -> None:
        if output_dir is not None:
            self._log_path = Path(output_dir) / self.log_filename
            log_file = open(self._log_path, "wb")
        else:
            log_file = subprocess.PIPE  # type: ignore[assignment]
        self._stopping = False
        self._crash_error = None
        self._process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        threading.Thread(target=self._monitor, daemon=True).start()

    def _monitor(self) -> None:
        assert self._process is not None
        self._process.wait()
        if self._stopping or self._process.returncode == 0:
            return
        if self._log_path and self._log_path.exists():
            tail = self._log_path.read_bytes()[-_TAIL:].decode(errors="replace")
        else:
            tail = ""
        self._crash_error = (
            f"{self.server_name} exited with code {self._process.returncode}:\n{tail}"
        )

    def stop(self) -> None:
        self._stopping = True
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log_path and self._log_path.exists():
            subprocess.run(["xz", str(self._log_path)], check=True)

    def check_alive(self) -> None:
        if self._crash_error:
            raise RuntimeError(self._crash_error)
        if self._process is not None and self._process.poll() is not None:
            if self._log_path and self._log_path.exists():
                tail = self._log_path.read_bytes()[-_TAIL:].decode(errors="replace")
            else:
                stdout, _ = self._process.communicate()
                tail = stdout.decode(errors="replace")[-_TAIL:]
            raise RuntimeError(
                f"{self.server_name} exited with code {self._process.returncode}:\n{tail}"
            )
