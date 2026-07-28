from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_bench.config import RunConfig
from auto_bench import cli, runner


def _run_config(name: str, output_dir: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": name,
            "output_dir": str(output_dir),
            "model": {"source": "local", "local_path": "/models/test"},
            "backend": {"type": "openai"},
            "evaluation": {"run_evaluation": False},
            "jobs_cleanup": "none",
        }
    )


class _FakeBackend:
    def stop(self) -> None:
        pass

    def check_alive(self) -> None:
        pass

    def get_version(self) -> str:
        return "test"


def test_compiled_data_excludes_local_credentials(tmp_path: Path) -> None:
    source = tmp_path / "experiment.jsonnet"
    source.write_text(
        """
        {
          name: "credential-test",
          backend_type: "openai",
          model: { source: "local", local_path: "/models/test" },
        }
        """
    )
    local = tmp_path / "local.yaml"
    local.write_text(
        """
        hf_token: local-hf-secret
        openai:
          api_key: local-openai-secret
        """
    )

    runs, compiled = cli._load_configs_with_data(source, local)

    assert runs[0].model.hf_token == "local-hf-secret"
    assert runs[0].backend.openai.api_key == "local-openai-secret"
    assert "hf_token" not in compiled["model"]
    assert "backend" not in compiled


def test_single_run_stores_source_and_compiled_json(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "experiment.jsonnet"
    source.write_text('{ name: "single" }\n')
    compiled = {"name": "single", "model": {"source": "local"}}
    config = _run_config("single", tmp_path / "results")

    monkeypatch.setattr(runner, "make_backend", lambda config: _FakeBackend())
    monkeypatch.setattr(runner, "_download", lambda backend: "")
    monkeypatch.setattr(runner, "_start_backend", lambda backend, model_path, output_dir: None)

    def fake_run_agent(config, backend, output_dir):
        jobs = output_dir / "jobs"
        jobs.mkdir()
        return jobs

    monkeypatch.setattr(runner, "run_agent", fake_run_agent)

    result = runner.run_single(
        config,
        config_source=source,
        compiled_config=compiled,
    )

    result_dir = Path(result["output_dir"])
    assert (result_dir / source.name).read_text() == source.read_text()
    assert json.loads((result_dir / "experiment.json").read_text()) == compiled


def test_new_sweep_stores_provenance_only_at_root(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sweep.jsonnet"
    source.write_text("[{ name: 'one' }, { name: 'two' }]\n")
    compiled = [{"name": "one"}, {"name": "two"}]
    runs = [
        _run_config("one", tmp_path / "results"),
        _run_config("two", tmp_path / "results"),
    ]
    entry_dirs: list[Path] = []

    def fake_run_single(config, **kwargs):
        entry_dir = Path(config.output_dir) / config.name
        entry_dir.mkdir()
        entry_dirs.append(entry_dir)
        assert kwargs["config_source"] is None
        return {
            "name": config.name,
            "results": {},
            "output_dir": str(entry_dir),
        }

    monkeypatch.setattr(runner, "run_single", fake_run_single)
    monkeypatch.setattr(runner, "_print_sweep_summary", lambda results: None)

    runner.run_pipeline(
        runs,
        sweep_name="sweep",
        config_source=source,
        compiled_config=compiled,
    )

    sweep_dirs = list((tmp_path / "results").glob("sweep_sweep_*"))
    assert len(sweep_dirs) == 1
    sweep_dir = sweep_dirs[0]
    assert (sweep_dir / source.name).read_text() == source.read_text()
    assert json.loads((sweep_dir / "sweep.json").read_text()) == compiled
    assert all(not (entry_dir / source.name).exists() for entry_dir in entry_dirs)
    assert all(not (entry_dir / "sweep.json").exists() for entry_dir in entry_dirs)


def test_resume_accepts_equivalent_json_without_recopying(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sweep.jsonnet"
    source.write_text("current source\n")
    sweep_dir = tmp_path / "existing-sweep"
    sweep_dir.mkdir()
    stored_source = sweep_dir / source.name
    stored_source.write_text("original source\n")
    stored_json = sweep_dir / "sweep.json"
    stored_json.write_text('{\n  "nested": {"b": 2, "a": 1}, "items": [1, true]\n}\n')
    original_json_text = stored_json.read_text()
    runs = [
        _run_config("one", tmp_path / "results"),
        _run_config("two", tmp_path / "results"),
    ]

    monkeypatch.setattr(runner, "run_single", lambda config, **kwargs: {
        "name": config.name,
        "results": {},
        "output_dir": str(sweep_dir / config.name),
    })
    monkeypatch.setattr(runner, "_print_sweep_summary", lambda results: None)

    runner.run_pipeline(
        runs,
        resume_from=sweep_dir,
        config_source=source,
        compiled_config={"items": [1, True], "nested": {"a": 1, "b": 2}},
    )

    assert stored_source.read_text() == "original source\n"
    assert stored_json.read_text() == original_json_text


@pytest.mark.parametrize(
    ("stored", "current"),
    [
        ('{"value": true}', {"value": 1}),
        ('{"items": [1, 2]}', {"items": [2, 1]}),
        ('{"value": "old"}', {"value": "new"}),
    ],
)
def test_resume_rejects_non_equivalent_json(
    tmp_path: Path,
    monkeypatch,
    stored: str,
    current: object,
) -> None:
    source = tmp_path / "sweep.jsonnet"
    source.write_text("{}\n")
    sweep_dir = tmp_path / "existing-sweep"
    sweep_dir.mkdir()
    (sweep_dir / "sweep.json").write_text(stored)
    runs = [
        _run_config("one", tmp_path / "results"),
        _run_config("two", tmp_path / "results"),
    ]
    called = False

    def fake_run_single(config, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "run_single", fake_run_single)

    with pytest.raises(runner.ConfigProvenanceError, match="does not match"):
        runner.run_pipeline(
            runs,
            resume_from=sweep_dir,
            config_source=source,
            compiled_config=current,
        )
    assert called is False


@pytest.mark.parametrize("stored", [None, "{not valid json", '{"value": NaN}'])
def test_resume_rejects_missing_or_invalid_json(
    tmp_path: Path,
    stored: str | None,
) -> None:
    source = tmp_path / "sweep.jsonnet"
    source.write_text("{}\n")
    sweep_dir = tmp_path / "existing-sweep"
    sweep_dir.mkdir()
    if stored is not None:
        (sweep_dir / "sweep.json").write_text(stored)

    with pytest.raises(runner.ConfigProvenanceError, match="Cannot resume"):
        runner._validate_resume_config(sweep_dir, source, {})
