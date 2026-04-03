"""Tests for `underfit.run.Run`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from underfit.artifact import Artifact
from underfit.backends.base import Backend
from underfit.media import Html
from underfit.run import Run


class _RecordingBackend(Backend):
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[dict[str, float], int | None]] = []
        self.media_calls: list[tuple[str, int | None, list[dict[str, Any]]]] = []
        self.artifact_calls: list[tuple[str, dict[str, Any]]] = []
        self.finish_calls = 0
        self.scalars_result = [{"step": 1, "values": {"loss": 0.5}}]
        self.logs_result = [{"workerId": "stdout", "content": "hello"}]
        self.artifacts_result = [{"artifactName": "model"}]

    @property
    def run_name(self) -> str:
        return "test-run"

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        self.scalar_calls.append((values, step))

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        _ = (worker_id, lines)

    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        self.media_calls.append((key, step, payloads))

    def upload_artifact_entry(self, artifact_name: str, entry: dict[str, Any]) -> None:
        self.artifact_calls.append((artifact_name, entry))

    def read_scalars(self) -> list[dict[str, Any]]:
        return self.scalars_result

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        if worker_id is None:
            return self.logs_result
        return [record for record in self.logs_result if record["workerId"] == worker_id]

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        if artifact_name is None:
            return self.artifacts_result
        return [record for record in self.artifacts_result if record["artifactName"] == artifact_name]

    def finish(self) -> None:
        self.finish_calls += 1


def test_run_copies_config_on_init() -> None:
    """Copy config input so the run keeps its own snapshot."""
    config = {"lr": 0.1}
    run = Run("project", "run", _RecordingBackend(), config)
    config["lr"] = 0.2
    assert run.config == {"lr": 0.1}


def test_log_records_scalars_and_media() -> None:
    """Send scalar and media payloads to the backend."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)

    run.log({"loss": 1, "report": Html("<h1>ok</h1>"), "gallery": [Html("<p>a</p>"), Html("<p>b</p>")]}, step=3)

    assert backend.scalar_calls == [({"loss": 1.0}, 3)]
    assert len(backend.media_calls) == 2
    assert backend.media_calls[0][0:2] == ("report", 3)
    assert backend.media_calls[1][0:2] == ("gallery", 3)
    assert [payload["_type"] for payload in backend.media_calls[1][2]] == ["html", "html"]


def test_log_rejects_boolean_scalars() -> None:
    """Reject booleans instead of treating them as numeric values."""
    run = Run("project", "run", _RecordingBackend())

    with pytest.raises(TypeError, match="Boolean values are not supported"):
        run.log({"done": True})


def test_log_code_respects_include_and_exclude_filters(tmp_path: Path) -> None:
    """Apply include and exclude callbacks to resolved paths."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)
    root = tmp_path / "src"
    root.mkdir()
    keep = root / "keep.py"
    skip = root / "skip.py"
    note = root / "note.txt"
    keep.write_text("print('keep')\n")
    skip.write_text("print('skip')\n")
    note.write_text("ignore\n")

    artifact = run.log_code(
        root,
        include=lambda path: path.suffix == ".py",
        exclude=lambda path: path.name == "skip.py",
    )

    assert artifact.name == "source-code"
    assert [name for name, _entry in backend.artifact_calls] == ["source-code"]
    assert backend.artifact_calls[0][1]["name"] == "keep.py"


def test_finish_is_idempotent_and_blocks_future_writes() -> None:
    """Allow repeated finish calls but reject later write operations."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)

    run.finish()
    run.finish()

    assert backend.finish_calls == 1
    with pytest.raises(RuntimeError, match="already finished"):
        run.log({"loss": 1.0})
    with pytest.raises(RuntimeError, match="already finished"):
        run.log_artifact(Artifact("model", "model"))


def test_read_methods_delegate_to_backend() -> None:
    """Return read results directly from the backend."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)

    assert run.read_scalars() == backend.scalars_result
    assert run.read_logs("stdout") == backend.logs_result
    assert run.read_artifact_entries("model") == backend.artifacts_result
