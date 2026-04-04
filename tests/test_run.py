"""Tests for `underfit.run.Run`."""

from __future__ import annotations

import base64
import importlib
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

from underfit.artifact import Artifact
from underfit.backends.base import Backend
from underfit.media import Html
from underfit.run import Run

run_module = importlib.import_module("underfit.run")


class _RecordingBackend(Backend):
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[dict[str, float], int | None]] = []
        self.media_calls: list[tuple[str, int | None, list[dict[str, Any]]]] = []
        self.artifact_calls: list[Artifact] = []
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

    def log_artifact(self, artifact: Any) -> None:
        if not isinstance(artifact, Artifact):
            raise TypeError("artifact must be an underfit.Artifact")
        self.artifact_calls.append(artifact)

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
    """Flatten nested data and send supported payloads to the backend."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)

    run.log(
        {
            "train": {"loss": 1, "done": True},
            "report": Html("<h1>ok</h1>"),
            "samples": {"gallery": [Html("<p>a</p>"), Html("<p>b</p>")]},
        },
        step=3,
    )

    assert backend.scalar_calls == [({"train/loss": 1.0, "train/done": 1.0}, 3)]
    assert len(backend.media_calls) == 2
    assert backend.media_calls[0][0:2] == ("report", 3)
    assert backend.media_calls[1][0:2] == ("samples/gallery", 3)
    assert [payload["_type"] for payload in backend.media_calls[1][2]] == ["html", "html"]


def test_log_rejects_unsupported_values() -> None:
    """Reject unsupported values instead of silently dropping them."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)

    with pytest.raises(TypeError, match="Lists passed to underfit.Run.log must contain only media objects: train/tags"):
        run.log({"train": {"loss": 1.0, "tags": ["baseline"]}})

    assert backend.scalar_calls == []
    assert backend.media_calls == []


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
    assert [logged.name for logged in backend.artifact_calls] == ["source-code"]
    uploads = backend.artifact_calls[0].upload_files()
    assert uploads[0]["path"] == "source-code.zip"

    archive_bytes = base64.b64decode(uploads[0]["data"])
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == ["keep.py"]
        assert archive.read("keep.py") == b"print('keep')\n"


def test_log_git_adds_patch_artifact_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture git metadata on the artifact and upload the working tree patch."""
    backend = _RecordingBackend()
    run = Run("project", "run", backend)
    repo_root = tmp_path.resolve()
    tracked_patch = b"diff --git a/app.py b/app.py\n"
    status_output = "# branch.oid abc123\n# branch.head main\n? new.py"

    def fake_run(
        args: list[str],
        cwd: Path,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert args[0] == "git"
        assert cwd == repo_root
        assert capture_output is True
        assert check is False

        command = args[1:]
        if command == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{repo_root}\n".encode(), stderr=b"")
        if command == ["status", "--porcelain=v2", "--branch"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{status_output}\n".encode(), stderr=b"")
        if command == ["diff", "--binary", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=tracked_patch, stderr=b"")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)

    artifact = run.log_git(repo_root)

    assert artifact.name == "git-state"
    assert artifact.metadata == {
        "commit": "abc123",
        "branch": "main",
        "is_dirty": True,
        "untracked_files": ["new.py"],
    }
    assert [logged.name for logged in backend.artifact_calls] == ["git-state"]
    uploads = backend.artifact_calls[0].upload_files()
    assert uploads[0]["path"] == "working-tree.patch"
    assert base64.b64decode(uploads[0]["data"]) == tracked_patch


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
