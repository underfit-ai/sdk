"""Tests for `underfit.run.RunSession`."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from underfit.artifact import Artifact, ArtifactDataUpload, ArtifactPathUpload
from underfit.clients import Client
from underfit.media import Html, Image, Media
from underfit.project import Project
from underfit.run import RunSession

artifact_module = importlib.import_module("underfit.artifact")


class _RecordingClient(Client):
    def __init__(self) -> None:
        self.project = Project(handle="acct", name="project", client=self)
        self.run_name = "test-run"
        self.scalar_calls: list[tuple[dict[str, float], int | None]] = []
        self.media_calls: list[tuple[str, int | None, Sequence[Media]]] = []
        self.artifact_calls: list[Artifact] = []
        self.finish_calls: list[str] = []

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        self.scalar_calls.append((values, step))

    def log_lines(self, lines: list[str]) -> None:
        _ = lines

    def log_media(self, key: str, step: int | None, media: Sequence[Media]) -> None:
        self.media_calls.append((key, step, media))

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        if not isinstance(artifact, Artifact):
            raise TypeError("artifact must be an underfit.Artifact")
        self.artifact_calls.append(artifact)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def log_project_artifact(self, project: Project, artifact: Artifact) -> Future[None]:
        _ = project
        return self.log_artifact(artifact)

    def finish(self, terminal_state: str = "finished") -> None:
        self.finish_calls.append(terminal_state)


def _make_session(client: Client, config: dict[str, Any] | None = None) -> RunSession:
    return RunSession(client.project, "run", config)


def test_run_copies_config_on_init() -> None:
    """Copy config input so the run keeps its own snapshot."""
    config = {"lr": 0.1}
    run = _make_session(_RecordingClient(), config)
    config["lr"] = 0.2
    assert run.config == {"lr": 0.1}


def test_log_records_scalars_and_media() -> None:
    """Flatten nested data and send supported payloads to the client."""
    client = _RecordingClient()
    run = _make_session(client)

    run.log({
        "train": {"loss": 1, "done": True},
        "report": Html("<h1>ok</h1>"),
        "samples": {"gallery": [Html("<p>a</p>"), Html("<p>b</p>")]},
    }, step=3)
    run.finish()

    assert client.scalar_calls == [({"train/loss": 1.0, "train/done": 1.0}, 3)]
    assert len(client.media_calls) == 2
    assert client.media_calls[0][0:2] == ("report", 3)
    assert client.media_calls[1][0:2] == ("samples/gallery", 3)
    assert all(isinstance(m, Html) for m in client.media_calls[1][2])


def test_log_rejects_unsupported_values() -> None:
    """Reject unsupported values instead of silently dropping them."""
    client = _RecordingClient()
    run = _make_session(client)

    with pytest.raises(TypeError, match="must contain only media objects: train/tags"):
        run.log({"train": {"loss": 1.0, "tags": ["baseline"]}})

    assert client.scalar_calls == []
    assert client.media_calls == []


def test_log_rejects_mixed_media_lists() -> None:
    """Reject media batches with inconsistent types."""
    run = _make_session(_RecordingClient())
    with pytest.raises(TypeError, match="only one media type: samples"):
        run.log({"samples": [Html("<p>a</p>"), Image(b"img", file_type="png")]})


def test_log_code_respects_include_and_exclude_filters(tmp_path: Path) -> None:
    """Apply include and exclude callbacks to resolved paths."""
    client = _RecordingClient()
    run = _make_session(client)
    root = tmp_path / "src"
    root.mkdir()
    keep = root / "keep.py"
    skip = root / "skip.py"
    note = root / "note.txt"
    keep.write_text("print('keep')\n")
    skip.write_text("print('skip')\n")
    note.write_text("ignore\n")

    run.log_code(
        root,
        include=lambda path: path.suffix == ".py",
        exclude=lambda path: path.name == "skip.py",
    ).result()
    assert [logged.name for logged in client.artifact_calls] == ["source-code"]
    uploads = client.artifact_calls[0].uploads()
    assert len(uploads) == 1
    assert isinstance(uploads[0], ArtifactPathUpload)
    assert uploads[0].path == "keep.py"
    assert Path(uploads[0].source_path).read_bytes() == b"print('keep')\n"


def test_log_git_adds_patch_artifact_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture git metadata on the artifact and upload the working tree patch."""
    client = _RecordingClient()
    run = _make_session(client)
    repo_root = tmp_path.resolve()
    tracked_patch = b"diff --git a/app.py b/app.py\n"
    status_output = "# branch.oid abc123\n# branch.head main\n? new.py"

    def fake_run(args: list[str], cwd: Path, capture_output: bool, check: bool) -> subprocess.CompletedProcess[bytes]:
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

    monkeypatch.setattr(artifact_module.subprocess, "run", fake_run)

    run.log_git(repo_root).result()
    assert client.artifact_calls[0].name == "git-state"
    assert client.artifact_calls[0].metadata == {
        "commit": "abc123",
        "branch": "main",
        "is_dirty": True,
        "untracked_files": ["new.py"],
    }
    assert [logged.name for logged in client.artifact_calls] == ["git-state"]
    uploads = client.artifact_calls[0].uploads()
    assert isinstance(uploads[0], ArtifactDataUpload)
    assert uploads[0].path == "working-tree.patch"
    assert uploads[0].data == tracked_patch


def test_log_model_logs_bytes_checkpoint() -> None:
    """Upload bytes checkpoints as a model artifact."""
    client = _RecordingClient()
    run = _make_session(client)

    run.log_model(b"weights").result()
    artifact = client.artifact_calls[0]
    assert artifact.name == "model-checkpoint"
    assert artifact.type == "model"
    assert len(client.artifact_calls) == 1
    assert len(artifact.uploads()) == 1
    upload = artifact.uploads()[0]
    assert isinstance(upload, ArtifactDataUpload)
    assert upload.path == "checkpoint.bin"
    assert upload.data == b"weights"


def test_finish_is_idempotent_and_blocks_future_writes() -> None:
    """Allow repeated finish calls but reject later write operations."""
    client = _RecordingClient()
    run = _make_session(client)

    run.finish()
    run.finish()

    assert client.finish_calls == ["finished"]
    with pytest.raises(RuntimeError, match="already finished"):
        run.log({"loss": 1.0})
    with pytest.raises(RuntimeError, match="already finished"):
        run.log_artifact(Artifact("model", "model"))
