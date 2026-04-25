"""Tests for `underfit.run.RunSession`."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import pytest

from underfit.artifact import Artifact, ArtifactDataUpload, StoredArtifact
from underfit.clients import Client
from underfit.media import Html, Image, Media
from underfit.project import Project
from underfit.run import Run, RunSession


class _RecordingClient(Client):
    def __init__(self) -> None:
        self.project = Project(handle="acct", name="project", client=self)
        self.run = Run(project=self.project, id="test-run", name="test-run")
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

    def log_artifact(self, run: Run, artifact: Artifact) -> Future[None]:
        _ = run
        if not isinstance(artifact, Artifact):
            raise TypeError("artifact must be an underfit.Artifact")
        self.artifact_calls.append(artifact)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def log_project_artifact(self, project: Project, artifact: Artifact) -> Future[None]:
        _ = project
        return self.log_artifact(self.run, artifact)

    def list_runs(self, project: Project) -> list[Run]:
        _ = project
        return []

    def get_run(self, project: Project, name: str) -> Run:
        _ = project, name
        raise FileNotFoundError(name)

    def list_artifacts(self, project: Project, run: Run | None = None) -> list[StoredArtifact]:
        _ = project, run
        return []

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


@pytest.mark.parametrize(("payload", "match"), [
    ({"train": {"loss": 1.0, "tags": ["baseline"]}}, "must contain only media objects: train/tags"),
    ({"samples": [Html("<p>a</p>"), Image(b"img", file_type="png")]}, "only one media type: samples"),
])
def test_log_rejects_invalid_payloads(payload: dict[str, Any], match: str) -> None:
    """Reject unsupported values and inconsistent media lists."""
    run = _make_session(_RecordingClient())
    with pytest.raises(TypeError, match=match):
        run.log(payload)


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
