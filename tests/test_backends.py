"""Tests for backend implementations."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from email.message import Message
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest

from underfit import Artifact, Html
from underfit.backends.api import APIBackend
from underfit.backends.local import LocalBackend


def test_local_backend_writes_backfill_layout(tmp_path: Path) -> None:
    """Write local run data in the layout consumed by the API backfill service."""
    backend = LocalBackend(project_name="Vision", run_name="Trial A", run_config={"lr": 0.01}, root_dir=tmp_path)

    backend.log_scalars({"loss": 0.8}, step=1)
    backend.log_lines("worker-1", ["hello", "world\n"])

    artifact = Artifact("dataset-v1", "dataset", metadata={"format": "json"})
    artifact.add_bytes(b"{}", name="payload.json")
    backend.log_artifact(artifact)

    backend.log_media("samples", 7, [Html("<h1>ok</h1>", caption="summary").to_payload()])
    backend.finish()

    UUID(backend.run_dir.name)
    metadata = json.loads((backend.run_dir / "run.json").read_text())
    assert metadata["project"] == "Vision"
    assert metadata["name"] == "Trial A"
    assert metadata["status"] == "finished"
    assert metadata["config"] == {"lr": 0.01}

    scalar_path = backend.run_dir / "scalars" / "0" / "raw.jsonl"
    scalar_lines = [json.loads(line) for line in scalar_path.read_text().splitlines()]
    assert scalar_lines == [{"step": 1, "values": {"loss": 0.8}, "timestamp": scalar_lines[0]["timestamp"]}]
    assert backend.read_scalars() == scalar_lines

    log_path = backend.run_dir / "logs" / "worker-1.log"
    assert log_path.read_text() == "hello\nworld\n"
    assert backend.read_logs("worker-1") == [
        {"workerId": "worker-1", "content": "hello"},
        {"workerId": "worker-1", "content": "world"},
    ]

    artifact_dirs = [path for path in (backend.run_dir / "artifacts").iterdir() if path.is_dir()]
    assert len(artifact_dirs) == 1
    artifact_dir = artifact_dirs[0]
    UUID(artifact_dir.name)
    assert json.loads((artifact_dir / "artifact.json").read_text()) == {
        "metadata": {"format": "json"},
        "name": "dataset-v1",
        "type": "dataset",
    }
    assert json.loads((artifact_dir / "manifest.json").read_text()) == {"files": ["payload.json"], "references": []}
    assert (artifact_dir / "files" / "payload.json").read_bytes() == b"{}"
    assert backend.read_artifact_entries("dataset-v1") == [{
        "artifactId": artifact_dir.name,
        "artifactName": "dataset-v1",
        "entry": {"kind": "file", "name": "payload.json", "path": str(artifact_dir / "files" / "payload.json")},
    }]

    media_dirs = [path for path in (backend.run_dir / "media").iterdir() if path.is_dir()]
    assert len(media_dirs) == 1
    media_dir = media_dirs[0]
    UUID(media_dir.name)
    assert (media_dir / "0").read_text() == "<h1>ok</h1>"
    assert json.loads((media_dir / "media.json").read_text()) == {
        "key": "samples",
        "metadata": {"caption": "summary", "inject": True},
        "step": 7,
        "type": "html",
    }


def test_api_backend_matches_worker_label_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the current workerLabel payloads, pass through custom run names, and retry start-line conflicts."""
    calls: list[tuple[str, str, object | None]] = []

    class _Response:
        def __init__(self, payload: object) -> None:
            self._payload = json.dumps(payload).encode()

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> _Response:  # noqa: PYI034
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)

    def _http_error(url: str, payload: object) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(url, 409, "Conflict", hdrs=Message(), fp=BytesIO(json.dumps(payload).encode()))

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        _ = timeout
        if request.data is None:
            payload = None
        else:
            assert isinstance(request.data, bytes)
            payload = json.loads(request.data.decode())
        url = request.full_url
        method = request.get_method()
        calls.append((method, url, payload))

        if url.endswith("/api/v1/me"):
            return _Response({"handle": "owner"})
        if url.endswith("/api/v1/accounts/owner/projects/vision/runs"):
            assert payload == {"name": "baseline-run", "workerLabel": "0", "status": "running", "config": {"lr": 0.1}}
            return _Response({"id": "run-123", "name": "baseline-run"})
        if url.endswith("/baseline-run/scalars"):
            assert payload is not None
            assert "workerId" not in payload
            if payload["startLine"] == 0:
                raise _http_error(url, {"detail": {"error": "Invalid startLine", "expectedStartLine": 3}})
            assert payload["workerLabel"] == "0"
            assert payload["startLine"] == 3
            return _Response({"status": "buffered"})
        if url.endswith("/baseline-run/workers"):
            assert payload == {"workerLabel": "worker-1", "status": "running"}
            return _Response({
                "id": "worker-1-id",
                "runId": "run-123",
                "workerLabel": "worker-1",
                "isPrimary": False,
                "status": "running",
                "joinedAt": "2025-01-01T00:00:00Z",
            })
        if url.endswith("/baseline-run/logs"):
            assert payload is not None
            assert payload["workerLabel"] == "worker-1"
            assert "workerId" not in payload
            return _Response({"status": "buffered"})
        if url.endswith("/baseline-run/logs/flush"):
            assert payload is not None
            assert payload["workerLabel"] in {"0", "worker-1"}
            assert "workerId" not in payload
            return _Response({"status": "flushed"})
        if url.endswith("/baseline-run/scalars/flush"):
            assert payload == {"workerLabel": "0"}
            return _Response({"status": "flushed"})
        if url.endswith("/baseline-run"):
            assert method == "PUT"
            assert payload == {"status": "finished"}
            return _Response({
                "id": "run-123",
                "name": "baseline-run",
                "status": "finished",
                "config": {"lr": 0.1},
            })
        raise AssertionError(f"unexpected request: {method} {url} {payload}")

    monkeypatch.setattr("underfit.backends.api.urllib.request.urlopen", fake_urlopen)

    backend = APIBackend(
        api_url="https://underfit.example",
        api_key="secret",
        project_name="Vision",
        run_name="BASELINE-RUN",
        run_config={"lr": 0.1},
    )

    backend.log_scalars({"loss": 0.5}, step=4)
    backend.log_lines("worker-1", ["hello"])
    backend.finish()

    assert backend.run_name == "baseline-run"
    assert backend.scalar_line == 4
    assert backend._log_line_offsets == {"worker-1": 1}  # noqa: SLF001
    assert all("workerId" not in json.dumps(payload) for _, _, payload in calls if payload is not None)
