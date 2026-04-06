"""Tests for the remote backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from underfit import Artifact, Html
from underfit.backends.remote import RemoteBackend

API_URL = "https://api.example.com"
API_KEY = "test-api-key"


class _MockResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = json.dumps(data).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _MockResponse:  # noqa: PYI034
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _mock_urlopen(requests: list[tuple[str, str, Any]], responses: list[dict[str, Any]]) -> Any:
    call_index = 0

    def handler(req: Any, **_: Any) -> _MockResponse:
        nonlocal call_index
        ct = req.get_header("Content-type")
        body = json.loads(req.data) if req.data and ct == "application/json" else req.data
        requests.append((req.get_method(), req.full_url, body))
        resp = responses[call_index] if call_index < len(responses) else {}
        call_index += 1
        return _MockResponse(resp)

    return handler


def _create_backend(requests: list[tuple[str, str, Any]], *, run_id: str | None = None) -> RemoteBackend:
    if run_id is None:
        init_responses = [{"id": "run-uuid", "name": "my-run", "workerToken": "wt-123"}]
    else:
        init_responses = [{"runId": "run-uuid", "workerToken": "wt-456"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(requests, init_responses)):
        return RemoteBackend(
            api_url=API_URL, api_key=API_KEY, project="owner/proj", run_name="my-run",
            run_config={"lr": 0.01}, worker_label="gpu-0", run_id=run_id,
        )


def test_create_run() -> None:
    """Create a run via the API and receive a worker token."""
    requests: list[tuple[str, str, Any]] = []
    backend = _create_backend(requests)
    assert backend.run_name == "my-run"
    method, url, body = requests[0]
    assert method == "POST"
    assert url == f"{API_URL}/accounts/owner/projects/proj/runs"
    assert body == {"name": "my-run", "worker_label": "gpu-0", "config": {"lr": 0.01}}


def test_join_run() -> None:
    """Join an existing run as a non-primary worker."""
    requests: list[tuple[str, str, Any]] = []
    backend = _create_backend(requests, run_id="existing-run")
    assert backend.run_name == "existing-run"
    method, url, body = requests[0]
    assert method == "POST"
    assert url == f"{API_URL}/accounts/owner/projects/proj/runs/existing-run/workers"
    assert body == {"workerLabel": "gpu-0"}


def test_log_scalars() -> None:
    """Send scalars with line number tracking."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    responses = [{"nextStartLine": 1, "status": "buffered"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_scalars({"loss": 0.5}, step=1)
    _, _, body = reqs[0]
    assert body["startLine"] == 0
    assert body["scalars"][0]["values"] == {"loss": 0.5}
    assert body["scalars"][0]["step"] == 1


def test_log_lines() -> None:
    """Send log lines with newline stripping and line numbers."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    responses = [{"nextStartLine": 2, "status": "buffered"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_lines(["hello\n", "world"])
    _, _, body = reqs[0]
    assert body["startLine"] == 0
    assert body["lines"][0]["content"] == "hello"
    assert body["lines"][1]["content"] == "world"


def test_log_media() -> None:
    """Upload media via multipart form data."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, [{}])):
        backend.log_media("samples", 1, [Html("<h1>ok</h1>", caption="hi")])
    method, url, data = reqs[0]
    assert method == "POST"
    assert url == f"{API_URL}/ingest/media"
    assert b"<h1>ok</h1>" in data


def test_log_artifact(tmp_path: Path) -> None:
    """Create, upload files, and finalize an artifact."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    (tmp_path / "data.json").write_text("{}")
    artifact = Artifact("ds", "dataset")
    artifact.add_file(tmp_path / "data.json")
    responses = [{"id": "art-uuid"}, {}, {"status": "ok"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_artifact(artifact)
    assert reqs[0][0] == "POST" and reqs[0][2]["name"] == "ds"
    assert reqs[1][0] == "PUT" and reqs[1][1].endswith("/artifacts/art-uuid/files/data.json")
    assert reqs[2][0] == "POST" and reqs[2][2] == {"manifest": {"files": ["data.json"], "references": []}}


def test_finish_sets_terminal_state() -> None:
    """Flush buffers and set terminal state on finish."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, [{}])):
        backend.finish()
    assert len(reqs) == 1
    assert reqs[0][1] == f"{API_URL}/runs/terminal-state"
    assert reqs[0][2] == {"terminalState": "finished"}


def test_line_numbers_increment() -> None:
    """Track line numbers across multiple flush calls."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    responses = [
        {"nextStartLine": 2, "status": "buffered"},
        {"nextStartLine": 5, "status": "buffered"},
    ]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_lines(["a", "b"])
        backend.log_lines(["c", "d", "e"])
    assert reqs[0][2]["startLine"] == 0
    assert reqs[1][2]["startLine"] == 2
