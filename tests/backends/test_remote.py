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


def _create_backend(requests: list[tuple[str, str, Any]]) -> RemoteBackend:
    init_responses = [{"id": "run-uuid", "name": "server-name", "workerToken": "wt-123"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(requests, init_responses)):
        backend = RemoteBackend(
            api_url=API_URL, api_key=API_KEY, project="owner/proj", run_name="my-run",
            launch_id="launch-1", run_config={"lr": 0.01}, worker_label="gpu-0",
        )
    backend._stop.set()  # noqa: SLF001
    backend._flush_thread.join()  # noqa: SLF001
    return backend


def test_launch_run() -> None:
    """Launch a run and adopt the server-returned name."""
    requests: list[tuple[str, str, Any]] = []
    backend = _create_backend(requests)
    assert backend.run_name == "server-name"
    method, url, body = requests[0]
    assert method == "POST"
    assert url == f"{API_URL}/accounts/owner/projects/proj/runs/launch"
    assert body == {"runName": "my-run", "launchId": "launch-1", "workerLabel": "gpu-0", "config": {"lr": 0.01}}


def test_log_scalars() -> None:
    """Send scalars with line number tracking."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    responses = [{"nextStartLine": 1, "status": "buffered"}]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_scalars({"loss": 0.5}, step=1)
        backend._flush_scalars()  # noqa: SLF001
    _, _, body = reqs[0]
    assert body["startLine"] == 0
    assert body["scalars"][0]["values"] == {"loss": 0.5}
    assert body["scalars"][0]["step"] == 1


def test_log_lines() -> None:
    """Send log lines with newline stripping and incrementing line numbers."""
    reqs: list[tuple[str, str, Any]] = []
    backend = _create_backend(reqs)
    reqs.clear()
    responses = [
        {"nextStartLine": 2, "status": "buffered"},
        {"nextStartLine": 5, "status": "buffered"},
    ]
    with patch("underfit.backends.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        backend.log_lines(["hello\n", "world"])
        backend._flush_logs()  # noqa: SLF001
        backend.log_lines(["a", "b", "c"])
        backend._flush_logs()  # noqa: SLF001
    assert reqs[0][2]["startLine"] == 0
    assert reqs[0][2]["lines"][0]["content"] == "hello"
    assert reqs[0][2]["lines"][1]["content"] == "world"
    assert reqs[1][2]["startLine"] == 2


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
        backend.log_artifact(artifact).result()
        backend._upload_pool.shutdown(wait=True)  # noqa: SLF001
    assert reqs[0][0] == "POST"
    assert reqs[0][1] == f"{API_URL}/accounts/owner/projects/proj/runs/server-name/artifacts"
    assert reqs[0][2] == {"name": "ds", "type": "dataset"}
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
