"""Tests for the remote client."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from underfit import Artifact, Html, Image
from underfit.clients.remote import RemoteClient

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


def _create_client(requests: list[tuple[str, str, Any]]) -> RemoteClient:
    launch_response = [{"id": "run-uuid", "name": "server-name", "workerToken": "wt-123"}]
    with patch("underfit.clients.remote.urllib.request.urlopen", side_effect=_mock_urlopen(requests, launch_response)):
        client = RemoteClient(api_url=API_URL, api_key=API_KEY, project="owner/proj")
        client.launch_run(run_name="my-run", launch_id="launch-1", run_config={"lr": 0.01}, worker_label="gpu-0")
    client._stop.set()  # noqa: SLF001
    client._flush_thread.join()  # noqa: SLF001
    return client


def test_log_lines_advances_start_line() -> None:
    """Send log lines with newline stripping and an incrementing line cursor across flushes."""
    reqs: list[tuple[str, str, Any]] = []
    client = _create_client(reqs)
    reqs.clear()
    responses = [
        {"nextStartLine": 2, "status": "buffered"},
        {"nextStartLine": 5, "status": "buffered"},
    ]
    with patch("underfit.clients.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        client.log_lines(["hello\n", "world"])
        client._flush_logs()  # noqa: SLF001
        client.log_lines(["a", "b", "c"])
        client._flush_logs()  # noqa: SLF001
    assert reqs[0][2]["startLine"] == 0
    assert reqs[0][2]["lines"][0]["content"] == "hello"
    assert reqs[0][2]["lines"][1]["content"] == "world"
    assert reqs[1][2]["startLine"] == 2


def test_log_media_uses_specific_part_content_types() -> None:
    """Send media parts with inferred content types instead of octet-stream."""
    reqs: list[tuple[str, str, Any]] = []
    client = _create_client(reqs)
    bodies: list[bytes] = []

    def handler(req: Any, **_: Any) -> _MockResponse:
        bodies.append(req.data)
        return _MockResponse({})

    with patch("underfit.clients.remote.urllib.request.urlopen", side_effect=handler):
        client.log_media("sample", 1, [Image(b"img", file_type="png")])
        client.log_media("sample", 1, [Html("<h1>ok</h1>")])

    assert b"Content-Type: image/png" in bodies[0]
    assert b"Content-Type: text/html" in bodies[1]


def test_finish_flushes_scalar_buffer_and_updates_terminal_state() -> None:
    """Flush queued scalars before reporting terminal state."""
    reqs: list[tuple[str, str, Any]] = []
    client = _create_client(reqs)
    reqs.clear()

    with patch(
        "underfit.clients.remote.urllib.request.urlopen",
        side_effect=_mock_urlopen(reqs, [{"nextStartLine": 1}, {}, {}]),
    ):
        client.log_scalars({"loss": 0.5}, step=3)
        client.finish("failed")

    assert reqs[0][2]["startLine"] == 0
    assert reqs[0][2]["scalars"][0]["step"] == 3
    assert reqs[0][2]["scalars"][0]["values"] == {"loss": 0.5}
    assert reqs[1] == ("PUT", f"{API_URL}/api/v1/runs/summary", {"summary": {"loss": 0.5}})
    assert reqs[2] == ("PUT", f"{API_URL}/api/v1/runs/terminal-state", {"terminalState": "failed"})


def test_log_project_artifact_posts_to_project_endpoint() -> None:
    """Project-level uploads hit the project artifacts endpoint instead of the run-scoped one."""
    reqs: list[tuple[str, str, Any]] = []
    responses = [{"id": "art-1"}, {}]
    with patch("underfit.clients.remote.urllib.request.urlopen", side_effect=_mock_urlopen(reqs, responses)):
        client = RemoteClient(api_url=API_URL, api_key=API_KEY, project="owner/proj")
        artifact = Artifact("eval-set", "dataset")
        artifact.add_bytes(b"{}", name="payload.json")
        client.log_project_artifact(client.project, artifact).result()
        client._upload_pool.shutdown(wait=True)  # noqa: SLF001

    assert reqs[0][:2] == ("POST", f"{API_URL}/api/v1/accounts/owner/projects/proj/artifacts")
    assert reqs[1] == ("PUT", f"{API_URL}/api/v1/artifacts/art-1/files/payload.json", b"{}")
    assert reqs[2][:2] == ("POST", f"{API_URL}/api/v1/artifacts/art-1/finalize")
