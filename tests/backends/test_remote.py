"""Tests for the remote backend."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

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


def test_log_lines_advances_start_line() -> None:
    """Send log lines with newline stripping and an incrementing line cursor across flushes."""
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
