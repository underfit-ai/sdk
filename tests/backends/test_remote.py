"""Tests for the remote backend."""

from __future__ import annotations

import pytest

from underfit.backends.remote import RemoteBackend, _RequestError


def test_remote_backend_retries_and_tracks_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry idempotency conflicts and keep scalar and log offsets in sync."""
    scalar_starts: list[int] = []
    log_starts: list[int] = []
    ingest_tokens: list[str | None] = []

    def fake_request(
        self: RemoteBackend, method: str, path: str, payload: dict | None = None, *, token: str | None = None,
    ) -> dict:
        if path == "/me":
            return {"handle": "owner"}
        if path.endswith("/runs"):
            return {"id": "abc123", "name": "trial", "workerToken": "worker-token-xyz"}
        if path == "/ingest/scalars":
            assert payload is not None
            ingest_tokens.append(token)
            scalar_starts.append(payload["startLine"])
            if len(scalar_starts) == 1:
                raise _RequestError(method, path, 409, "conflict", {"expectedStartLine": 2})
            return {"status": "buffered"}
        if path == "/ingest/logs":
            assert payload is not None
            ingest_tokens.append(token)
            log_starts.append(payload["startLine"])
            if len(log_starts) == 1:
                raise _RequestError(method, path, 409, "conflict", {"expectedStartLine": 4})
            return {"status": "buffered"}
        raise AssertionError(path)

    monkeypatch.setattr(RemoteBackend, "_request", fake_request)

    backend = RemoteBackend(
        api_url="https://underfit.example",
        api_key="k",
        project_name="demo",
        run_name="Run-1",
        run_config={},
        worker_label="primary",
    )

    backend.log_scalars({"loss": 0.1}, step=5)
    backend.log_lines(["hello", "world"])

    assert scalar_starts == [0, 2]
    assert backend.scalar_line == 3
    assert log_starts == [0, 4]
    assert backend.log_line == 6
    assert ingest_tokens == ["worker-token-xyz"] * 4
