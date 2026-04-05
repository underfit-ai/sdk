"""Tests for the API backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from email.message import Message
from io import BytesIO
from typing import Any, Callable, cast

import pytest

from underfit.backends.api import APIBackend

_RouteHandler = Callable[[Any], object]


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:  # noqa: PYI034
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)


class _APIRouter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self._routes: dict[tuple[str, str], list[object]] = {}
        monkeypatch.setattr("underfit.backends.api.urllib.request.urlopen", self.urlopen)

    def add(self, method: str, path: str, *responses: object) -> None:
        self._routes[(method, path)] = list(responses)

    def urlopen(self, request: urllib.request.Request, timeout: int) -> _Response:
        _ = timeout
        if request.data is None:
            payload = None
        else:
            assert isinstance(request.data, bytes)
            payload = json.loads(request.data.decode())
        path = request.full_url.rsplit("/api/v1", 1)[-1]
        method = request.get_method()
        self.calls.append((method, path, payload))

        key = (method, path)
        if key not in self._routes or not self._routes[key]:
            raise AssertionError(f"unexpected request: {method} {path} {payload}")

        response = self._routes[key].pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = cast(_RouteHandler, response)(payload)
        return _Response(response)


def _http_error(path: str, payload: object) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(path, 409, "Conflict", hdrs=Message(), fp=BytesIO(json.dumps(payload).encode()))


def _expect_payload(expected: object, response: object) -> _RouteHandler:
    def handler(payload: Any) -> object:
        assert payload == expected
        return response

    return handler


def _expect_with(check: Callable[[Any], None], response: object) -> _RouteHandler:
    def handler(payload: Any) -> object:
        check(payload)
        return response

    return handler


def _mock_secondary_worker_attach(
    router: _APIRouter,
    *,
    worker_label: str,
    step: int,
    values: dict[str, float],
) -> None:
    def assert_scalar_payload(payload: Any) -> None:
        assert payload["workerLabel"] == worker_label
        assert payload["startLine"] == 0
        assert len(payload["scalars"]) == 1
        assert payload["scalars"][0]["step"] == step
        assert payload["scalars"][0]["values"] == values

    router.add("GET", "/me", {"handle": "owner"})
    router.add("GET", "/accounts/owner/projects/vision/runs/baseline-run", {"id": "run-123", "name": "baseline-run"})
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/workers",
        _expect_payload(
            {"workerLabel": worker_label, "status": "running"},
            {
                "id": f"worker-{worker_label}",
                "runId": "run-123",
                "workerLabel": worker_label,
                "isPrimary": False,
                "status": "running",
                "joinedAt": "2025-01-01T00:00:00Z",
            },
        ),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/scalars",
        _expect_with(assert_scalar_payload, {"status": "buffered"}),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/logs/flush",
        _expect_payload({"workerLabel": worker_label}, {"status": "flushed"}),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/scalars/flush",
        _expect_payload({"workerLabel": worker_label}, {"status": "flushed"}),
    )
    router.add(
        "PUT",
        f"/accounts/owner/projects/vision/runs/baseline-run/workers/{worker_label}",
        _expect_payload(
            {"status": "finished"},
            {
                "id": f"worker-{worker_label}",
                "runId": "run-123",
                "workerLabel": worker_label,
                "isPrimary": False,
                "status": "finished",
                "joinedAt": "2025-01-01T00:00:00Z",
            },
        ),
    )


def test_api_backend_matches_worker_label_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the current workerLabel payloads, pass through custom run names, and retry start-line conflicts."""
    def assert_scalar_retry(payload: Any) -> None:
        assert "workerId" not in payload
        assert payload["workerLabel"] == "0"
        assert payload["startLine"] == 3

    def assert_worker_log_payload(payload: Any) -> None:
        assert payload["workerLabel"] == "worker-1"
        assert "workerId" not in payload

    def assert_flush_payload(payload: Any) -> None:
        assert payload["workerLabel"] in {"0", "worker-1"}
        assert "workerId" not in payload

    router = _APIRouter(monkeypatch)
    router.add("GET", "/me", {"handle": "owner"})
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs",
        _expect_payload(
            {"name": "baseline-run", "workerLabel": "0", "status": "running", "config": {"lr": 0.1}},
            {"id": "run-123", "name": "baseline-run"},
        ),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/scalars",
        _http_error(
            "/accounts/owner/projects/vision/runs/baseline-run/scalars",
            {"detail": {"error": "Invalid startLine", "expectedStartLine": 3}},
        ),
        _expect_with(assert_scalar_retry, {"status": "buffered"}),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/workers",
        _expect_payload(
            {"workerLabel": "worker-1", "status": "running"},
            {
                "id": "worker-1-id",
                "runId": "run-123",
                "workerLabel": "worker-1",
                "isPrimary": False,
                "status": "running",
                "joinedAt": "2025-01-01T00:00:00Z",
            },
        ),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/logs",
        _expect_with(assert_worker_log_payload, {"status": "buffered"}),
    )
    router.add(
        "POST",
        "/accounts/owner/projects/vision/runs/baseline-run/logs/flush",
        _expect_with(assert_flush_payload, {"status": "flushed"}),
        _expect_with(assert_flush_payload, {"status": "flushed"}),
    )
    router.add("POST", "/accounts/owner/projects/vision/runs/baseline-run/scalars/flush", {"status": "flushed"})
    router.add(
        "PUT",
        "/accounts/owner/projects/vision/runs/baseline-run",
        _expect_payload(
            {"status": "finished"},
            {"id": "run-123", "name": "baseline-run", "status": "finished", "config": {"lr": 0.1}},
        ),
    )

    backend = APIBackend(
        api_url="https://underfit.example",
        api_key="secret",
        project_name="Vision",
        run_name="BASELINE-RUN",
        run_config={"lr": 0.1},
        worker_label="0",
    )

    backend.log_scalars({"loss": 0.5}, step=4)
    backend.log_lines("worker-1", ["hello"])
    backend.finish()

    assert backend.run_name == "baseline-run"
    assert backend.scalar_line == 4
    assert backend._log_line_offsets == {"worker-1": 1}  # noqa: SLF001
    assert all("workerId" not in json.dumps(payload) for _, _, payload in router.calls if payload is not None)


def test_api_backend_attaches_to_existing_run_as_secondary_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach to a pre-existing run and register as a non-primary worker."""
    router = _APIRouter(monkeypatch)
    _mock_secondary_worker_attach(router, worker_label="gpu-02", step=1, values={"loss": 0.25})

    backend = APIBackend(
        api_url="https://underfit.example",
        api_key="secret",
        project_name="Vision",
        run_name=None,
        run_config={},
        worker_label="gpu-02",
        run_id="BASELINE-RUN",
    )

    backend.log_scalars({"loss": 0.25}, step=1)
    backend.finish()

    assert backend.run_name == "baseline-run"
    assert backend._run_id == "run-123"  # noqa: SLF001
    assert not backend._is_primary  # noqa: SLF001
    methods = [(method, path) for method, path, _ in router.calls]
    assert ("PUT", "/accounts/owner/projects/vision/runs/baseline-run/workers/gpu-02") in methods
    assert ("PUT", "/accounts/owner/projects/vision/runs/baseline-run") not in methods
