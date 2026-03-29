"""HTTP backend client for Underfit."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from underfit.backends.base import Backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_api_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized
    return f"{normalized}/api/v1"


class APIBackend(Backend):
    """Send run data to the remote Underfit API."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        account_handle: str,
        project_name: str,
        run_name: str | None,
        run_config: dict[str, Any],
    ) -> None:
        self.api_url = _normalize_api_url(api_url)
        self.api_key = api_key
        self.account_handle = account_handle.lower()
        self.project_name = project_name.lower()
        self._run_name = run_name.lower() if run_name else None
        self.scalar_line = 0
        self._log_line_offsets: dict[str, int] = {"stdout": 0, "stderr": 0}
        self._create_run(run_config)

    @property
    def run_name(self) -> str:
        if self._run_name is None:
            raise RuntimeError("Run name is not initialized")
        return self._run_name

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310
            url=f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                body = response.read().decode("utf-8")
                return {} if not body else json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            details = body
            if body:
                try:
                    details = json.dumps(json.loads(body), sort_keys=True)
                except json.JSONDecodeError:
                    details = body
            raise RuntimeError(f"Underfit request failed ({e.code} {method} {path}): {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Underfit request failed ({method} {path}): {e.reason}") from e

    def _base_run_path(self) -> str:
        return f"/accounts/{self.account_handle}/projects/{self.project_name}/runs/{self.run_name}"

    def _create_run(self, run_config: dict[str, Any]) -> None:
        response = self._request(
            "POST",
            f"/accounts/{self.account_handle}/projects/{self.project_name}/runs",
            {"status": "running", "config": run_config or None},
        )
        name = response.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Underfit run creation response did not include a run name")
        self._run_name = name.lower()

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        if not values:
            return

        payload = {
            "startLine": self.scalar_line,
            "scalars": [{"step": step, "values": values, "timestamp": _now_iso()}],
        }
        response = self._request("POST", f"{self._base_run_path()}/scalars", payload)
        if response.get("status") == "buffered":
            self.scalar_line += 1
            return

        expected = response.get("expectedStartLine")
        if isinstance(expected, int):
            payload["startLine"] = expected
            retry = self._request("POST", f"{self._base_run_path()}/scalars", payload)
            if retry.get("status") == "buffered":
                self.scalar_line = expected + 1
                return

        raise RuntimeError("Failed to append scalars to Underfit API")

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        if not lines:
            return

        start_line = self._log_line_offsets.get(worker_id, 0)
        payload = {
            "workerId": worker_id,
            "startLine": start_line,
            "lines": [{"timestamp": _now_iso(), "content": line} for line in lines],
        }
        response = self._request("POST", f"{self._base_run_path()}/logs", payload)
        if response.get("status") == "buffered":
            self._log_line_offsets[worker_id] = start_line + len(lines)
            return

        expected = response.get("expectedStartLine")
        if isinstance(expected, int):
            payload["startLine"] = expected
            retry = self._request("POST", f"{self._base_run_path()}/logs", payload)
            if retry.get("status") == "buffered":
                self._log_line_offsets[worker_id] = expected + len(lines)
                return

        raise RuntimeError("Failed to append logs to Underfit API")

    def upload_artifact_entry(self, artifact_name: str, entry: dict[str, Any]) -> None:
        _ = artifact_name, entry
        # Placeholder until artifact upload API shape is finalized.
        return

    def read_scalars(self) -> list[dict[str, Any]]:
        raise NotImplementedError("Reading scalars is not implemented for API backend")

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        _ = worker_id
        raise NotImplementedError("Reading logs is not implemented for API backend")

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        _ = artifact_name
        raise NotImplementedError("Reading artifact entries is not implemented for API backend")

    def finish(self) -> None:
        for worker_id in ("stdout", "stderr"):
            self._request("POST", f"{self._base_run_path()}/logs/flush", {"workerId": worker_id})
        self._request("POST", f"{self._base_run_path()}/scalars/flush", {})
        self._request("PUT", f"{self._base_run_path()}", {"status": "finished"})
