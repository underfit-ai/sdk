"""HTTP backend client for Underfit."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from underfit.artifact import ArtifactDataUpload, ArtifactPathUpload, ArtifactUpload
from underfit.backends.base import Backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_api_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    return normalized if normalized.endswith("/api/v1") else f"{normalized}/api/v1"


class _RequestError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int | None, details: str, payload: Any = None) -> None:
        label = f"{status_code} {method} {path}" if status_code is not None else f"{method} {path}"
        super().__init__(f"Underfit request failed ({label}): {details}")
        self.status_code = status_code
        self.payload = payload


class APIBackend(Backend):
    """Send run data to the remote Underfit API."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        project_name: str,
        run_name: str | None,
        run_config: dict[str, Any],
        worker_label: str,
        run_id: str | None = None,
    ) -> None:
        """Initialize an API-backed run transport.

        Args:
            api_url: Base Underfit API URL.
            api_key: API token used for authentication.
            project_name: Project name for the run.
            run_name: Optional requested run name.
            run_config: Run configuration payload.
            worker_label: Label identifying this worker within the run.
            run_id: Optional identifier of an existing run to attach to as a
                non-primary worker. When set, ``run_name`` and ``run_config``
                are ignored and no new run is created.

        Raises:
            RuntimeError: If the API request to initialize or attach to the run fails.
        """
        self.api_url = _normalize_api_url(api_url)
        self.api_key = api_key
        self.account_handle = self._resolve_account_handle()
        self.project_name = project_name.lower()
        self.worker_label = worker_label
        self.scalar_line = 0
        self._run_id: str | None = None
        self._run_name = run_name.lower() if run_name else None
        self._is_primary = run_id is None
        self._registered_workers: set[str] = set()
        self._log_line_offsets: dict[str, int] = {}
        if run_id is None:
            self._create_run(run_config)
            self._registered_workers.add(worker_label)
        else:
            self._attach_to_run(run_id)
            self._ensure_worker(worker_label)

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""
        if self._run_name is None:
            raise RuntimeError("Run name is not initialized")
        return self._run_name

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310
            url=f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        return self._send_request(request, method, path, timeout=10)

    def _request_multipart(self, method: str, path: str, body: bytes, boundary: str) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310
            url=f"{self.api_url}{path}",
            data=body,
            method=method,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response = self._send_request(request, method, path, timeout=30)
        if not isinstance(response, dict):
            raise RuntimeError(f"Underfit request returned an unexpected response type for {method} {path}")
        return response

    def _request_bytes(
        self,
        method: str,
        path: str,
        body: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310
            url=f"{self.api_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": content_type, "Authorization": f"Bearer {self.api_key}"},
        )
        response = self._send_request(request, method, path, timeout=30)
        if not isinstance(response, dict):
            raise RuntimeError(f"Underfit request returned an unexpected response type for {method} {path}")
        return response

    def _send_request(self, request: urllib.request.Request, method: str, path: str, timeout: int) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8")
                return {} if not body else json.loads(body)
        except urllib.error.HTTPError as e:
            payload, details = self._decode_error_body(e.read().decode("utf-8"))
            raise _RequestError(method, path, e.code, details, payload) from e
        except urllib.error.URLError as e:
            raise _RequestError(method, path, None, str(e.reason)) from e

    @staticmethod
    def _decode_error_body(body: str) -> tuple[Any, str]:
        if not body:
            return None, ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body, body
        return payload, json.dumps(payload, sort_keys=True)

    @staticmethod
    def _expected_start_line(error: _RequestError) -> int | None:
        payload = error.payload
        if not isinstance(payload, dict):
            return None
        detail = payload.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("expectedStartLine"), int):
            return detail["expectedStartLine"]
        if isinstance(payload.get("expectedStartLine"), int):
            return payload["expectedStartLine"]
        return None

    def _build_media_file(self, key: str, payload: dict[str, Any]) -> tuple[str, bytes, str]:
        if "path" in payload and payload["path"] is not None:
            path = Path(payload["path"])
            content = path.read_bytes()
            filename = path.name
        elif "data" in payload and payload["data"] is not None:
            content = base64.b64decode(payload["data"])
            filename = f"{key}.{payload.get('file_type') or 'bin'}"
        elif "html" in payload and payload["html"] is not None:
            content = str(payload["html"]).encode("utf-8")
            filename = f"{key}.html"
        else:
            raise RuntimeError("media payload is missing content")

        mime, _ = guess_type(filename)
        return filename, content, mime or "application/octet-stream"

    def _encode_multipart(self, files: list[tuple[str, bytes, str]], metadata: dict[str, Any]) -> tuple[bytes, str]:
        boundary = f"----underfit-{uuid.uuid4().hex}"
        parts: list[bytes] = []

        for filename, content, content_type in files:
            headers = [
                f"--{boundary}",
                f"Content-Disposition: form-data; name=\"files\"; filename=\"{filename}\"",
                f"Content-Type: {content_type}",
                "",
            ]
            parts.append("\r\n".join(headers).encode("utf-8"))
            parts.append(content)
            parts.append(b"\r\n")

        metadata_headers = [
            f"--{boundary}",
            "Content-Disposition: form-data; name=\"metadata\"",
            "Content-Type: application/json",
            "",
        ]
        parts.append("\r\n".join(metadata_headers).encode("utf-8"))
        parts.append(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return b"".join(parts), boundary

    def _resolve_account_handle(self) -> str:
        response = self._request("GET", "/me")
        handle = response.get("handle") if isinstance(response, dict) else None
        if not isinstance(handle, str) or not handle:
            raise RuntimeError("Underfit /me response missing handle")
        return handle.lower()

    def _base_run_path(self) -> str:
        return f"/accounts/{self.account_handle}/projects/{self.project_name}/runs/{self.run_name}"

    def _create_run(self, run_config: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"workerLabel": self.worker_label, "status": "running", "config": run_config or None}
        if self._run_name is not None:
            payload["name"] = self._run_name
        response = self._request(
            "POST",
            f"/accounts/{self.account_handle}/projects/{self.project_name}/runs",
            payload,
        )
        run_id = response.get("id") if isinstance(response, dict) else None
        name = response.get("name") if isinstance(response, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Underfit run creation response did not include a run id")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Underfit run creation response did not include a run name")
        self._run_id = run_id
        self._run_name = name.lower()

    def _attach_to_run(self, run_id: str) -> None:
        self._run_name = run_id.lower()
        response = self._request("GET", self._base_run_path())
        resolved_id = response.get("id") if isinstance(response, dict) else None
        resolved_name = response.get("name") if isinstance(response, dict) else None
        if not isinstance(resolved_id, str) or not resolved_id:
            raise RuntimeError("Underfit run lookup response did not include a run id")
        if not isinstance(resolved_name, str) or not resolved_name:
            raise RuntimeError("Underfit run lookup response did not include a run name")
        self._run_id = resolved_id
        self._run_name = resolved_name.lower()
    def _ensure_worker(self, worker_label: str) -> None:
        if worker_label in self._registered_workers:
            return
        try:
            self._request(
                "POST",
                f"{self._base_run_path()}/workers",
                {"workerLabel": worker_label, "status": "running"},
            )
        except _RequestError as e:
            if e.status_code != 409:
                raise
        self._registered_workers.add(worker_label)

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return

        payload = {
            "workerLabel": self.worker_label,
            "startLine": self.scalar_line,
            "scalars": [{"step": step, "values": values, "timestamp": _now_iso()}],
        }
        try:
            response = self._request("POST", f"{self._base_run_path()}/scalars", payload)
        except _RequestError as e:
            expected = self._expected_start_line(e)
            if e.status_code != 409 or expected is None:
                raise
            payload["startLine"] = expected
            response = self._request("POST", f"{self._base_run_path()}/scalars", payload)
            self.scalar_line = expected + len(payload["scalars"])
        else:
            self.scalar_line += len(payload["scalars"])

        if not isinstance(response, dict) or response.get("status") != "buffered":
            raise RuntimeError("Failed to append scalars to Underfit API")

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        """Append console log lines for a run."""
        if not lines:
            return

        self._ensure_worker(worker_id)
        start_line = self._log_line_offsets.get(worker_id, 0)
        payload = {
            "workerLabel": worker_id,
            "startLine": start_line,
            "lines": [{"timestamp": _now_iso(), "content": line} for line in lines],
        }
        try:
            response = self._request("POST", f"{self._base_run_path()}/logs", payload)
        except _RequestError as e:
            expected = self._expected_start_line(e)
            if e.status_code != 409 or expected is None:
                raise
            payload["startLine"] = expected
            response = self._request("POST", f"{self._base_run_path()}/logs", payload)
            self._log_line_offsets[worker_id] = expected + len(lines)
        else:
            self._log_line_offsets[worker_id] = start_line + len(lines)

        if not isinstance(response, dict) or response.get("status") != "buffered":
            raise RuntimeError("Failed to append logs to Underfit API")

    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""
        if not payloads:
            return

        media_type = payloads[0].get("_type")
        if any(payload.get("_type") != media_type for payload in payloads):
            raise RuntimeError("All media payloads in a batch must share the same type")
        if media_type not in {"image", "video", "audio", "html"}:
            raise RuntimeError(f"unsupported media type: {media_type}")

        files = [self._build_media_file(f"{key}-{idx}", payload) for idx, payload in enumerate(payloads)]
        excluded = {"_type", "path", "data", "html"}
        metadata_fields = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        body, boundary = self._encode_multipart(files, {
            "key": key,
            "step": step,
            "type": media_type,
            "metadata": metadata_fields or None,
        })
        self._request_multipart("POST", f"{self._base_run_path()}/media", body, boundary)

    def log_artifact(self, artifact: Any) -> None:
        """Store an artifact for a run."""
        if self._run_id is None:
            raise RuntimeError("Run id is not initialized")

        created = self._request(
            "POST",
            f"/accounts/{self.account_handle}/projects/{self.project_name}/artifacts",
            {"run_id": self._run_id, **asdict(artifact.create_request())},
        )
        artifact_id = created.get("id") if isinstance(created, dict) else None
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RuntimeError("Underfit artifact creation response did not include an id")

        for upload in artifact.uploads():
            artifact_path = upload.path
            if not artifact_path:
                raise RuntimeError("Artifact upload is missing a valid path")
            path = f"/artifacts/{artifact_id}/files/{urllib.parse.quote(artifact_path, safe='/')}"
            self._request_bytes("PUT", path, self._artifact_bytes(upload))

        self._request("POST", f"/artifacts/{artifact_id}/finalize", {"manifest": asdict(artifact.manifest())})

    def _artifact_bytes(self, upload: ArtifactUpload) -> bytes:
        if isinstance(upload, ArtifactPathUpload):
            return Path(upload.source_path).read_bytes()
        if isinstance(upload, ArtifactDataUpload):
            return base64.b64decode(upload.data)
        raise RuntimeError("Artifact upload is missing file content")

    def read_scalars(self) -> list[dict[str, Any]]:
        """Return scalar records that were stored for a run."""
        raise NotImplementedError("Reading scalars is not implemented for API backend")

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        """Return log records, optionally filtered by worker id."""
        _ = worker_id
        raise NotImplementedError("Reading logs is not implemented for API backend")

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact entries, optionally filtered by artifact name."""
        _ = artifact_name
        raise NotImplementedError("Reading artifact entries is not implemented for API backend")

    def finish(self) -> None:
        """Finalize a run and flush backend state."""
        for worker_label in sorted(self._registered_workers):
            self._request("POST", f"{self._base_run_path()}/logs/flush", {"workerLabel": worker_label})
        self._request("POST", f"{self._base_run_path()}/scalars/flush", {"workerLabel": self.worker_label})
        if self._is_primary:
            self._request("PUT", f"{self._base_run_path()}", {"status": "finished"})
        else:
            self._request(
                "PUT",
                f"{self._base_run_path()}/workers/{self.worker_label}",
                {"status": "finished"},
            )
