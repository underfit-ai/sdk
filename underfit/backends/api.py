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
        project_name: str,
        run_name: str | None,
        run_config: dict[str, Any],
    ) -> None:
        """Initialize an API-backed run transport.

        Args:
            api_url: Base Underfit API URL.
            api_key: API token used for authentication.
            project_name: Project name for the run.
            run_name: Optional requested run name.
            run_config: Run configuration payload.

        Raises:
            RuntimeError: If the API request to initialize the run fails.
        """
        self.api_url = _normalize_api_url(api_url)
        self.api_key = api_key
        self.account_handle = self._resolve_account_handle()
        self.project_name = project_name.lower()
        self.scalar_line = 0
        self._run_id: str | None = None
        self._run_name = run_name.lower() if run_name else None
        self._log_line_offsets: dict[str, int] = {"stdout": 0, "stderr": 0}
        self._create_run(run_config)

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""
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

    def _request_multipart(self, method: str, path: str, body: bytes, boundary: str) -> dict[str, Any]:
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(  # noqa: S310
            url=f"{self.api_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                body_text = response.read().decode("utf-8")
                return {} if not body_text else json.loads(body_text)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            details = error_body
            if error_body:
                try:
                    details = json.dumps(json.loads(error_body), sort_keys=True)
                except json.JSONDecodeError:
                    details = error_body
            raise RuntimeError(f"Underfit request failed ({e.code} {method} {path}): {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Underfit request failed ({method} {path}): {e.reason}") from e

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
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                response_body = response.read().decode("utf-8")
                return {} if not response_body else json.loads(response_body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            details = error_body
            if error_body:
                try:
                    details = json.dumps(json.loads(error_body), sort_keys=True)
                except json.JSONDecodeError:
                    details = error_body
            raise RuntimeError(f"Underfit request failed ({e.code} {method} {path}): {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Underfit request failed ({method} {path}): {e.reason}") from e

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
            part_headers = [
                f"--{boundary}",
                f"Content-Disposition: form-data; name=\"files\"; filename=\"{filename}\"",
                f"Content-Type: {content_type}",
                "",
            ]
            parts.append("\r\n".join(part_headers).encode("utf-8"))
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

        body = b"".join(parts)
        return body, boundary

    def _resolve_account_handle(self) -> str:
        response = self._request("GET", "/me")
        handle = response.get("handle")
        if not isinstance(handle, str) or not handle:
            raise RuntimeError("Underfit /me response missing handle")
        return handle.lower()

    def _base_run_path(self) -> str:
        return f"/accounts/{self.account_handle}/projects/{self.project_name}/runs/{self.run_name}"

    def _create_run(self, run_config: dict[str, Any]) -> None:
        response = self._request(
            "POST",
            f"/accounts/{self.account_handle}/projects/{self.project_name}/runs",
            {"status": "running", "config": run_config or None},
        )
        run_id = response.get("id")
        name = response.get("name")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Underfit run creation response did not include a run id")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Underfit run creation response did not include a run name")
        self._run_id = run_id
        self._run_name = name.lower()

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
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
        """Append console log lines for a run."""
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

    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""
        if not payloads:
            return

        media_type = payloads[0].get("_type")
        if any(payload.get("_type") != media_type for payload in payloads):
            raise RuntimeError("All media payloads in a batch must share the same type")
        if media_type not in {"image", "video", "audio", "html"}:
            raise RuntimeError(f"unsupported media type: {media_type}")

        files = []
        for idx, payload in enumerate(payloads):
            name = f"{key}-{idx}"
            files.append(self._build_media_file(name, payload))

        excluded = {"_type", "path", "data", "html"}
        metadata_fields = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        metadata_payload = {"key": key, "step": step, "type": media_type, "metadata": metadata_fields or None}

        body, boundary = self._encode_multipart(files, metadata_payload)
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
        artifact_id = created.get("id")
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
        for worker_id in ("stdout", "stderr"):
            self._request("POST", f"{self._base_run_path()}/logs/flush", {"workerId": worker_id})
        self._request("POST", f"{self._base_run_path()}/scalars/flush", {})
        self._request("PUT", f"{self._base_run_path()}", {"status": "finished"})
