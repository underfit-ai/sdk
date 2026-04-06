"""Remote API backend for Underfit runs."""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact, ArtifactDataUpload, ArtifactPathUpload
from underfit.media import Media


def _extract_media_content(payload: dict[str, Any]) -> bytes:
    if "path" in payload and payload["path"] is not None:
        return Path(payload["path"]).read_bytes()
    if "data" in payload and payload["data"] is not None:
        return base64.b64decode(payload["data"])
    if "html" in payload and payload["html"] is not None:
        return str(payload["html"]).encode("utf-8")
    raise RuntimeError("media payload is missing content")


def _multipart_body(metadata: dict[str, Any], files: list[bytes]) -> tuple[bytes, str]:
    boundary = uuid4().hex
    buf = BytesIO()
    buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n".encode())
    buf.write(f"\r\n{json.dumps(metadata)}\r\n".encode())
    for i, data in enumerate(files):
        buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{i}\"\r\n".encode())
        buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


class RemoteBackend:
    """Push run data to a remote Underfit API server.

    Args:
        api_url: Base URL for the Underfit API.
        api_key: API key for authentication.
        project: Project identifier as ``"owner/project-name"``.
        run_name: Optional run name for new runs.
        run_config: Run configuration payload.
        worker_label: Label identifying this worker.
        run_id: Run name of an existing run to join as a non-primary worker.
    """

    def __init__(  # noqa: D107
        self,
        *,
        api_url: str,
        api_key: str,
        project: str,
        run_name: str | None,
        run_config: dict[str, Any],
        worker_label: str,
        run_id: str | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._handle, self._project_name = project.split("/", 1)
        self._log_buffer: list[dict[str, Any]] = []
        self._scalar_buffer: list[dict[str, Any]] = []
        self._next_log_line = 0
        self._next_scalar_line = 0

        base = f"{self._api_url}/accounts/{self._handle}/projects/{self._project_name}"
        if run_id is None:
            body: dict[str, Any] = {"worker_label": worker_label, "config": run_config}
            if run_name:
                body["name"] = run_name.strip()
            resp = self._request("POST", f"{base}/runs", body, auth="api_key")
            self._run_name = resp["name"]
            self._run_id = resp["id"]
            self._worker_token = resp["workerToken"]
        else:
            self._run_name = run_id
            url = f"{base}/runs/{run_id}/workers"
            resp = self._request("POST", url, {"workerLabel": worker_label}, auth="api_key")
            self._run_id = resp["runId"]
            self._worker_token = resp["workerToken"]

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""
        return self._run_name

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self._scalar_buffer.append({"step": step, "values": values, "timestamp": ts})
        self._flush_scalars()

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""
        if not lines:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        for line in lines:
            self._log_buffer.append({"timestamp": ts, "content": line.rstrip("\n")})
        self._flush_logs()

    def log_media(self, key: str, step: int | None, media: list[Media]) -> None:
        """Append media files for a run under a shared key and step."""
        if not media:
            return
        payloads = [m.to_payload() for m in media]
        meta: dict[str, Any] = {"key": key, "step": step, "type": payloads[0].get("_type")}
        excluded = {"_type", "path", "data", "html"}
        extra = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        if extra:
            meta["metadata"] = extra
        data, content_type = _multipart_body(meta, [_extract_media_content(p) for p in payloads])
        req = urllib.request.Request(  # noqa: S310
            f"{self._api_url}/ingest/media", data=data, method="POST",
        )
        req.add_header("Authorization", f"Bearer {self._worker_token}")
        req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            resp.read()

    def log_artifact(self, artifact: Artifact) -> None:
        """Store an artifact for a run."""
        base = f"{self._api_url}/accounts/{self._handle}/projects/{self._project_name}"
        create_req = artifact.create_request()
        body: dict[str, Any] = {"name": create_req.name, "type": create_req.type, "run_id": self._run_id}
        if create_req.metadata:
            body["metadata"] = create_req.metadata
        created = self._request("POST", f"{base}/artifacts", body, auth="api_key")
        artifact_id = created["id"]
        for upload in artifact.uploads():
            if isinstance(upload, ArtifactPathUpload):
                file_data = Path(upload.source_path).read_bytes()
            elif isinstance(upload, ArtifactDataUpload):
                file_data = base64.b64decode(upload.data)
            else:
                raise RuntimeError("Artifact upload is missing file content")
            url = f"{self._api_url}/artifacts/{artifact_id}/files/{upload.path}"
            self._request_raw("PUT", url, file_data, auth="api_key")
        manifest = asdict(artifact.manifest())
        self._request(
            "POST", f"{self._api_url}/artifacts/{artifact_id}/finalize", {"manifest": manifest}, auth="api_key",
        )

    def finish(self) -> None:
        """Finalize a run and flush backend state."""
        self._flush_logs()
        self._flush_scalars()
        body = {"terminalState": "finished"}
        self._request("PUT", f"{self._api_url}/runs/terminal-state", body, auth="worker")

    def _flush_logs(self) -> None:
        if not self._log_buffer:
            return
        body = {"startLine": self._next_log_line, "lines": self._log_buffer}
        resp = self._request("POST", f"{self._api_url}/ingest/logs", body)
        self._next_log_line = resp["nextStartLine"]
        self._log_buffer.clear()

    def _flush_scalars(self) -> None:
        if not self._scalar_buffer:
            return
        body = {"startLine": self._next_scalar_line, "scalars": self._scalar_buffer}
        resp = self._request("POST", f"{self._api_url}/ingest/scalars", body)
        self._next_scalar_line = resp["nextStartLine"]
        self._scalar_buffer.clear()

    def _request(
        self, method: str, url: str, body: dict[str, Any] | None = None, *, auth: str = "worker",
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        req.add_header("Content-Type", "application/json")
        token = self._api_key if auth == "api_key" else self._worker_token
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            return json.loads(resp.read())

    def _request_raw(self, method: str, url: str, data: bytes, *, auth: str = "worker") -> None:
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        req.add_header("Content-Type", "application/octet-stream")
        token = self._api_key if auth == "api_key" else self._worker_token
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            resp.read()
