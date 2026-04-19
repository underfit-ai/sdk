"""Remote API backend for Underfit runs."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact, ArtifactDataUpload, ArtifactPathUpload
from underfit.lib.metrics import SystemMetrics
from underfit.media import Media


def _multipart_body(metadata: dict[str, Any], files: list[tuple[bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid4().hex
    buf = BytesIO()
    buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n".encode())
    buf.write(f"\r\n{json.dumps(metadata)}\r\n".encode())
    for i, (data, media_type) in enumerate(files):
        buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{i}\"\r\n".encode())
        buf.write(f"Content-Type: {media_type}\r\n\r\n".encode())
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"

class RemoteBackend:
    """Push run data to a remote Underfit API server."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        project: str,
        run_name: str,
        launch_id: str,
        run_config: dict[str, Any],
        worker_label: str,
    ) -> None:
        """Initialize a remote backend.

        Args:
            api_url: Base URL for the Underfit API.
            api_key: API key for authentication.
            project: Project identifier as either ``"<account-handle>/<project-name>"`` or a bare
                ``"<project-name>"``. Bare names resolve to projects owned by the authenticated user.
            run_name: Run name for the launch request.
            launch_id: Launch ID grouping workers in a single run.
            run_config: Run configuration payload.
            worker_label: Label identifying this worker.
        """
        self._api_url = api_url.rstrip("/") + "/api/v1"
        self._api_key = api_key
        if "/" not in project:
            user = self._request("GET", f"{self._api_url}/me", auth="api_key")
            project = f"{user['handle']}/{project}"
        self._handle, self._project_name = project.split("/", 1)
        self._runs_url = f"{self._api_url}/accounts/{self._handle}/projects/{self._project_name}/runs"
        self._log_buffer: list[dict[str, Any]] = []
        self._scalar_buffer: list[dict[str, Any]] = []
        self._next_log_line = 0
        self._next_scalar_line = 0
        self._last_scalar_timestamp: datetime | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._metrics = SystemMetrics()
        self._upload_pool = ThreadPoolExecutor(max_workers=4)

        body: dict[str, Any] = {"runName": run_name, "launchId": launch_id, "workerLabel": worker_label}
        if run_config:
            body["config"] = run_config
        resp = self._request("POST", f"{self._runs_url}/launch", body, auth="api_key")
        self.run_name = resp["name"]
        self._worker_token = resp["workerToken"]
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        if self._metrics.available:
            self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self._metrics_thread.start()

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            if self._last_scalar_timestamp is not None and timestamp <= self._last_scalar_timestamp:
                timestamp = self._last_scalar_timestamp + timedelta(microseconds=1)
            self._last_scalar_timestamp = timestamp
            ts = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
            self._scalar_buffer.append({"step": step, "values": values, "timestamp": ts})

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""
        if not lines:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._lock:
            for line in lines:
                self._log_buffer.append({"timestamp": ts, "content": line.rstrip("\n")})

    def log_media(self, key: str, step: int | None, media: Sequence[Media]) -> None:
        """Append media files for a run under a shared key and step."""
        if not media:
            return
        payload_data = asdict(media[0])
        meta: dict[str, Any] = {"key": key, "step": step, "type": payload_data["_type"]}
        excluded = {"_type", "data", "mime_type"}
        extra = {k: v for k, v in payload_data.items() if k not in excluded and v is not None}
        if extra:
            meta["metadata"] = extra
        files = [(payload.data, payload.mime_type) for payload in media]
        data, content_type = _multipart_body(meta, files)
        req = urllib.request.Request(f"{self._api_url}/ingest/media", data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self._worker_token}")
        req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req) as resp:
            resp.read()

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        """Store an artifact for a run."""
        return self._upload_pool.submit(self._upload_artifact, artifact)

    def finish(self, terminal_state: str = "finished") -> None:
        """Finalize a run and flush backend state."""
        self._stop.set()
        self._flush_thread.join()
        if hasattr(self, "_metrics_thread"):
            self._metrics_thread.join()
        self._metrics.shutdown()
        self._upload_pool.shutdown(wait=True)
        self._flush_logs()
        self._flush_scalars()
        body = {"terminalState": terminal_state}
        self._request("PUT", f"{self._api_url}/runs/terminal-state", body, auth="worker")

    def _metrics_loop(self) -> None:
        while not self._stop.wait(timeout=10.0):
            metrics = self._metrics.sample()
            if metrics:
                self.log_scalars(metrics, step=None)

    def _flush_loop(self) -> None:
        while not self._stop.wait(timeout=2.0):
            flushed = self._flush_logs()
            flushed = self._flush_scalars() or flushed
            if not flushed:
                self._request("POST", f"{self._api_url}/workers/heartbeat")

    def _flush_logs(self) -> bool:
        with self._lock:
            if not self._log_buffer:
                return False
            logs, self._log_buffer = self._log_buffer, []
        body = {"startLine": self._next_log_line, "lines": logs}
        resp = self._request("POST", f"{self._api_url}/ingest/logs", body)
        self._next_log_line = resp["nextStartLine"]
        return True

    def _flush_scalars(self) -> bool:
        with self._lock:
            if not self._scalar_buffer:
                return False
            scalars, self._scalar_buffer = self._scalar_buffer, []
        body = {"startLine": self._next_scalar_line, "scalars": scalars}
        resp = self._request("POST", f"{self._api_url}/ingest/scalars", body)
        self._next_scalar_line = resp["nextStartLine"]
        return True

    def _upload_artifact(self, artifact: Artifact) -> None:
        body: dict[str, Any] = {"name": artifact.name, "type": artifact.type}
        if artifact.metadata:
            body["metadata"] = artifact.metadata
        if artifact.step is not None:
            body["step"] = artifact.step
        created = self._request("POST", f"{self._runs_url}/{self.run_name}/artifacts", body, auth="api_key")
        for upload in artifact.uploads():
            if isinstance(upload, ArtifactPathUpload):
                file_data = Path(upload.source_path).read_bytes()
            elif isinstance(upload, ArtifactDataUpload):
                file_data = upload.data
            else:
                raise RuntimeError("Artifact upload is missing file content")
            url = f"{self._api_url}/artifacts/{created['id']}/files/{upload.path}"
            self._request_raw("PUT", url, file_data, auth="api_key")
        manifest = asdict(artifact.manifest())
        url = f"{self._api_url}/artifacts/{created['id']}/finalize"
        self._request("POST", url, {"manifest": manifest}, auth="api_key")

    def _request(
        self, method: str, url: str, body: dict[str, Any] | None = None, *, auth: str = "worker",
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        token = self._api_key if auth == "api_key" else self._worker_token
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _request_raw(self, method: str, url: str, data: bytes, *, auth: str = "worker") -> None:
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/octet-stream")
        token = self._api_key if auth == "api_key" else self._worker_token
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:
            resp.read()
