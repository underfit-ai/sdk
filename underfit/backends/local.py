"""Local filesystem backend for offline Underfit runs."""

from __future__ import annotations

import base64
import json
import shutil
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact, ArtifactDataUpload, ArtifactPathUpload
from underfit.lib.metrics import SystemMetrics
from underfit.media import Media


class LocalBackend:
    """Persist run data in local files for offline usage."""

    def __init__(
        self,
        *,
        project_name: str,
        run_name: str,
        run_config: dict[str, Any],
        root_dir: str | Path | None = None,
    ) -> None:
        """Initialize a local filesystem backend.

        Args:
            project_name: Project name for the run.
            run_name: Run name.
            run_config: Run configuration payload.
            root_dir: Root directory for local run data.
        """
        self._run_name = run_name
        self.run_dir = Path(root_dir or Path.cwd() / "underfit") / str(uuid4())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"project": project_name, "name": self.run_name, "config": run_config}
        (self.run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self._metrics = SystemMetrics()
        self._stop = threading.Event()
        if self._metrics.available:
            self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self._metrics_thread.start()

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""
        return self._run_name

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return
        path = self.run_dir / "scalars" / "0" / "raw.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "values": values, "timestamp": ts}, sort_keys=True))
            f.write("\n")

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""
        if not lines:
            return
        path = self.run_dir / "logs" / "log.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")

    def log_media(self, key: str, step: int | None, media: list[Media]) -> None:
        """Append media files for a run under a shared key and step."""
        if not media:
            return
        payloads = [m.to_payload() for m in media]
        dest = self.run_dir / "media" / str(uuid4())
        dest.mkdir(parents=True, exist_ok=True)
        for idx, payload in enumerate(payloads):
            (dest / str(idx)).write_bytes(self._extract_media_content(payload))
        excluded = {"_type", "path", "data", "html"}
        metadata = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        info = {"key": key, "step": step, "type": payloads[0].get("_type"), "metadata": metadata or None}
        (dest / "media.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    def log_artifact(self, artifact: Artifact) -> None:
        """Store an artifact for a run."""
        artifact_dir = self.run_dir / "artifacts" / str(uuid4())
        files_dir = artifact_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        for upload in artifact.uploads():
            if not upload.path:
                raise RuntimeError("Artifact upload is missing a valid path")
            destination = files_dir / upload.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(upload, ArtifactPathUpload):
                source = Path(upload.source_path)
                if not source.is_file():
                    raise FileNotFoundError(f"artifact source file does not exist: {source}")
                shutil.copy2(source, destination)
            elif isinstance(upload, ArtifactDataUpload):
                destination.write_bytes(base64.b64decode(upload.data))
            else:
                raise RuntimeError("Artifact upload is missing file content")

        info = {"name": artifact.name, "type": artifact.type, "metadata": artifact.metadata or None}
        (artifact_dir / "artifact.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        manifest = json.dumps(asdict(artifact.manifest()), indent=2, sort_keys=True)
        (artifact_dir / "manifest.json").write_text(manifest, encoding="utf-8")

    def _metrics_loop(self) -> None:
        while not self._stop.wait(timeout=10.0):
            metrics = self._metrics.sample()
            if metrics:
                self.log_scalars(metrics, step=None)

    def finish(self, terminal_state: str = "finished") -> None:
        """Finalize a run and flush backend state."""
        self._stop.set()
        if hasattr(self, "_metrics_thread"):
            self._metrics_thread.join()
        self._metrics.shutdown()

    @staticmethod
    def _extract_media_content(payload: dict[str, Any]) -> bytes:
        if "path" in payload and payload["path"] is not None:
            return Path(payload["path"]).read_bytes()
        if "data" in payload and payload["data"] is not None:
            return base64.b64decode(payload["data"])
        if "html" in payload and payload["html"] is not None:
            return str(payload["html"]).encode("utf-8")
        raise RuntimeError("media payload is missing content")
