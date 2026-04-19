"""Local filesystem backend for offline Underfit runs."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact, ArtifactDataUpload, ArtifactPathUpload
from underfit.lib.metrics import SystemMetrics
from underfit.media import Html, Media


class LocalBackend:
    """Persist run data in local files for offline usage."""

    _RAW_SCALAR_RESOLUTION = 1

    def __init__(
        self,
        *,
        project_name: str,
        run_name: str,
        run_config: dict[str, Any],
        worker_label: str,
        root_dir: str | Path | None = None,
    ) -> None:
        """Initialize a local filesystem backend.

        Args:
            project_name: Project name for the run.
            run_name: Run name.
            run_config: Run configuration payload.
            worker_label: Label identifying this worker.
            root_dir: Root directory for local run data.
        """
        self.run_name = run_name
        self._worker_label = worker_label
        self.run_dir = Path(root_dir or Path.cwd() / "underfit") / str(uuid4())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._run_meta: dict[str, Any] = {
            "project": project_name, "name": self.run_name, "config": run_config, "summary": {},
        }
        self._write_run_meta()
        self._metrics = SystemMetrics(worker_label)
        self._scalar_lock = threading.Lock()
        self._stop = threading.Event()
        if self._metrics.available:
            self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self._metrics_thread.start()

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return
        path = self.run_dir / "scalars" / self._worker_label / f"r{self._RAW_SCALAR_RESOLUTION}" / "0.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._scalar_lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "values": values, "timestamp": ts}, sort_keys=True) + "\n")
            self._run_meta["summary"] = dict(values)
            self._write_run_meta()

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""
        if not lines:
            return
        path = self.run_dir / "logs" / self._worker_label / "segments" / "0.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")

    def log_media(self, key: str, step: int | None, media: Sequence[Media]) -> None:
        """Append media files for a run under a shared key and step."""
        if not media:
            return
        media_type = str(asdict(media[0]).get("_type") or "unknown")
        for idx, payload in enumerate(media):
            path = self.run_dir / "media" / media_type / f"{key}_{step}_{idx}{self._media_suffix(payload)}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload.data)

    def log_artifact(self, artifact: Artifact) -> Future[None]:
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
                destination.write_bytes(upload.data)
            else:
                raise RuntimeError("Artifact upload is missing file content")

        info = {"name": artifact.name, "type": artifact.type, "metadata": artifact.metadata or None}
        if artifact.step is not None:
            info["step"] = artifact.step
        (artifact_dir / "artifact.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        manifest = json.dumps(asdict(artifact.manifest()), indent=2, sort_keys=True)
        (artifact_dir / "manifest.json").write_text(manifest, encoding="utf-8")
        future: Future[None] = Future()
        future.set_result(None)
        return future

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
        self._run_meta["terminal_state"] = terminal_state
        self._write_run_meta()

    def _write_run_meta(self) -> None:
        (self.run_dir / "run.json").write_text(json.dumps(self._run_meta, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _media_suffix(payload: Media) -> str:
        if (file_type := getattr(payload, "file_type", None)) is not None:
            return f".{str(file_type).lstrip('.')}"
        if isinstance(payload, Html):
            return ".html"
        return ".bin"
