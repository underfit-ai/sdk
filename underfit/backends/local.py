"""Local filesystem backend for offline Underfit runs."""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import ArtifactDataUpload, ArtifactPathUpload, ArtifactUpload
from underfit.backends.base import Backend

_META_FILE_NAME = "run.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


class LocalBackend(Backend):
    """Persist run data in local files for offline usage."""

    def __init__(
        self,
        *,
        project_name: str,
        run_name: str | None,
        run_config: dict[str, Any],
        root_dir: str | Path | None = None,
    ) -> None:
        """Initialize a local filesystem backend.

        Args:
            project_name: Project name for the run.
            run_name: Optional requested run name.
            run_config: Run configuration payload.
            root_dir: Root directory for local run data.
        """
        self.project_name = project_name
        self._run_name = run_name.strip() if isinstance(run_name, str) and run_name.strip() else _default_run_name()
        self.run_id = uuid4()
        self.root_dir = Path(root_dir or Path.cwd() / "underfit")
        self.run_dir = self.root_dir / str(self.run_id)
        self.logs_dir = self.run_dir / "logs"
        self.scalars_dir = self.run_dir / "scalars" / "0"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.media_dir = self.run_dir / "media"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.scalars_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata({
            "mode": "offline",
            "project": project_name,
            "user": "local",
            "name": self.run_name,
            "status": "running",
            "config": run_config,
            "createdAt": _now_iso(),
        })

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""
        return self._run_name

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""
        if not values:
            return
        self._append_jsonl(self.scalars_dir / "raw.jsonl", {"step": step, "values": values, "timestamp": _now_iso()})

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        """Append console log lines for a run."""
        if not lines:
            return
        path = self.logs_dir / f"{worker_id}.log"
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                if not line.endswith("\n"):
                    handle.write("\n")

    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""
        if not payloads:
            return

        media_id = uuid4()
        destination_root = self.media_dir / str(media_id)
        destination_root.mkdir(parents=True, exist_ok=True)

        for idx, payload in enumerate(payloads):
            (destination_root / str(idx)).write_bytes(self._extract_media_content(payload))

        excluded = {"_type", "path", "data", "html"}
        metadata = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        self._write_json(destination_root / "media.json", {
            "key": key,
            "step": step,
            "type": payloads[0].get("_type"),
            "metadata": metadata or None,
        })

    def log_artifact(self, artifact: Any) -> None:
        """Store an artifact for a run."""
        artifact_name = getattr(artifact, "name", None)
        artifact_type = getattr(artifact, "type", None)
        artifact_metadata = getattr(artifact, "metadata", None)
        if not isinstance(artifact_name, str) or not artifact_name:
            raise RuntimeError("Artifact is missing a valid name")
        if not isinstance(artifact_type, str) or not artifact_type:
            raise RuntimeError("Artifact is missing a valid type")

        artifact_id = uuid4()
        artifact_dir = self.artifacts_dir / str(artifact_id)
        files_dir = artifact_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        for upload in artifact.uploads():
            self._store_artifact_file(files_dir, upload)

        self._write_json(artifact_dir / "artifact.json", {
            "name": artifact_name,
            "type": artifact_type,
            "metadata": artifact_metadata or None,
        })
        self._write_json(artifact_dir / "manifest.json", asdict(artifact.manifest()))

    def read_scalars(self) -> list[dict[str, Any]]:
        """Return scalar records that were stored for a run."""
        return self._read_jsonl(self.scalars_dir / "raw.jsonl")

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        """Return log records, optionally filtered by worker id."""
        paths = [self.logs_dir / f"{worker_id}.log"] if worker_id else sorted(self.logs_dir.glob("*.log"))
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            worker_label = path.stem
            lines = path.read_text(encoding="utf-8").splitlines()
            records.extend({"workerId": worker_label, "content": line} for line in lines)
        return records

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact entries, optionally filtered by artifact name."""
        records: list[dict[str, Any]] = []
        for artifact_dir in sorted(self.artifacts_dir.iterdir()) if self.artifacts_dir.exists() else []:
            if not artifact_dir.is_dir():
                continue
            metadata_path = artifact_dir / "artifact.json"
            manifest_path = artifact_dir / "manifest.json"
            if not metadata_path.exists() or not manifest_path.exists():
                continue
            metadata = self._read_json(metadata_path)
            name = metadata.get("name")
            if artifact_name is not None and name != artifact_name:
                continue
            manifest = self._read_json(manifest_path)
            files_dir = artifact_dir / "files"
            for file_path in manifest.get("files", []):
                records.append({
                    "artifactId": artifact_dir.name,
                    "artifactName": name,
                    "entry": {"kind": "file", "name": file_path, "path": str(files_dir / file_path)},
                })
            for reference in manifest.get("references", []):
                records.append({
                    "artifactId": artifact_dir.name,
                    "artifactName": name,
                    "entry": {"kind": "reference", **reference},
                })
        return records

    def finish(self) -> None:
        """Finalize a run and flush backend state."""
        metadata = self._read_metadata()
        metadata["status"] = "finished"
        metadata["finishedAt"] = _now_iso()
        self._write_metadata(metadata)

    def _store_artifact_file(self, files_dir: Path, upload: ArtifactUpload) -> None:
        artifact_path = upload.path
        if not artifact_path:
            raise RuntimeError("Artifact upload is missing a valid path")

        destination = files_dir / artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(upload, ArtifactPathUpload):
            source = Path(upload.source_path)
            if not source.is_file():
                raise FileNotFoundError(f"artifact source file does not exist: {source}")
            shutil.copy2(source, destination)
            return
        if isinstance(upload, ArtifactDataUpload):
            destination.write_bytes(base64.b64decode(upload.data))
            return
        raise RuntimeError("Artifact upload is missing file content")

    @staticmethod
    def _extract_media_content(payload: dict[str, Any]) -> bytes:
        if "path" in payload and payload["path"] is not None:
            return Path(payload["path"]).read_bytes()
        if "data" in payload and payload["data"] is not None:
            return base64.b64decode(payload["data"])
        if "html" in payload and payload["html"] is not None:
            return str(payload["html"]).encode("utf-8")
        raise RuntimeError("media payload is missing content")

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
            return [json.loads(line) for line in lines if line]

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_metadata(self) -> dict[str, Any]:
        path = self.run_dir / _META_FILE_NAME
        return self._read_json(path) if path.exists() else {}

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_metadata(self, payload: dict[str, Any]) -> None:
        self._write_json(self.run_dir / _META_FILE_NAME, payload)
