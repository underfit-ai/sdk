"""Local filesystem backend for offline Underfit runs."""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from underfit.artifact import ArtifactDataUpload, ArtifactPathUpload, ArtifactUpload
from underfit.backends.base import Backend

_ARTIFACT_FILE_NAME = "artifacts.jsonl"
_LOG_FILE_NAME = "logs.jsonl"
_META_FILE_NAME = "run.json"
_SCALAR_FILE_NAME = "scalars.jsonl"
_MEDIA_FILE_NAME = "media.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _slug(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "-")
    clean = "".join(char for char in lowered if char.isalnum() or char in {"-", "_"})
    return clean or "default"


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
        self._run_name = _slug(run_name) if run_name else _default_run_name()
        self.root_dir = Path(root_dir or Path.cwd() / "underfit")
        self.run_dir = self.root_dir / _slug(project_name or "default") / self.run_name
        self.artifact_dir = self.run_dir / "artifacts"
        self.media_dir = self.run_dir / "media"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata({
            "mode": "offline",
            "project": project_name,
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
        record = {"step": step, "values": values, "timestamp": _now_iso()}
        self._append_jsonl(self.run_dir / _SCALAR_FILE_NAME, record)

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        """Append console log lines for a run."""
        if not lines:
            return
        log_path = self.run_dir / _LOG_FILE_NAME
        for line in lines:
            self._append_jsonl(log_path, {"workerId": worker_id, "timestamp": _now_iso(), "content": line})

    def log_artifact(self, artifact: Any) -> None:
        """Store an artifact for a run."""
        artifact_name = getattr(artifact, "name", None)
        if not isinstance(artifact_name, str) or not artifact_name:
            raise RuntimeError("Artifact is missing a valid name")

        for upload in artifact.uploads():
            stored_entry = self._store_artifact_file(artifact_name, upload)
            self._append_jsonl(
                self.run_dir / _ARTIFACT_FILE_NAME,
                {"artifactName": artifact_name, "entry": stored_entry},
            )

        for reference in artifact.manifest().references:
            record = {"artifactName": artifact_name, "entry": {"kind": "reference", **asdict(reference)}}
            self._append_jsonl(self.run_dir / _ARTIFACT_FILE_NAME, record)

    def read_scalars(self) -> list[dict[str, Any]]:
        """Return scalar records that were stored for a run."""
        return self._read_jsonl(self.run_dir / _SCALAR_FILE_NAME)

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        """Return log records, optionally filtered by worker id."""
        records = self._read_jsonl(self.run_dir / _LOG_FILE_NAME)
        if worker_id is None:
            return records
        return [record for record in records if record.get("workerId") == worker_id]

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact entries, optionally filtered by artifact name."""
        records = self._read_jsonl(self.run_dir / _ARTIFACT_FILE_NAME)
        if artifact_name is None:
            return records
        return [record for record in records if record.get("artifactName") == artifact_name]

    def finish(self) -> None:
        """Finalize a run and flush backend state."""
        metadata = self._read_metadata()
        metadata["status"] = "finished"
        metadata["finishedAt"] = _now_iso()
        self._write_metadata(metadata)

    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""
        if not payloads:
            return

        media_id = _slug(key) + "-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        destination_root = self.media_dir / media_id
        destination_root.mkdir(parents=True, exist_ok=True)

        stored_files: list[str] = []
        for idx, payload in enumerate(payloads):
            filename, content = self._extract_media_content(payload, key, idx)
            dest = destination_root / str(idx)
            dest.write_bytes(content)
            stored_files.append(str(dest))

        excluded = {"_type", "path", "data", "html"}
        metadata: dict[str, Any] = {k: v for k, v in payloads[0].items() if k not in excluded and v is not None}
        record = {
            "id": media_id,
            "key": key,
            "step": step,
            "type": payloads[0].get("_type"),
            "files": stored_files,
            "metadata": metadata or None,
            "createdAt": _now_iso(),
        }
        self._append_jsonl(self.run_dir / _MEDIA_FILE_NAME, record)

    def _store_artifact_file(self, artifact_name: str, upload: ArtifactUpload) -> dict[str, Any]:
        destination_root = self.artifact_dir / _slug(artifact_name)
        destination_root.mkdir(parents=True, exist_ok=True)

        artifact_path = upload.path
        if not artifact_path:
            raise RuntimeError("Artifact upload is missing a valid path")

        destination = destination_root / artifact_path
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
        return {"kind": "file", "name": artifact_path, "path": str(destination)}

    @staticmethod
    def _extract_media_content(payload: dict[str, Any], key: str, index: int) -> tuple[str, bytes]:
        if "path" in payload and payload["path"] is not None:
            path = Path(payload["path"])
            return path.name, path.read_bytes()
        if "data" in payload and payload["data"] is not None:
            filename = f"{key}-{index}.{payload.get('file_type') or 'bin'}"
            return filename, base64.b64decode(payload["data"])
        if "html" in payload and payload["html"] is not None:
            return f"{key}-{index}.html", str(payload["html"]).encode("utf-8")
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

    def _read_metadata(self) -> dict[str, Any]:
        path = self.run_dir / _META_FILE_NAME
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _write_metadata(self, payload: dict[str, Any]) -> None:
        path = self.run_dir / _META_FILE_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
