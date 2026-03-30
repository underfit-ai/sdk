"""Local filesystem backend for offline Underfit runs."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from underfit.backends.base import Backend

_ARTIFACT_FILE_NAME = "artifacts.jsonl"
_LOG_FILE_NAME = "logs.jsonl"
_META_FILE_NAME = "run.json"
_SCALAR_FILE_NAME = "scalars.jsonl"


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
        self.project_name = project_name
        self._run_name = _slug(run_name) if run_name else _default_run_name()
        self.root_dir = Path(root_dir or Path.cwd() / "underfit")
        self.run_dir = self.root_dir / _slug(project_name or "default") / self.run_name
        self.artifact_dir = self.run_dir / "artifacts"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
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
        return self._run_name

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        if not values:
            return
        record = {"step": step, "values": values, "timestamp": _now_iso()}
        self._append_jsonl(self.run_dir / _SCALAR_FILE_NAME, record)

    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        if not lines:
            return
        log_path = self.run_dir / _LOG_FILE_NAME
        for line in lines:
            self._append_jsonl(log_path, {"workerId": worker_id, "timestamp": _now_iso(), "content": line})

    def upload_artifact_entry(self, artifact_name: str, entry: dict[str, Any]) -> None:
        stored_entry = self._store_artifact_entry(artifact_name, entry)
        record = {"artifactName": artifact_name, "entry": stored_entry}
        self._append_jsonl(self.run_dir / _ARTIFACT_FILE_NAME, record)

    def read_scalars(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.run_dir / _SCALAR_FILE_NAME)

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        records = self._read_jsonl(self.run_dir / _LOG_FILE_NAME)
        if worker_id is None:
            return records
        return [record for record in records if record.get("workerId") == worker_id]

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        records = self._read_jsonl(self.run_dir / _ARTIFACT_FILE_NAME)
        if artifact_name is None:
            return records
        return [record for record in records if record.get("artifactName") == artifact_name]

    def finish(self) -> None:
        metadata = self._read_metadata()
        metadata["status"] = "finished"
        metadata["finishedAt"] = _now_iso()
        self._write_metadata(metadata)

    def _store_artifact_entry(self, artifact_name: str, entry: dict[str, Any]) -> dict[str, Any]:
        destination_root = self.artifact_dir / _slug(artifact_name)
        destination_root.mkdir(parents=True, exist_ok=True)

        kind = entry.get("kind")
        name = entry.get("name")
        if not isinstance(kind, str) or not isinstance(name, str):
            raise RuntimeError("Artifact entry is missing required kind/name fields")

        destination = destination_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)

        if kind == "file":
            source = Path(entry["path"])
            if not source.is_file():
                raise FileNotFoundError(f"artifact source file does not exist: {source}")
            shutil.copy2(source, destination)
            return {"kind": kind, "name": name, "path": str(destination)}

        if kind == "directory":
            source = Path(entry["path"])
            if not source.is_dir():
                raise FileNotFoundError(f"artifact source directory does not exist: {source}")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            return {"kind": kind, "name": name, "path": str(destination)}

        if kind == "bytes":
            data = base64.b64decode(entry["data"])
            destination.write_bytes(data)
            return {"kind": kind, "name": name, "path": str(destination)}

        if kind == "media":
            destination = destination.with_suffix(".json")
            destination.write_text(json.dumps(entry["payload"], sort_keys=True), encoding="utf-8")
            return {"kind": kind, "name": name, "path": str(destination)}

        destination = destination.with_suffix(".json")
        destination.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
        return {"kind": kind, "name": name, "path": str(destination)}

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
