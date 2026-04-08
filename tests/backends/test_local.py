"""Tests for the local backend."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from underfit import Artifact, Html
from underfit.backends.local import LocalBackend


def test_local_backend_writes_backfill_layout(tmp_path: Path) -> None:
    """Write local run data in the layout consumed by the API backfill service."""
    backend = LocalBackend(
        project_name="Vision", run_name="Trial A", run_config={"lr": 0.01}, worker_label="worker-7", root_dir=tmp_path,
    )

    backend.log_scalars({"loss": 0.8}, step=1)
    backend.log_lines(["hello", "world\n"])

    artifact = Artifact("dataset-v1", "dataset", metadata={"format": "json"})
    artifact.add_bytes(b"{}", name="payload.json")
    backend.log_artifact(artifact)

    backend.log_media("samples", 7, [Html("<h1>ok</h1>", caption="summary")])
    backend.finish()

    UUID(backend.run_dir.name)
    metadata = json.loads((backend.run_dir / "run.json").read_text())
    assert metadata["project"] == "Vision"
    assert metadata["name"] == "Trial A"
    assert metadata["config"] == {"lr": 0.01}
    assert metadata["terminal_state"] == "finished"
    assert set(metadata.keys()) == {"project", "name", "config", "terminal_state"}

    scalar_path = backend.run_dir / "scalars" / "worker-7" / "r1" / "0.jsonl"
    scalar_lines = [json.loads(line) for line in scalar_path.read_text().splitlines()]
    assert scalar_lines == [{"step": 1, "values": {"loss": 0.8}, "timestamp": scalar_lines[0]["timestamp"]}]

    assert (backend.run_dir / "logs" / "worker-7" / "segments" / "0.log").read_text() == "hello\nworld\n"

    artifact_dirs = [p for p in (backend.run_dir / "artifacts").iterdir() if p.is_dir()]
    assert len(artifact_dirs) == 1
    artifact_dir = artifact_dirs[0]
    UUID(artifact_dir.name)
    artifact_meta = json.loads((artifact_dir / "artifact.json").read_text())
    assert artifact_meta == {"metadata": {"format": "json"}, "name": "dataset-v1", "type": "dataset"}
    assert json.loads((artifact_dir / "manifest.json").read_text()) == {"files": ["payload.json"], "references": []}
    assert (artifact_dir / "files" / "payload.json").read_bytes() == b"{}"

    media_dirs = [p for p in (backend.run_dir / "media").iterdir() if p.is_dir()]
    assert len(media_dirs) == 1
    media_dir = media_dirs[0]
    UUID(media_dir.name)
    assert (media_dir / "0").read_text() == "<h1>ok</h1>"
    media_meta = json.loads((media_dir / "media.json").read_text())
    expected = {"key": "samples", "metadata": {"caption": "summary", "inject": True}, "step": 7, "type": "html"}
    assert media_meta == expected
