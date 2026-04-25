"""Tests for the local client."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from underfit import Artifact, Html
from underfit.clients.local import LocalClient


def test_local_client_writes_backfill_layout(tmp_path: Path) -> None:
    """Write local run data in the layout consumed by the API backfill service."""
    client = LocalClient(project="Vision", root_dir=tmp_path)
    client.launch_run(run_name="Trial A", run_config={"lr": 0.01}, worker_label="worker-7")

    client.log_scalars({"loss": 0.8}, step=1)
    client.log_lines(["hello", "world\n"])

    artifact = Artifact("dataset-v1", "dataset", metadata={"format": "json"})
    artifact.add_bytes(b"{}", name="payload.json")
    client.log_artifact(artifact)

    client.log_media("samples", 7, [Html("<h1>ok</h1>", caption="summary")])
    client.finish()

    UUID(client.run_dir.name)
    metadata = json.loads((client.run_dir / "run.json").read_text())
    assert metadata["project"] == "Vision"
    assert metadata["name"] == "Trial A"
    assert metadata["config"] == {"lr": 0.01}
    assert metadata["terminal_state"] == "finished"
    assert metadata["summary"] == {"loss": 0.8}
    assert set(metadata.keys()) == {"project", "name", "config", "terminal_state", "summary"}

    scalar_path = client.run_dir / "scalars" / "worker-7" / "r1" / "0.jsonl"
    scalar_lines = [json.loads(line) for line in scalar_path.read_text().splitlines()]
    assert scalar_lines == [{"step": 1, "values": {"loss": 0.8}, "timestamp": scalar_lines[0]["timestamp"]}]

    assert (client.run_dir / "logs" / "worker-7" / "segments" / "0.log").read_text() == "hello\nworld\n"

    artifact_dirs = [p for p in (client.run_dir / "artifacts").iterdir() if p.is_dir()]
    assert len(artifact_dirs) == 1
    artifact_dir = artifact_dirs[0]
    UUID(artifact_dir.name)
    artifact_meta = json.loads((artifact_dir / "artifact.json").read_text())
    assert artifact_meta == {"metadata": {"format": "json"}, "name": "dataset-v1", "type": "dataset"}
    assert json.loads((artifact_dir / "manifest.json").read_text()) == {"files": ["payload.json"], "references": []}
    assert (artifact_dir / "files" / "payload.json").read_bytes() == b"{}"

    media_path = client.run_dir / "media" / "html" / "samples_7_0.html"
    assert media_path.read_text() == "<h1>ok</h1>"


def test_local_client_writes_project_artifact_under_project_dir(tmp_path: Path) -> None:
    """Project-level artifacts land in the layout the API backfill walks for project artifacts."""
    client = LocalClient(project="Vision", root_dir=tmp_path)

    artifact = Artifact("eval-set", "dataset")
    artifact.add_bytes(b"{}", name="payload.json")
    client.log_project_artifact(client.project, artifact).result()

    [artifact_dir] = (tmp_path / "projects" / "Vision" / "artifacts").iterdir()
    assert (artifact_dir / "files" / "payload.json").read_bytes() == b"{}"
    assert json.loads((artifact_dir / "artifact.json").read_text())["name"] == "eval-set"
