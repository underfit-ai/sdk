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


def test_local_client_reads_back_runs_and_artifacts(tmp_path: Path) -> None:
    """Round-trip writes and reads through the local client: runs, run artifacts, project artifacts."""
    writer = LocalClient(project="Vision", root_dir=tmp_path)
    writer.launch_run(run_name="alpha", run_config={"lr": 0.01}, worker_label="w0")
    writer.log_scalars({"loss": 0.4}, step=1)
    run_artifact = Artifact("ckpt", "model")
    run_artifact.add_bytes(b"weights", name="model.bin")
    writer.log_artifact(run_artifact).result()
    project_artifact = Artifact("eval-set", "dataset")
    project_artifact.add_bytes(b'{"x": 1}', name="payload.json")
    writer.log_project_artifact(writer.project, project_artifact).result()
    writer.finish()

    reader = LocalClient(project="Vision", root_dir=tmp_path)
    [run] = reader.project.list_runs()
    assert run.name == "alpha"
    assert run.summary == {"loss": 0.4}
    assert run.terminal_state == "finished"
    assert reader.project.get_run("alpha").id == run.id

    [run_ref] = run.list_artifacts()
    assert run_ref.name == "ckpt" and run_ref.files == ["model.bin"]
    assert run_ref.read("model.bin") == b"weights"

    appended = Artifact("post-hoc", "report")
    appended.add_bytes(b"summary", name="summary.txt")
    run.log_artifact(appended).result()
    assert sorted(a.name for a in run.list_artifacts()) == ["ckpt", "post-hoc"]

    [project_ref] = reader.project.list_artifacts()
    project_ref.download(tmp_path / "out")
    assert (tmp_path / "out" / "payload.json").read_bytes() == b'{"x": 1}'
