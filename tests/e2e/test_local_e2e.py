"""End-to-end test: SDK LocalBackend → backfill → API read endpoints."""

from __future__ import annotations

import time
from typing import Any

import underfit
from tests.e2e.conftest import boot_backfill_client
from underfit import Artifact, Html


def _wait_for(client: Any, path: str, *, predicate: Any = None, timeout: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        response = client.get(path)
        if response.status_code == 200:
            last = response.json()
            if predicate is None or predicate(last):
                return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}, last={last!r}")


def test_local_backend_round_trip(local_env: dict[str, Any]) -> None:
    """Write via the LocalBackend, then verify via the API after backfill ingests the run."""
    log_dir = local_env["log_dir"]
    api_tmp_path = local_env["api_tmp_path"]

    run = underfit.init(
        project="vision", name="alpha", log_dir=log_dir,
        config={"lr": 0.01}, worker_label="w0",
    )
    run.log({"loss": 0.5}, step=1)
    run.log({"loss": 0.4}, step=2)
    run.backend.log_lines(["hello", "world"])
    run.log({"sample": Html("<h1>ok</h1>", caption="hi")}, step=1)

    artifact = Artifact("ds", "dataset", metadata={"format": "json"})
    artifact.add_bytes(b'{"x": 1}', name="data.json")
    run.log_artifact(artifact).result()
    underfit.finish()

    base = "/api/v1/accounts/local/projects/vision/runs/alpha"
    with boot_backfill_client(api_tmp_path, log_dir) as client:
        run_payload = _wait_for(client, base, predicate=lambda r: r.get("terminalState") == "finished")
        assert run_payload["name"] == "alpha"

        scalars = _wait_for(
            client, f"{base}/scalars?workerLabel=w0",
            predicate=lambda points: any(p["step"] == 2 for p in points),
        )
        loss_by_step = {p["step"]: p["values"]["loss"] for p in scalars}
        assert loss_by_step[1] == 0.5
        assert loss_by_step[2] == 0.4

        logs_payload = _wait_for(
            client, f"{base}/logs?workerLabel=w0",
            predicate=lambda payload: bool(payload.get("entries")),
        )
        log_text = "\n".join(entry["content"] for entry in logs_payload["entries"])
        assert "hello" in log_text and "world" in log_text

        media = _wait_for(client, f"{base}/media", predicate=bool)
        assert len(media) == 1
        assert media[0]["key"] == "sample"
        assert media[0]["step"] == 1
        assert media[0]["type"] == "html"
        media_file = client.get(f"{base}/media/{media[0]['id']}/file")
        assert media_file.status_code == 200
        assert media_file.content == b"<h1>ok</h1>"

        artifacts = _wait_for(client, "/api/v1/accounts/local/projects/vision/artifacts", predicate=bool)
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "ds"
        assert artifacts[0]["type"] == "dataset"
        assert artifacts[0]["finalizedAt"] is not None

        artifact_id = artifacts[0]["id"]
        file_resp = client.get(f"/api/v1/artifacts/{artifact_id}/files/data.json")
        assert file_resp.status_code == 200
        assert file_resp.content == b'{"x": 1}'
