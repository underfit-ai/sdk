"""End-to-end test: SDK LocalClient -> backfill -> API read endpoints."""

from __future__ import annotations

import time
from typing import Any

import underfit
from tests.e2e.conftest import boot_backfill_client, flatten_scalar_series
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


def test_local_client_round_trip(local_env: dict[str, Any]) -> None:
    """Write via the LocalClient, then verify via the API after backfill ingests the run."""
    log_dir = local_env["log_dir"]
    api_tmp_path = local_env["api_tmp_path"]

    run = underfit.init(
        project="vision", name="alpha", log_dir=log_dir,
        config={"lr": 0.01}, worker_label="w0",
    )
    underfit.log({"loss": 0.5}, step=1)
    underfit.log({"loss": 0.4}, step=2)
    run.client.log_lines(["hello", "world"])
    underfit.log({"sample": Html("<h1>ok</h1>", caption="hi")}, step=1)

    artifact = Artifact("ds", "dataset", metadata={"format": "json"})
    artifact.add_bytes(b'{"x": 1}', name="data.json")
    run.log_artifact(artifact).result()
    underfit.finish()

    base = "/api/v1/accounts/local/projects/vision/runs/alpha"
    with boot_backfill_client(api_tmp_path, log_dir) as client:
        run_payload = _wait_for(client, base, predicate=lambda r: r.get("terminalState") == "finished")
        assert run_payload["name"] == "alpha"

        scalars_payload = _wait_for(
            client, f"{base}/scalars",
            predicate=lambda payload: (2, "loss") in flatten_scalar_series(payload),
        )
        loss_by_step = {
            step: value for (step, key), value in flatten_scalar_series(scalars_payload).items() if key == "loss"
        }
        assert loss_by_step[1] == 0.5
        assert loss_by_step[2] == 0.4

        logs_payload = _wait_for(
            client, f"{base}/logs/w0",
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


def test_log_code_round_trip(local_env: dict[str, Any]) -> None:
    """Upload source code through the SDK and fetch the stored files through the API."""
    log_dir = local_env["log_dir"]
    api_tmp_path = local_env["api_tmp_path"]
    root = api_tmp_path / "src"
    root.mkdir()
    (root / "train.py").write_text("print('ok')\n")
    (root / "notes.txt").write_text("ignore\n")

    underfit.init(project="vision", name="alpha", log_dir=log_dir, worker_label="w0")
    underfit.log_code(root).result()
    underfit.finish()

    with boot_backfill_client(api_tmp_path, log_dir) as client:
        _wait_for(client, "/api/v1/accounts/local/projects/vision/runs/alpha", predicate=lambda r: r["name"] == "alpha")
        artifacts = _wait_for(client, "/api/v1/accounts/local/projects/vision/artifacts", predicate=bool)
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "source-code"
        assert artifacts[0]["type"] == "code"
        artifact = client.get(f"/api/v1/artifacts/{artifacts[0]['id']}")
        assert artifact.status_code == 200
        assert artifact.json()["manifest"]["files"] == ["train.py"]
        source = client.get(f"/api/v1/artifacts/{artifacts[0]['id']}/files/train.py")
        assert source.status_code == 200
        assert source.content == b"print('ok')\n"


def test_local_project_artifact_round_trip(local_env: dict[str, Any]) -> None:
    """Project-level artifacts written locally appear in the project after backfill."""
    log_dir = local_env["log_dir"]
    api_tmp_path = local_env["api_tmp_path"]

    proj = underfit.project("vision", log_dir=log_dir)
    artifact = Artifact("eval-set", "dataset")
    artifact.add_bytes(b'{"x": 1}', name="payload.json")
    proj.log_artifact(artifact).result()

    with boot_backfill_client(api_tmp_path, log_dir) as client:
        artifacts = _wait_for(client, "/api/v1/accounts/local/projects/vision/artifacts", predicate=bool)
        assert [a["name"] for a in artifacts] == ["eval-set"]
        assert artifacts[0]["runId"] is None
        file_resp = client.get(f"/api/v1/artifacts/{artifacts[0]['id']}/files/payload.json")
        assert file_resp.content == b'{"x": 1}'


