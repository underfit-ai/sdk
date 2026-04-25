"""End-to-end test: SDK RemoteClient → API → API read endpoints."""

from __future__ import annotations

from typing import Any

import underfit
from tests.e2e.conftest import flatten_scalar_series
from underfit import Artifact, Html


def test_remote_client_round_trip(remote_env: dict[str, Any]) -> None:
    """Log scalars, lines, media, and an artifact via the SDK and verify them through the API."""
    client = remote_env["client"]
    handle = remote_env["handle"]
    project = remote_env["project"]
    auth = {"Authorization": f"Bearer {remote_env['api_key']}"}
    local_file = remote_env["local_file"]
    reference_file = remote_env["reference_file"]

    run = underfit.init(
        project=project, name="alpha", remote_url="http://testserver",
        config={"lr": 0.01}, worker_label="w0",
    )
    underfit.log({"loss": 0.5, "accuracy": 0.9}, step=1)
    underfit.log({"loss": 0.4, "accuracy": 0.92}, step=2)
    run.client.log_lines(["hello", "world"])
    underfit.log({"sample": Html("<h1>ok</h1>", caption="hi")}, step=1)
    artifact = Artifact("bundle", "dataset", metadata={"format": "json", "tag": "best"}, step=7)
    artifact.add_file(local_file)
    artifact.add_bytes(b'{"x": 1}', name="inline.json")
    artifact.add_url(reference_file.as_uri())
    run.log_artifact(artifact).result()

    underfit.finish()

    base = f"/api/v1/accounts/{handle}/projects/{project}/runs/alpha"

    run_resp = client.get(base, headers=auth)
    assert run_resp.status_code == 200, run_resp.text
    assert run_resp.json()["name"] == "alpha"
    assert run_resp.json()["terminalState"] == "finished"

    scalars_resp = client.get(f"{base}/scalars", headers=auth)
    assert scalars_resp.status_code == 200, scalars_resp.text
    points = flatten_scalar_series(scalars_resp.json())
    assert points[(1, "loss")] == 0.5
    assert points[(2, "loss")] == 0.4
    assert points[(1, "accuracy")] == 0.9
    assert points[(2, "accuracy")] == 0.92

    logs_resp = client.get(f"{base}/logs/w0", headers=auth)
    assert logs_resp.status_code == 200, logs_resp.text
    log_text = "\n".join(entry["content"] for entry in logs_resp.json()["entries"])
    assert "hello" in log_text and "world" in log_text

    media_resp = client.get(f"{base}/media", headers=auth)
    assert media_resp.status_code == 200, media_resp.text
    media = media_resp.json()
    assert len(media) == 1
    assert media[0]["key"] == "sample"
    assert media[0]["step"] == 1
    assert media[0]["type"] == "html"
    media_file = client.get(f"{base}/media/{media[0]['id']}/file", headers=auth)
    assert media_file.status_code == 200
    assert media_file.content == b"<h1>ok</h1>"

    artifacts_resp = client.get(f"/api/v1/accounts/{handle}/projects/{project}/artifacts", headers=auth)
    assert artifacts_resp.status_code == 200, artifacts_resp.text
    artifacts = artifacts_resp.json()
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "bundle"
    assert artifacts[0]["type"] == "dataset"
    assert artifacts[0]["step"] == 7
    assert artifacts[0]["metadata"] == {"format": "json", "tag": "best"}
    assert artifacts[0]["finalizedAt"] is not None

    artifact_id = artifacts[0]["id"]
    local_file_resp = client.get(f"/api/v1/artifacts/{artifact_id}/files/payload.json", headers=auth)
    assert local_file_resp.status_code == 200
    assert local_file_resp.content == b'{"y": 2}'

    inline_file_resp = client.get(f"/api/v1/artifacts/{artifact_id}/files/inline.json", headers=auth)
    assert inline_file_resp.status_code == 200
    assert inline_file_resp.content == b'{"x": 1}'


def test_remote_project_artifact_round_trip(remote_env: dict[str, Any]) -> None:
    """Project-level artifacts upload via Project.log_artifact and appear with no run attached."""
    client = remote_env["client"]
    handle, project_name = remote_env["handle"], remote_env["project"]
    auth = {"Authorization": f"Bearer {remote_env['api_key']}"}

    proj = underfit.project(f"{handle}/{project_name}", remote_url="http://testserver")
    artifact = Artifact("eval-set", "dataset")
    artifact.add_bytes(b'{"x": 1}', name="payload.json")
    proj.log_artifact(artifact).result()

    artifacts = client.get(f"/api/v1/accounts/{handle}/projects/{project_name}/artifacts", headers=auth).json()
    assert [a["name"] for a in artifacts] == ["eval-set"]
    assert artifacts[0]["runId"] is None
    file_resp = client.get(f"/api/v1/artifacts/{artifacts[0]['id']}/files/payload.json", headers=auth)
    assert file_resp.content == b'{"x": 1}'
