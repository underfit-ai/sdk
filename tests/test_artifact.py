"""Tests for `underfit.artifact.Artifact`."""

from __future__ import annotations

import base64
import os
import subprocess
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from underfit.artifact import (
    Artifact,
    ArtifactDataUpload,
    ArtifactManifest,
    ArtifactPathUpload,
    ArtifactReference,
)
from underfit.media import Audio, Html, Image, Video


def test_artifact_collects_uploads_and_manifest(tmp_path: Path) -> None:
    """Collect artifact data before backend serialization."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"loss": 0.1}\n')
    model_card = tmp_path / "model-card.txt"
    model_card.write_text("hello\n")
    os.utime(model_card, (1445412480, 1445412480))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.txt").write_text("a")
    (bundle / "nested").mkdir()
    (bundle / "nested" / "b.txt").write_text("b")

    artifact = Artifact("checkpoint", "model", metadata={"tag": "best"})
    artifact.add_file(metrics, name="reports/metrics.json")
    artifact.add_dir(bundle, name="files")
    artifact.add_bytes(b"weights", name="weights.bin")
    artifact.add_url(model_card.as_uri())

    assert artifact.uploads() == [
        ArtifactPathUpload(path="reports/metrics.json", source_path=str(metrics)),
        ArtifactPathUpload(path="files/a.txt", source_path=str(bundle / "a.txt")),
        ArtifactPathUpload(path="files/nested/b.txt", source_path=str(bundle / "nested" / "b.txt")),
        ArtifactDataUpload(path="weights.bin", data=base64.b64encode(b"weights").decode("ascii")),
    ]
    assert artifact.manifest() == ArtifactManifest(
        files=["reports/metrics.json", "files/a.txt", "files/nested/b.txt", "weights.bin"],
        references=[ArtifactReference(
            url=model_card.as_uri(),
            size=6,
            sha256="5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
            last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        )],
    )


def test_artifact_add_media_uses_uploadable_file_content(tmp_path: Path) -> None:
    """Convert media payloads into artifact file uploads."""
    artifact = Artifact("report", "report")

    path = artifact.add_media(Html("<h1>ok</h1>"))

    assert path == "media-1.html"
    assert artifact.uploads() == [
        ArtifactDataUpload(path="media-1.html", data=base64.b64encode(b"<h1>ok</h1>").decode("ascii")),
    ]
    assert artifact.manifest() == ArtifactManifest(files=["media-1.html"], references=[])
    image_path = tmp_path / "sample.png"
    html_path = tmp_path / "sample.html"
    image_path.write_bytes(b"img")
    html_path.write_text("<h1>ok</h1>")
    assert Image(image_path).to_payload()["file_type"] == "png"
    assert Html(html_path).to_payload()["path"] == str(html_path)
    assert Audio(b"audio", file_type="wav").to_payload()["file_type"] == "wav"
    assert Video(b"video", file_type="mp4").to_payload()["file_type"] == "mp4"
    with pytest.raises(ValueError, match="file_type is required"):
        Audio(b"audio")


def test_artifact_rejects_invalid_and_conflicting_paths() -> None:
    """Reject artifact paths that the API would not accept."""
    artifact = Artifact("report", "report")
    artifact.add_bytes(b"a", name="dir/file.txt")

    with pytest.raises(ValueError, match="valid relative path"):
        artifact.add_bytes(b"b", name="/bad.txt")

    with pytest.raises(ValueError, match="already exists"):
        artifact.add_bytes(b"c", name="dir")


def test_artifact_rejects_file_urls_with_authorities() -> None:
    """Reject file URLs that include an authority component."""
    artifact = Artifact("report", "report")

    with pytest.raises(ValueError, match="must not include an authority"):
        artifact.add_url("file://localhost/tmp/report.txt")


def test_artifact_add_url_uses_head_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate reference metadata from an HTTP HEAD response."""
    headers = Message()
    headers["Content-Length"] = "7"
    headers["ETag"] = '"abc"'
    headers["Last-Modified"] = "Wed, 21 Oct 2015 07:28:00 GMT"
    headers["X-Checksum-Sha256"] = "ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0="

    class _Response:
        def __init__(self, response_headers: Message) -> None:
            self.headers = response_headers

        def __enter__(self) -> _Response:  # noqa: PYI034
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)

    def fake_urlopen(request: object, timeout: int) -> _Response:
        assert getattr(request, "method", None) == "HEAD"
        assert timeout == 10
        return _Response(headers)

    monkeypatch.setattr("underfit.artifact.urllib.request.urlopen", fake_urlopen)

    artifact = Artifact("report", "report")
    artifact.add_url("https://example.com/model")

    assert artifact.manifest().references == [ArtifactReference(
        url="https://example.com/model",
        size=7,
        sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        etag='"abc"',
        last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )]


def test_artifact_add_url_leaves_missing_headers_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep reference metadata empty when the HEAD request fails."""
    error = urllib.error.URLError("offline")

    def fake_urlopen(request: object, timeout: int) -> None:
        _ = (request, timeout)
        raise error

    monkeypatch.setattr("underfit.artifact.urllib.request.urlopen", fake_urlopen)

    artifact = Artifact("report", "report")
    artifact.add_url("https://example.com/model")

    assert artifact.manifest().references == [ArtifactReference(url="https://example.com/model")]


def test_artifact_from_model_directory_and_git_repo(tmp_path: Path) -> None:
    """Build artifacts from a directory checkpoint and a real git repo."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    assert Artifact.from_model(checkpoint, step=7).uploads() == [
        ArtifactPathUpload(path="checkpoint/weights.bin", source_path=str(checkpoint / "weights.bin")),
    ]

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)  # noqa: S607
    (repo / "tracked.txt").write_text("v1\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)  # noqa: S607
    (repo / "tracked.txt").write_text("v1\nv2\n")
    (repo / "new.txt").write_text("new\n")

    artifact = Artifact.from_git(repo)
    upload = artifact.uploads()[0]
    assert artifact.metadata["commit"] is None and artifact.metadata["is_dirty"] is True
    assert artifact.metadata["untracked_files"] == ["new.txt"]
    assert isinstance(upload, ArtifactDataUpload) and upload.path == "working-tree.patch"
    assert b"tracked.txt" in base64.b64decode(upload.data)
