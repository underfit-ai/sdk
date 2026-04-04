"""Tests for `underfit.artifact.Artifact`."""

from __future__ import annotations

import base64
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from underfit.artifact import Artifact
from underfit.media import Html


def test_artifact_builds_create_payload_uploads_and_manifest(tmp_path: Path) -> None:
    """Collect files first and emit the final manifest separately."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"loss": 0.1}\n')
    model_card = tmp_path / "model-card.txt"
    model_card.write_text("hello\n")
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

    assert artifact.create_payload() == {"name": "checkpoint", "type": "model", "metadata": {"tag": "best"}}
    assert artifact.upload_files() == [
        {"path": "reports/metrics.json", "source_path": str(metrics)},
        {"path": "files/a.txt", "source_path": str(bundle / "a.txt")},
        {"path": "files/nested/b.txt", "source_path": str(bundle / "nested" / "b.txt")},
        {"path": "weights.bin", "data": base64.b64encode(b"weights").decode("ascii")},
    ]
    assert artifact.finalize_manifest() == {
        "files": ["reports/metrics.json", "files/a.txt", "files/nested/b.txt", "weights.bin"],
        "references": [{
            "url": model_card.as_uri(),
            "size": 6,
            "sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
            "etag": None,
            "last_modified": artifact.finalize_manifest()["references"][0]["last_modified"],
        }],
    }


def test_artifact_add_media_uses_uploadable_file_content() -> None:
    """Convert media payloads into artifact file uploads."""
    artifact = Artifact("report", "report")

    path = artifact.add_media(Html("<h1>ok</h1>"))

    assert path == "media-1.html"
    assert artifact.upload_files() == [{
        "path": "media-1.html",
        "data": base64.b64encode(b"<h1>ok</h1>").decode("ascii"),
    }]
    assert artifact.finalize_manifest() == {"files": ["media-1.html"], "references": []}


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

    assert artifact.finalize_manifest()["references"] == [{
        "url": "https://example.com/model",
        "size": 7,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "etag": '"abc"',
        "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
    }]


def test_artifact_add_url_leaves_missing_headers_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep reference metadata empty when the HEAD request fails."""
    error = urllib.error.URLError("offline")

    def fake_urlopen(request: object, timeout: int) -> None:
        _ = (request, timeout)
        raise error

    monkeypatch.setattr("underfit.artifact.urllib.request.urlopen", fake_urlopen)

    artifact = Artifact("report", "report")
    artifact.add_url("https://example.com/model")

    assert artifact.finalize_manifest()["references"] == [{
        "url": "https://example.com/model",
        "size": None,
        "sha256": None,
        "etag": None,
        "last_modified": None,
    }]
