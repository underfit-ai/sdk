"""Tests for public media classes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from underfit.media import Audio, Html, Image, Video


@pytest.mark.parametrize(("cls", "name", "data"), [
    (Audio, "sample.wav", b"audio"),
    (Image, "sample.png", b"img"),
    (Video, "sample.mp4", b"video"),
    (Html, "report.html", b"<h1>ok</h1>"),
])
def test_media_path_inputs_embed_data(tmp_path: Path, cls: type[object], name: str, data: bytes) -> None:
    """Embed file contents for supported path-based media inputs."""
    path = tmp_path / name
    path.write_bytes(data)
    assert cls(path).data == data


@pytest.mark.parametrize(("factory", "message"), [
    (lambda: Audio(b"audio"), "file_type is required"),
    (lambda: Audio(b"audio", sample_rate=0, file_type="wav"), "sample_rate must be a positive integer"),
    (lambda: Image(b"img", width=0), "width must be a positive integer"),
    (lambda: Image(b"img", height=0), "height must be a positive integer"),
    (lambda: Video(b"video", fps=0), "fps must be a positive integer"),
    (lambda: Html(b"\xff"), "HTML bytes must be UTF-8 encoded"),
])
def test_media_rejects_invalid_bytes_inputs(factory: Callable[[], object], message: str) -> None:
    """Reject invalid bytes inputs and numeric metadata."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("cls", [Audio, Image, Video])
def test_media_rejects_wrong_file_types(tmp_path: Path, cls: type[object]) -> None:
    """Reject non-media files for path-based inputs."""
    path = tmp_path / "sample.txt"
    path.write_text("not media")
    with pytest.raises(ValueError, match="path must be"):
        cls(path)


def test_html_rejects_non_html_paths(tmp_path: Path) -> None:
    """Reject non-HTML files for path-based HTML inputs."""
    path = tmp_path / "report.txt"
    path.write_text("not html")
    with pytest.raises(ValueError, match="path must be an HTML file"):
        Html(path)
