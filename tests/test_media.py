"""Tests for public media classes."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import pytest

from underfit.media import Audio, Html, Image, Video


def test_media_payloads_include_expected_fields(tmp_path: Path) -> None:
    """Build stable payloads from valid media inputs."""
    html_path = tmp_path / "report.html"
    html_path.write_text("<h1>ok</h1>")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    assert Html(html_path, caption="report").to_payload() == {
        "_type": "html", "caption": "report", "inject": True, "path": str(html_path),
    }
    assert Image(b"img", file_type="png", width=2, height=1).to_payload() == {
        "_type": "image", "file_type": "png", "width": 2, "height": 1, "data": base64.b64encode(b"img").decode("ascii"),
    }
    assert Audio(audio_path, sample_rate=16000).to_payload() == {
        "_type": "audio", "sample_rate": 16000, "file_type": "wav", "path": str(audio_path),
    }
    assert Video(video_path, fps=12).to_payload() == {
        "_type": "video", "fps": 12, "file_type": "mp4", "path": str(video_path),
    }


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
