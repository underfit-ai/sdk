"""Shared helpers for Underfit media payloads."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_media_path(path: Path, mime_prefix: str, description: str) -> None:
    """Validate that a path points to a file with the expected media type."""
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"path must point to a file: {path}")
    mime, _ = mimetypes.guess_type(path.as_posix())
    if mime is None or not mime.startswith(f"{mime_prefix}/"):
        raise ValueError(f"path must be {description} file: {path}")


def infer_media_file_type(path: Path, mime_prefix: str) -> str:
    """Infer a media file type from a path suffix or MIME type."""
    suffix = path.suffix.lstrip(".").lower()
    if suffix:
        return suffix
    mime, _ = mimetypes.guess_type(path.as_posix())
    if mime is None or not mime.startswith(f"{mime_prefix}/"):
        raise ValueError(f"unable to infer file type from path: {path}")
    return mime.split("/")[-1]


def extract_media_content(payload: Mapping[str, Any]) -> bytes:
    """Return the uploaded bytes stored in a media payload."""
    if (path := payload.get("path")) is not None:
        return Path(path).read_bytes()
    if (data := payload.get("data")) is not None:
        return base64.b64decode(data)
    if (html := payload.get("html")) is not None:
        return str(html).encode("utf-8")
    raise RuntimeError("media payload is missing content")
