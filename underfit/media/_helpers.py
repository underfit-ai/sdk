"""Shared helpers for Underfit media payloads."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path


def guess_mime_type(value: str) -> str | None:
    """Guess a MIME type from a file extension."""
    return mimetypes.guess_type(f"file.{value.lstrip('.')}")[0]


def validate_path(path: Path, mime_pattern: str, description: str) -> str:
    """Validate that a path points to a file with the expected media type."""
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"path must point to a file: {path}")
    mime = mimetypes.guess_type(path.as_posix())[0]
    if mime is None or re.fullmatch(mime_pattern, mime) is None:
        raise ValueError(f"path must be {description} file: {path}")
    return mime
