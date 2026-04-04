"""HTML media type for Underfit logging."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Union

HtmlLike = Union[str, Path, bytes, bytearray, memoryview]


class Html:
    """Represent an HTML payload for logging.

    The HTML source can be inline markup, a filesystem path, or UTF-8 encoded
    bytes. This class stores metadata used by the uploader and keeps a stable,
    inspectable payload shape for SDK consumers.

    Examples:
        >>> from underfit import Html
        >>> Html("<h1>Report</h1>", caption="summary")
    """

    path: Path | None
    html: str | None
    caption: str | None
    inject: bool

    def __init__(self, data_or_path: HtmlLike, *, caption: str | None = None, inject: bool = True) -> None:
        """Initialize an HTML payload.

        Args:
            data_or_path: HTML markup, path, or UTF-8 encoded bytes.
            caption: Optional display caption.
            inject: Enable script and style injection for supported display contexts.

        Raises:
            TypeError: If ``data_or_path`` is not a supported input type.
            ValueError: If ``data_or_path`` bytes cannot be decoded as UTF-8.
        """
        path: Path | None = None
        html: str | None = None

        if isinstance(data_or_path, Path):
            path = data_or_path
        elif isinstance(data_or_path, str):
            html = data_or_path
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            try:
                html = bytes(data_or_path).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("HTML bytes must be UTF-8 encoded") from exc
        else:
            raise TypeError("data_or_path must be an HTML string, Path, or bytes-like object")

        self.path = path
        self.html = html
        self.caption = caption
        self.inject = inject

    def to_payload(self) -> dict[str, Any]:
        """Return a transport-ready dictionary for this HTML payload.

        Returns:
            Dictionary with stable keys used by future uploader integrations.
        """
        payload: dict[str, Any] = {"_type": "html", "caption": self.caption, "inject": self.inject}

        if self.path is not None:
            payload["path"] = str(self.path)
        elif self.html is not None:
            payload["html"] = self.html

        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"path must point to a file: {path}")
        mime, _ = mimetypes.guess_type(path.as_posix())
        if mime is None or mime not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"path must be an HTML file: {path}")
