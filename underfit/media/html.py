"""HTML media type for Underfit logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from underfit.media._helpers import validate_path

HtmlLike = Union[str, Path, bytes, bytearray, memoryview]


@dataclass(frozen=True, init=False)
class Html:
    """Represent an HTML payload for logging.

    The HTML source can be inline markup, a filesystem path, or UTF-8 encoded
    bytes. This class stores metadata used by the uploader and keeps a stable,
    inspectable payload shape for SDK consumers.

    Examples:
        >>> from underfit import Html
        >>> Html("<h1>Report</h1>", caption="summary")
    """

    _type: Literal["html"] = field(init=False, default="html")
    data: bytes
    caption: str | None
    inject: bool
    mime_type: str

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
        data: bytes
        mime_type = "text/html"

        if isinstance(data_or_path, Path):
            mime_type = validate_path(data_or_path, r"text/html|application/xhtml\+xml", "an HTML")
            data = data_or_path.read_bytes()
            self._validate_utf8(data, "HTML files must be UTF-8 encoded")
        elif isinstance(data_or_path, str):
            data = data_or_path.encode("utf-8")
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
            self._validate_utf8(data, "HTML bytes must be UTF-8 encoded")
        else:
            raise TypeError("data_or_path must be an HTML string, Path, or bytes-like object")

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "caption", caption)
        object.__setattr__(self, "inject", inject)
        object.__setattr__(self, "mime_type", mime_type)

    @staticmethod
    def _validate_utf8(data: bytes, message: str) -> None:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(message) from exc
