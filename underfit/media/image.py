"""Image media type for Underfit logging."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Union

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


class Image:
    """Represent an image payload for logging.

    The image source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.

    Args:
        data_or_path: Image source as a path or raw bytes.
        caption: Optional display caption.
        file_type: Optional file type hint like ``"png"`` or ``"jpeg"``.
        width: Optional image width in pixels.
        height: Optional image height in pixels.

    Raises:
        TypeError: If ``data_or_path`` is not a supported input type.
        ValueError: If width or height are not positive when provided.

    Examples:
        >>> from underfit import Image
        >>> Image("./predictions/sample.png", caption="model output")
    """

    path: Path | None
    data: bytes | None
    caption: str | None
    file_type: str | None
    width: int | None
    height: int | None

    def __init__(
        self,
        data_or_path: PathLikeOrBytes,
        *,
        caption: str | None = None,
        file_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if width is not None and (not isinstance(width, int) or width <= 0):
            raise ValueError("width must be a positive integer")
        if height is not None and (not isinstance(height, int) or height <= 0):
            raise ValueError("height must be a positive integer")

        path: Path | None = None
        data: bytes | None = None

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            self._validate_path(path, "image")
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
            if file_type is None:
                raise ValueError("file_type is required when providing raw bytes")
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        self.path = path
        self.data = data
        self.caption = caption
        self.file_type = file_type or (self._infer_file_type(path, "image") if path else None)
        self.width = width
        self.height = height

    def to_payload(self) -> dict[str, Any]:
        """Return a transport-ready dictionary for this image.

        Returns:
            Dictionary with stable keys used by future uploader integrations.
        """

        payload: dict[str, Any] = {
            "_type": "image",
            "caption": self.caption,
            "file_type": self.file_type,
            "width": self.width,
            "height": self.height,
        }

        if self.path is not None:
            payload["path"] = str(self.path)
        elif self.data is not None:
            payload["data"] = base64.b64encode(self.data).decode("ascii")

        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _validate_path(path: Path, expected_prefix: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"path must point to a file: {path}")
        mime, _ = mimetypes.guess_type(path.as_posix())
        if mime is None or not mime.startswith(f"{expected_prefix}/"):
            raise ValueError(f"path must be an {expected_prefix} file: {path}")

    @staticmethod
    def _infer_file_type(path: Path, expected_prefix: str) -> str:
        suffix = path.suffix.lstrip(".").lower()
        if suffix:
            return suffix
        mime, _ = mimetypes.guess_type(path.as_posix())
        if mime is None or not mime.startswith(f"{expected_prefix}/"):
            raise ValueError(f"unable to infer file type from path: {path}")
        return mime.split("/")[-1]
