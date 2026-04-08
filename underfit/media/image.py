"""Image media type for Underfit logging."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Union

from underfit.media._helpers import infer_media_file_type, validate_media_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


class Image:
    """Represent an image payload for logging.

    The image source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.

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
        """Initialize an image payload.

        Args:
            data_or_path: Image source as a path or raw bytes.
            caption: Optional display caption.
            file_type: Optional file type hint like ``"png"`` or ``"jpeg"``.
            width: Optional image width in pixels.
            height: Optional image height in pixels.

        Raises:
            TypeError: If ``data_or_path`` is not a supported input type.
            ValueError: If width or height are not positive when provided.
        """
        if width is not None and (not isinstance(width, int) or width <= 0):
            raise ValueError("width must be a positive integer")
        if height is not None and (not isinstance(height, int) or height <= 0):
            raise ValueError("height must be a positive integer")

        path: Path | None = None
        data: bytes | None = None

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            validate_media_path(path, "image", "an image")
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        self.path = path
        self.data = data
        self.caption = caption
        self.file_type = file_type or (infer_media_file_type(path, "image") if path else None)
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
