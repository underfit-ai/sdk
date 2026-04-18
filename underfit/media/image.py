"""Image media type for Underfit logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from underfit.media._helpers import guess_mime_type, validate_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


@dataclass(frozen=True, init=False)
class Image:
    """Represent an image payload for logging.

    The image source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.

    Examples:
        >>> from underfit import Image
        >>> Image("./predictions/sample.png", caption="model output")
    """

    _type: Literal["image"] = field(init=False, default="image")
    data: bytes
    caption: str | None
    file_type: str | None
    mime_type: str
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

        inferred_file_type = file_type
        mime_type = "application/octet-stream"

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            mime_type = validate_path(path, r"image/.+", "an image")
            data = path.read_bytes()
            inferred_file_type = file_type or mime_type.split("/")[-1]
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
            if file_type is not None:
                mime_type = guess_mime_type(file_type) or "application/octet-stream"
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "caption", caption)
        object.__setattr__(self, "file_type", inferred_file_type)
        object.__setattr__(self, "mime_type", mime_type)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
