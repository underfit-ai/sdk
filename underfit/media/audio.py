"""Audio media type for Underfit logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from underfit.media._helpers import guess_mime_type, validate_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


@dataclass(frozen=True, init=False)
class Audio:
    """Represent an audio payload for logging.

    The audio source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.
    """

    _type: Literal["audio"] = field(init=False, default="audio")
    data: bytes
    caption: str | None
    sample_rate: int | None
    file_type: str | None
    mime_type: str

    def __init__(
        self,
        data_or_path: PathLikeOrBytes,
        *,
        caption: str | None = None,
        sample_rate: int | None = None,
        file_type: str | None = None,
    ) -> None:
        """Initialize an audio payload.

        Args:
            data_or_path: Audio source as a path or raw bytes.
            caption: Optional display caption.
            sample_rate: Optional sample rate in Hertz.
            file_type: Optional file type hint like ``"wav"`` or ``"mp3"``.

        Raises:
            TypeError: If ``data_or_path`` is not a supported input type.
            ValueError: If ``sample_rate`` is invalid or bytes are missing a file type.
        """
        if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError("sample_rate must be a positive integer")

        inferred_file_type = file_type
        mime_type = "application/octet-stream"

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            mime_type = validate_path(path, r"audio/.+", "an audio")
            data = path.read_bytes()
            inferred_file_type = file_type or mime_type.split("/")[-1]
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
            if file_type is None:
                raise ValueError("file_type is required when providing raw bytes")
            mime_type = guess_mime_type(file_type) or "application/octet-stream"
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "caption", caption)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "file_type", inferred_file_type)
        object.__setattr__(self, "mime_type", mime_type)
