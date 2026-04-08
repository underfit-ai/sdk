"""Audio media type for Underfit logging."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Union

from underfit.media._helpers import infer_media_file_type, validate_media_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


class Audio:
    """Represent an audio payload for logging.

    The audio source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.
    """

    path: Path | None
    data: bytes | None
    caption: str | None
    sample_rate: int | None
    file_type: str | None

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

        path: Path | None = None
        data: bytes | None = None

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            validate_media_path(path, "audio", "an audio")
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
            if file_type is None:
                raise ValueError("file_type is required when providing raw bytes")
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        inferred_file_type = file_type or (infer_media_file_type(path, "audio") if path else None)

        self.path = path
        self.data = data
        self.caption = caption
        self.sample_rate = sample_rate
        self.file_type = inferred_file_type

    def to_payload(self) -> dict[str, Any]:
        """Return a transport-ready dictionary for this audio payload."""
        payload: dict[str, Any] = {
            "_type": "audio",
            "caption": self.caption,
            "sample_rate": self.sample_rate,
            "file_type": self.file_type,
        }

        if self.path is not None:
            payload["path"] = str(self.path)
        elif self.data is not None:
            payload["data"] = base64.b64encode(self.data).decode("ascii")

        return {key: value for key, value in payload.items() if value is not None}
