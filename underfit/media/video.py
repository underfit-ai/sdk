"""Video media type for Underfit logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from underfit.media._helpers import guess_mime_type, validate_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


@dataclass(frozen=True, init=False)
class Video:
    """Represent a video payload for logging.

    The video source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.

    Examples:
        >>> from underfit import Video
        >>> Video("./predictions/rollout.mp4", caption="rollout")
    """

    _type: Literal["video"] = field(init=False, default="video")
    data: bytes
    caption: str | None
    fps: int
    file_type: str | None
    mime_type: str

    def __init__(
        self,
        data_or_path: PathLikeOrBytes,
        *,
        caption: str | None = None,
        fps: int = 4,
        file_type: str | None = None,
    ) -> None:
        """Initialize a video payload.

        Args:
            data_or_path: Video source as a path or raw bytes.
            caption: Optional display caption.
            fps: Playback frame rate in frames per second.
            file_type: Optional file type hint like ``"mp4"`` or ``"webm"``.

        Raises:
            TypeError: If ``data_or_path`` is not a supported input type.
            ValueError: If ``fps`` is not positive.
        """
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")

        inferred_file_type = file_type
        mime_type = "application/octet-stream"

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            mime_type = validate_path(path, r"video/.+", "a video")
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
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "file_type", inferred_file_type)
        object.__setattr__(self, "mime_type", mime_type)
