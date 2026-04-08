"""Video media type for Underfit logging."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Union

from underfit.media._helpers import infer_media_file_type, validate_media_path

PathLikeOrBytes = Union[str, Path, bytes, bytearray, memoryview]


class Video:
    """Represent a video payload for logging.

    The video source can be a filesystem path or raw bytes. This class stores
    metadata used by the uploader and keeps a stable, inspectable payload shape
    for SDK consumers.

    Examples:
        >>> from underfit import Video
        >>> Video("./predictions/rollout.mp4", caption="rollout")
    """

    path: Path | None
    data: bytes | None
    caption: str | None
    fps: int
    file_type: str | None

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

        path: Path | None = None
        data: bytes | None = None

        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            validate_media_path(path, "video", "a video")
        elif isinstance(data_or_path, (bytes, bytearray, memoryview)):
            data = bytes(data_or_path)
        else:
            raise TypeError("data_or_path must be a path string, Path, or bytes-like object")

        inferred_file_type = file_type or (infer_media_file_type(path, "video") if path else None)

        self.path = path
        self.data = data
        self.caption = caption
        self.fps = fps
        self.file_type = inferred_file_type

    def to_payload(self) -> dict[str, Any]:
        """Return a transport-ready dictionary for this video.

        Returns:
            Dictionary with stable keys used by future uploader integrations.
        """
        payload: dict[str, Any] = {
            "_type": "video",
            "caption": self.caption,
            "fps": self.fps,
            "file_type": self.file_type,
        }

        if self.path is not None:
            payload["path"] = str(self.path)
        elif self.data is not None:
            payload["data"] = base64.b64encode(self.data).decode("ascii")

        return {key: value for key, value in payload.items() if value is not None}
