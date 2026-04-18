"""Media types for Underfit logging."""

from __future__ import annotations

from typing import Union

from underfit.media.audio import Audio
from underfit.media.html import Html
from underfit.media.image import Image
from underfit.media.video import Video

Media = Union[Audio, Html, Image, Video]


__all__ = [
    "Audio", "Html", "Image", "Media", "Video",
]
