"""Media types for Underfit logging."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from underfit.media.audio import Audio
from underfit.media.html import Html
from underfit.media.image import Image
from underfit.media.video import Video


@runtime_checkable
class Media(Protocol):
    """Protocol for media objects that can be serialized for transport."""

    def to_payload(self) -> dict[str, Any]:
        """Return a transport-ready dictionary for this media object."""


__all__ = ["Audio", "Html", "Image", "Media", "Video"]
