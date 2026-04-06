"""Backend implementations for Underfit."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from underfit.artifact import Artifact
from underfit.backends.local import LocalBackend
from underfit.media import Media


@runtime_checkable
class Backend(Protocol):
    """Define the backend contract used by ``underfit.Run``."""

    @property
    def run_name(self) -> str:
        """Return the normalized backend run name."""

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""

    def log_media(self, key: str, step: int | None, media: list[Media]) -> None:
        """Append media files for a run under a shared key and step."""

    def log_artifact(self, artifact: Artifact) -> None:
        """Store an artifact for a run."""

    def finish(self) -> None:
        """Finalize a run and flush backend state."""


__all__ = ["Backend", "LocalBackend"]
