"""Backend interface for Underfit run storage and transport."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Define the backend contract used by ``underfit.Run``."""

    @property
    @abstractmethod
    def run_name(self) -> str:
        """Return the normalized backend run name."""

    @abstractmethod
    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""

    @abstractmethod
    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""

    @abstractmethod
    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""

    @abstractmethod
    def log_artifact(self, artifact: Any) -> None:
        """Store an artifact for a run."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize a run and flush backend state."""
