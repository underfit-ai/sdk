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
    def log_lines(self, worker_id: str, lines: list[str]) -> None:
        """Append console log lines for a run."""

    @abstractmethod
    def log_media(self, key: str, step: int | None, payloads: list[dict[str, Any]]) -> None:
        """Append media files for a run under a shared key and step."""

    @abstractmethod
    def upload_artifact_entry(self, artifact_name: str, entry: dict[str, Any]) -> None:
        """Store an artifact entry for a run."""

    @abstractmethod
    def read_scalars(self) -> list[dict[str, Any]]:
        """Return scalar records that were stored for a run."""

    @abstractmethod
    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        """Return log records, optionally filtered by worker id."""

    @abstractmethod
    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact entries, optionally filtered by artifact name."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize a run and flush backend state."""
