"""Client implementations for Underfit."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from typing import Literal, Protocol, runtime_checkable

from underfit.artifact import Artifact
from underfit.clients.local import LocalClient
from underfit.clients.remote import RemoteClient
from underfit.media import Media

TerminalState = Literal["finished", "failed", "cancelled"]


@runtime_checkable
class Client(Protocol):
    """Define the storage client contract used by ``underfit.Run``."""

    run_name: str

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""

    def log_media(self, key: str, step: int | None, media: Sequence[Media]) -> None:
        """Append media files for a run under a shared key and step."""

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        """Store an artifact for a run."""

    def finish(self, terminal_state: TerminalState = "finished") -> None:
        """Finalize a run and flush client state."""


__all__ = ["Client", "LocalClient", "RemoteClient", "TerminalState"]
