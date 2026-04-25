"""Client implementations for Underfit."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from underfit.artifact import Artifact
from underfit.clients.local import LocalClient
from underfit.clients.remote import RemoteClient
from underfit.media import Media
from underfit.project import Project

if TYPE_CHECKING:
    from underfit.run import Run

TerminalState = Literal["finished", "failed", "cancelled"]


@runtime_checkable
class Client(Protocol):
    """Define the storage client contract used by ``underfit.Run``."""

    project: Project
    run_name: str

    def log_scalars(self, values: dict[str, float], step: int | None) -> None:
        """Append scalar metric values for a run."""

    def log_lines(self, lines: list[str]) -> None:
        """Append console log lines for the run's worker."""

    def log_media(self, key: str, step: int | None, media: Sequence[Media]) -> None:
        """Append media files for a run under a shared key and step."""

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        """Store an artifact for the active run."""

    def log_project_artifact(self, project: Project, artifact: Artifact) -> Future[None]:
        """Store an artifact directly under a project."""

    def log_run_artifact(self, run: Run, artifact: Artifact) -> Future[None]:
        """Store an artifact under a previously created run."""

    def list_runs(self, project: Project) -> list[Run]:
        """Return the runs stored under a project."""

    def get_run(self, project: Project, name: str) -> Run:
        """Return a single run by name."""

    def list_artifacts(self, project: Project, run: Run | None = None) -> list[Artifact]:
        """Return project-scoped artifacts, or run-scoped artifacts when ``run`` is given."""

    def finish(self, terminal_state: TerminalState = "finished") -> None:
        """Finalize a run and flush client state."""


__all__ = ["Client", "LocalClient", "RemoteClient", "TerminalState"]
