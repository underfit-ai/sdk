"""Project model for Underfit."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from underfit.artifact import Artifact
    from underfit.clients import Client
    from underfit.run import Run


@dataclass
class Project:
    """A project owned by an account."""

    handle: str
    name: str
    client: Client

    @property
    def identifier(self) -> str:
        """Return the canonical ``"handle/name"`` identifier."""
        return f"{self.handle}/{self.name}"

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        """Upload an artifact attached to this project rather than to a run."""
        return self.client.log_project_artifact(self, artifact)

    def list_runs(self) -> list[Run]:
        """Return the runs that belong to this project."""
        return self.client.list_runs(self)

    def get_run(self, name: str) -> Run:
        """Return a single run by name."""
        return self.client.get_run(self, name)

    def list_artifacts(self) -> list[Artifact]:
        """Return artifacts attached directly to this project (not to any run)."""
        return self.client.list_artifacts(self)
