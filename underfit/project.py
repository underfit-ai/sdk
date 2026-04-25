"""Project model for Underfit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from underfit.clients import Client


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
