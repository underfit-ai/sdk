"""Underfit Python SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from underfit.artifact import Artifact
from underfit.backends.api import APIBackend
from underfit.backends.local import LocalBackend
from underfit.media import Audio, Html, Image, Video
from underfit.run import PathLike, PathOrBytes, Run

__all__ = [
    "Artifact", "Audio", "Html", "Image", "Run", "Video",
    "finish", "init", "log", "log_git", "log_model", "run",
]

run: Run | None = None


def _require_run() -> Run:
    if run is None:
        raise RuntimeError("underfit.init must be called before using the active run")
    return run


def init(
    project: str,
    name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    remote_url: str | None = None,
) -> Run:
    """Initialize a new Underfit run.

    Args:
        project: Project name.
        name: Optional run name.
        config: Run configuration dictionary.
        log_dir: Directory for local run logs. Defaults to ``./underfit``.
        remote_url: Base URL for the self-hosted Underfit backend API.
    """
    global run  # noqa: PLW0603
    if run is not None:
        return run

    resolved_config = dict(config or {})
    if remote_url is None:
        root_dir = log_dir or Path(os.environ.get("UNDERFIT_LOG_DIR", "./underfit"))
        backend = LocalBackend(
            project_name=project,
            run_name=name,
            run_config=resolved_config,
            root_dir=root_dir.resolve(),
        )
    else:
        if not (api_key := os.environ.get("UNDERFIT_API_KEY")):
            raise RuntimeError("UNDERFIT_API_KEY is required when initializing with a remote URL")
        backend = APIBackend(
            api_url=remote_url,
            api_key=api_key,
            project_name=project,
            run_name=name,
            run_config=resolved_config,
        )

    run = Run(project=project, name=backend.run_name, backend=backend, config=resolved_config)
    return run


def log(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to the current run."""
    _require_run().log(data, step=step)


def log_git(repo_path: PathLike | None = None, *, name: str | None = None) -> Artifact:
    """Log the current git state to the active run."""
    return _require_run().log_git(repo_path, name=name)


def log_model(checkpoint: PathOrBytes, *, name: str | None = None) -> Artifact:
    """Log a model checkpoint to the current run."""
    return _require_run().log_model(checkpoint, name=name)


def finish() -> None:
    """Finish the current run."""
    global run  # noqa: PLW0603
    if run is None:
        return

    run.finish()
    run = None
