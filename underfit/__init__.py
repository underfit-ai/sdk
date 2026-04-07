"""Underfit Python SDK."""

from __future__ import annotations

import os
import random
import re
import socket
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact
from underfit.backends import Backend, LocalBackend, RemoteBackend
from underfit.lib.terminal import capture
from underfit.media import Audio, Html, Image, Video
from underfit.run import PathLike, PathOrBytes, Run

__all__ = [
    "Artifact", "Audio", "Html", "Image", "Run", "Video",
    "finish", "init", "log", "log_git", "log_model", "run",
]

run: Run | None = None
_capture_context: AbstractContextManager[None] | None = None


def _require_run() -> Run:
    if run is None:
        raise RuntimeError("underfit.init must be called before using the active run")
    return run


def _default_worker_label() -> str:
    hostname = socket.gethostname().strip().lower()
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", hostname).strip("-._") if hostname else ""
    return sanitized or "worker"


def _generate_run_name() -> str:
    wordlists = Path(__file__).parent / "wordlists"
    adjectives = (wordlists / "adjectives.txt").read_text().splitlines()
    nouns = (wordlists / "nouns.txt").read_text().splitlines()
    return f"{random.choice(adjectives)}-{random.choice(nouns)}"  # noqa: S311


@contextmanager
def _capture_output(backend: Backend) -> Iterator[None]:
    pending = {"stdout": "", "stderr": ""}

    def write(stream: str, data: str) -> None:
        pending[stream] += data
        lines = pending[stream].split("\n")
        pending[stream] = lines.pop()
        if lines:
            backend.log_lines(lines)

    with capture(write):
        yield
    if tail := [line for line in pending.values() if line]:
        backend.log_lines(tail)


def init(
    project: str,
    name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    remote_url: str | None = None,
    launch_id: str | None = None,
    worker_label: str | None = None,
) -> Run:
    """Initialize a new Underfit run.

    Args:
        project: Project name.
        name: Optional run name. When omitted a random name is generated.
        config: Run configuration dictionary.
        log_dir: Directory for local run logs. Defaults to ``./underfit``.
        remote_url: Base URL for the self-hosted Underfit backend API.
        launch_id: Shared launch identifier for multi-worker runs. When
            omitted a unique ID is generated for a single-worker run.
        worker_label: Label identifying this worker within the run. Defaults
            to a value derived from the hostname when omitted.
    """
    global run, _capture_context  # noqa: PLW0603
    if run is not None:
        return run

    resolved_config = dict(config or {})
    resolved_name = name.strip() if name else _generate_run_name()
    resolved_worker_label = worker_label or _default_worker_label()
    if remote_url is None:
        root_dir = log_dir or Path(os.environ.get("UNDERFIT_LOG_DIR", "./underfit"))
        backend = LocalBackend(
            project_name=project,
            run_name=resolved_name,
            run_config=resolved_config,
            root_dir=root_dir.resolve(),
        )
    else:
        if not (api_key := os.environ.get("UNDERFIT_API_KEY")):
            raise RuntimeError("UNDERFIT_API_KEY is required when initializing with a remote URL")
        backend = RemoteBackend(
            api_url=remote_url,
            api_key=api_key,
            project=project,
            run_name=resolved_name,
            launch_id=launch_id or uuid4().hex,
            run_config=resolved_config,
            worker_label=resolved_worker_label,
        )

    run = Run(project=project, name=backend.run_name, backend=backend, config=resolved_config)
    _capture_context = _capture_output(backend)
    _capture_context.__enter__()
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
    global run, _capture_context  # noqa: PLW0603
    if run is None:
        return

    run.finish()
    if _capture_context is not None:
        _capture_context.__exit__(None, None, None)
        _capture_context = None
    run = None
