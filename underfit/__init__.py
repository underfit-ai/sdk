"""Underfit Python SDK."""

from __future__ import annotations

import os
import random
import re
import socket
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from underfit.artifact import Artifact
from underfit.clients import Client, LocalClient, RemoteClient, TerminalState
from underfit.lib.terminal import capture
from underfit.media import Audio, Html, Image, Video
from underfit.project import Project
from underfit.run import PathFilter, PathLike, PathOrBytes, RunSession

__all__ = [
    "Artifact", "Audio", "Html", "Image", "Project", "RunSession", "Video",
    "finish", "init", "log", "log_git", "log_model", "project", "session",
]

session: RunSession | None = None
_capture_context: AbstractContextManager[None] | None = None


def _require_session() -> RunSession:
    if session is None:
        raise RuntimeError("underfit.init must be called before using the active run")
    return session


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
def _capture_output(client: Client) -> Iterator[None]:
    pending = {"stdout": "", "stderr": ""}

    def write(stream: str, data: str) -> None:
        pending[stream] += data
        lines = pending[stream].split("\n")
        pending[stream] = lines.pop()
        if lines:
            client.log_lines(lines)

    with capture(write):
        yield
    if tail := [line for line in pending.values() if line]:
        client.log_lines(tail)


def init(
    project: str,
    name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    remote_url: str | None = None,
    launch_id: str | None = None,
    worker_label: str | None = None,
) -> RunSession:
    """Initialize a new Underfit run session.

    Args:
        project: Project identifier as either ``"<account-handle>/<project-name>"`` or a bare
            ``"<project-name>"``. Bare names resolve to projects owned by the authenticated user.
        name: Optional run name. When omitted a random name is generated.
        config: Run configuration dictionary.
        log_dir: Directory for local run logs. Defaults to ``./underfit``.
        remote_url: Base URL for the self-hosted Underfit API.
        launch_id: Shared launch identifier for multi-worker runs. When
            omitted a unique ID is generated for a single-worker run.
        worker_label: Label identifying this worker within the run. Defaults
            to a value derived from the hostname when omitted.
    """
    global session, _capture_context  # noqa: PLW0603
    if session is not None:
        return session

    resolved_config = dict(config or {})
    resolved_name = name.strip() if name else _generate_run_name()
    resolved_worker_label = worker_label or _default_worker_label()
    client: Client
    if remote_url is None:
        root_dir = log_dir or Path(os.environ.get("UNDERFIT_LOG_DIR", "./underfit"))
        local = LocalClient(project=project, root_dir=root_dir.resolve())
        local.launch_run(run_name=resolved_name, run_config=resolved_config, worker_label=resolved_worker_label)
        client = local
    else:
        if not (api_key := os.environ.get("UNDERFIT_API_KEY")):
            raise RuntimeError("UNDERFIT_API_KEY is required when initializing with a remote URL")
        remote = RemoteClient(api_url=remote_url, api_key=api_key, project=project)
        remote.launch_run(
            run_name=resolved_name, launch_id=launch_id or uuid4().hex,
            run_config=resolved_config, worker_label=resolved_worker_label,
        )
        client = remote

    session = RunSession(project=client.project, name=client.run_name, config=resolved_config, on_finish=finish)
    _capture_context = _capture_output(client)
    _capture_context.__enter__()
    return session


def project(identifier: str, *, remote_url: str | None = None, log_dir: Path | None = None) -> Project:
    """Return a :class:`Project` for an existing project.

    Args:
        identifier: ``"<account-handle>/<project-name>"`` or a bare project name.
        remote_url: Base URL for the self-hosted Underfit API. When omitted, a local client is used.
        log_dir: Root directory for local project data. Defaults to ``./underfit``.
    """
    if remote_url is None:
        root_dir = log_dir or Path(os.environ.get("UNDERFIT_LOG_DIR", "./underfit"))
        return LocalClient(project=identifier, root_dir=root_dir.resolve()).project
    if not (api_key := os.environ.get("UNDERFIT_API_KEY")):
        raise RuntimeError("UNDERFIT_API_KEY is required when accessing a remote project")
    return RemoteClient(api_url=remote_url, api_key=api_key, project=identifier).project


def log(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to the current run."""
    _require_session().log(data, step=step)


def log_code(
    root_path: PathLike | None = None,
    *,
    name: str | None = None,
    include: PathFilter | None = None,
    exclude: PathFilter | None = None
) -> Future[None]:
    """Log the source code under a root path to the active run."""
    return _require_session().log_code(root_path, name=name, include=include, exclude=exclude)


def log_git(repo_path: PathLike | None = None, *, name: str | None = None) -> Future[None]:
    """Log the current git state to the active run."""
    return _require_session().log_git(repo_path, name=name)


def log_model(checkpoint: PathOrBytes, *, name: str | None = None, step: int | None = None) -> Future[None]:
    """Log a model checkpoint to the current run."""
    return _require_session().log_model(checkpoint, name=name, step=step)


def finish(terminal_state: TerminalState = "finished") -> None:
    """Finish the current run."""
    global session, _capture_context  # noqa: PLW0603
    if session is None:
        return
    if _capture_context is not None:
        _capture_context.__exit__(None, None, None)
        _capture_context = None
    session.finish(terminal_state)
    session = None
