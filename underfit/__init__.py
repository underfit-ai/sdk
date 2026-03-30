"""Underfit Python SDK."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from underfit.artifact import Artifact
from underfit.backends.api import APIBackend, resolve_account_handle
from underfit.backends.base import Backend
from underfit.backends.local import LocalBackend
from underfit.media import Audio, Html, Image, Video
from underfit.run import PathOrBytes, Run

__all__ = [
    "Artifact",
    "Audio",
    "Html",
    "Image",
    "Run",
    "Video",
    "config",
    "finish",
    "init",
    "log",
    "log_model",
    "login",
    "run",
]
_ = (sys, urllib.request)

_config: dict[str, Any] = {}
config: dict[str, Any] = _config
run: Run | None = None
_api_key: str | None = None


def init(
    *,
    project: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    reinit: bool = False,
    url: str | None = None,
    offline: bool = False,
) -> Run:
    """Initialize a new Underfit run.

    Args:
        project: Project name (similar to wandb project).
        name: Optional run name.
        config: Run configuration dictionary.
        reinit: If True, start a new run even if one is active.
        url: Base URL for the self-hosted Underfit backend API.
        offline: If True, use a local filesystem backend instead of the API backend.
    """

    global run  # noqa: PLW0603

    if run is not None and not reinit:
        return run

    resolved_config = dict(config or {})
    backend: Backend | None = None
    api_url = url or os.environ.get("UNDERFIT_URL")
    mode = os.environ.get("UNDERFIT_MODE", "").strip().lower()
    use_offline_backend = offline or mode == "offline"

    if use_offline_backend:
        offline_root = os.environ.get("UNDERFIT_OFFLINE_DIR")
        backend = LocalBackend(
            project_name=project or "default",
            run_name=name,
            run_config=resolved_config,
            root_dir=None if offline_root is None else Path(offline_root).resolve(),
        )
        name = backend.run_name
    elif api_url and project:
        api_key = _api_key or os.environ.get("UNDERFIT_API_KEY")
        if not api_key:
            raise RuntimeError("UNDERFIT_API_KEY is required when initializing with a backend URL")
        account_handle = os.environ.get("UNDERFIT_HANDLE")
        if not account_handle:
            account_handle = resolve_account_handle(api_url, api_key)
        backend = APIBackend(
            api_url=api_url,
            api_key=api_key,
            account_handle=account_handle,
            project_name=project,
            run_name=name,
            run_config=resolved_config,
        )
        name = backend.run_name

    run = Run(project=project, name=name, config=resolved_config, _backend=backend)
    _config.clear()
    _config.update(resolved_config)
    return run


def log(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to the current run."""

    if run is None:
        raise RuntimeError("underfit.log called before underfit.init")

    run.log(data, step=step)


def log_model(checkpoint: PathOrBytes, *, name: str | None = None) -> Artifact:
    """Log a model checkpoint to the current run."""

    if run is None:
        raise RuntimeError("underfit.log_model called before underfit.init")

    return run.log_model(checkpoint, name=name)


def finish() -> None:
    """Finish the current run."""

    global run  # noqa: PLW0603

    if run is None:
        return

    run.finish()
    run = None


def login(*, api_key: str | None = None) -> None:
    """Authenticate with the Underfit backend.

    Args:
        api_key: Underfit API token used as a Bearer token for API requests.
    """

    global _api_key  # noqa: PLW0603

    _api_key = api_key
