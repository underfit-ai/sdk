"""Run model for Underfit."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Union

from underfit.artifact import Artifact
from underfit.backends import Backend, TerminalState
from underfit.media import Audio, Html, Image, Video

PathLike = Union[str, Path]
PathOrBytes = Union[str, Path, bytes, bytearray, memoryview]
PathFilter = Callable[[Path], bool]


class Run:
    """Represent an Underfit run."""

    def __init__(
        self,
        project: str,
        name: str,
        backend: Backend,
        config: dict[str, Any] | None = None,
        on_finish: Callable[[TerminalState], None] | None = None,
    ) -> None:
        """Initialize a run.

        Args:
            project: Project name.
            name: Run name.
            backend: Backend used to store run data.
            config: Run configuration dictionary.
            on_finish: Optional callback used when exiting a context.
        """
        self.project = project
        self.name = name
        self.backend = backend
        self.config = {} if config is None else dict(config)
        self._finished = False
        self._on_finish = on_finish
        self._scalar_lock = threading.Lock()
        self._pending_step: int | None = None
        self._pending_values: dict[str, float] = {}

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("run is already finished")

    def __enter__(self) -> Run:  # noqa: PYI034
        """Return the active run context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _: BaseException | None,
        __: TracebackType | None,
    ) -> None:
        """Finish the run when exiting a context."""
        terminal_state: TerminalState = "finished"
        if exc_type is not None:
            terminal_state = "cancelled" if issubclass(exc_type, KeyboardInterrupt) else "failed"
        if self._on_finish:
            self._on_finish(terminal_state)
            return
        self.finish(terminal_state)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Record metrics and media for the run.

        Scalars are sent to the scalar endpoint; media objects (Html, Image,
        Video, Audio) are uploaded via the media endpoint under the dict key.

        Args:
            data: Mapping of metric names to values or media objects.
            step: Optional global step.

        Raises:
            RuntimeError: If the run has already been finished.
            TypeError: If a logged value or key is unsupported.
        """
        self._require_active()
        scalar_values: dict[str, float] = {}
        media_batches: list[tuple[str, list[Any]]] = []
        items: list[tuple[str, Any]] = []

        def flatten(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    if not isinstance(child_key, str):
                        raise TypeError(f"Log keys must be strings: {prefix}")
                    flatten(f"{prefix}/{child_key}", child_value)
                return
            items.append((prefix, value))

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("Log keys must be strings")
            flatten(key, value)

        for key, value in items:
            if isinstance(value, (bool, int, float)):
                scalar_values[key] = float(value)
            elif isinstance(value, (Html, Image, Video, Audio)):
                media_batches.append((key, [value]))
            elif isinstance(value, (list, tuple)):
                if not value or not all(isinstance(v, (Html, Image, Video, Audio)) for v in value):
                    raise TypeError(f"Lists passed to underfit.Run.log must contain only media objects: {key}")
                if any(type(v) is not type(value[0]) for v in value[1:]):
                    raise TypeError(f"Lists passed to underfit.Run.log must contain only one media type: {key}")
                media_batches.append((key, list(value)))
            else:
                raise TypeError(f"Unsupported value for underfit.Run.log: {key}")

        if scalar_values:
            self._buffer_scalars(scalar_values, step)
        for key, media_list in media_batches:
            self.backend.log_media(key, step, media_list)

    def _buffer_scalars(self, values: dict[str, float], step: int | None) -> None:
        with self._scalar_lock:
            if step is None:
                self._flush_pending_locked()
                self.backend.log_scalars(values, None)
            elif self._pending_step is None or step == self._pending_step:
                self._pending_step = step
                self._pending_values.update(values)
            else:
                self._flush_pending_locked()
                self._pending_step = step
                self._pending_values = dict(values)

    def _flush_pending_locked(self) -> None:
        if self._pending_step is not None:
            self.backend.log_scalars(self._pending_values, self._pending_step)
            self._pending_step = None
            self._pending_values = {}

    def log_code(
        self,
        root_path: PathLike | None = None,
        *,
        name: str | None = None,
        include: PathFilter | None = None,
        exclude: PathFilter | None = None,
    ) -> Future[None]:
        """Upload source code under a root path as a zip artifact.

        Args:
            root_path: Root directory to scan. Defaults to the current working directory.
            name: Optional artifact name.
            include: Optional whitelist callable. It receives each resolved
                absolute file path and returns True to include it.
            exclude: Optional blacklist callable. It receives each resolved
                absolute file path and returns True to exclude it.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
            RuntimeError: If the run has already been finished.
            ValueError: If ``root_path`` is not a directory.
        """
        self._require_active()
        artifact = Artifact.from_code(root_path, name=name, include=include, exclude=exclude)
        return self.log_artifact(artifact)

    def log_git(self, repo_path: PathLike | None = None, *, name: str | None = None) -> Future[None]:
        """Upload the current git state as an artifact.

        Args:
            repo_path: Git repository path. Defaults to the current working directory.
            name: Optional artifact name.

        Raises:
            FileNotFoundError: If ``repo_path`` does not exist.
            RuntimeError: If git is not installed, the path is not in a git
                repository, or the run has already been finished.
            ValueError: If ``repo_path`` is not a directory.
        """
        self._require_active()
        artifact = Artifact.from_git(repo_path, name=name)
        return self.log_artifact(artifact)

    def log_model(self, checkpoint: PathOrBytes, *, name: str | None = None, step: int | None = None) -> Future[None]:
        """Upload a model checkpoint as an artifact.

        Args:
            checkpoint: Model checkpoint as a file path, directory path, or byte buffer.
            name: Optional artifact name.
            step: Optional global step for the artifact.

        Raises:
            FileNotFoundError: If a path checkpoint does not exist.
            RuntimeError: If the run has already been finished.
            TypeError: If ``checkpoint`` is not path-like or bytes-like.
            ValueError: If a path checkpoint is neither a file nor directory.
        """
        self._require_active()
        artifact = Artifact.from_model(checkpoint, name=name, step=step)
        return self.log_artifact(artifact)

    def log_artifact(self, artifact: Artifact) -> Future[None]:
        """Upload an artifact to the backend.

        Args:
            artifact: Artifact to upload.

        Raises:
            RuntimeError: If the run has already been finished.
            TypeError: If ``artifact`` is not an ``underfit.Artifact``.
        """
        self._require_active()
        if not isinstance(artifact, Artifact):
            raise TypeError("artifact must be an underfit.Artifact")
        return self.backend.log_artifact(artifact)

    def finish(self, terminal_state: TerminalState = "finished") -> None:
        """Finalize the run."""
        if self._finished:
            return
        with self._scalar_lock:
            self._flush_pending_locked()
        self.backend.finish(terminal_state)
        self._finished = True
