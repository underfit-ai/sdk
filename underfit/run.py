"""Run model for Underfit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

from underfit.artifact import Artifact
from underfit.backends.base import Backend
from underfit.media import Audio, Html, Image, Video

PathLike = Union[str, Path]
PathOrBytes = Union[str, Path, bytes, bytearray, memoryview]


class Run:
    """Represent an Underfit run.

    Args:
        project: Project name.
        name: Run name.
        backend: Backend used to store run data.
        config: Run configuration dictionary.
    """

    def __init__(self, project: str, name: str, backend: Backend, config: dict[str, Any] | None = None) -> None:
        self.project = project
        self.name = name
        self.backend = backend
        self.config = {} if config is None else dict(config)
        self._finished = False

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("run is already finished")

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Record metrics and media for the run.

        Scalars are sent to the scalar endpoint; media objects (Html, Image,
        Video, Audio) are uploaded via the media endpoint under the dict key.

        Args:
            data: Mapping of metric names to values or media objects.
            step: Optional global step.

        Raises:
            RuntimeError: If the run has already been finished.
            TypeError: If a logged value is boolean.
        """
        self._require_active()
        scalar_values: dict[str, float] = {}
        media_batches: list[tuple[str, list[Any]]] = []

        for key, value in data.items():
            if isinstance(value, bool):
                raise TypeError(f"Boolean values are not supported for underfit.Run.log: {key}")
            if isinstance(value, (int, float)):
                scalar_values[key] = float(value)
            elif isinstance(value, (Html, Image, Video, Audio)):
                media_batches.append((key, [value]))
            elif isinstance(value, (list, tuple)) and value and all(
                isinstance(v, (Html, Image, Video, Audio)) for v in value
            ):
                media_batches.append((key, list(value)))
            else:
                continue

        if scalar_values:
            self.backend.log_scalars(scalar_values, step)
        for key, media_list in media_batches:
            payloads = [media.to_payload() for media in media_list]
            self.backend.log_media(key, step, payloads)

    def log_code(
        self,
        root_path: PathLike | None = None,
        *,
        name: str | None = None,
        include: Callable[[Path], bool] | None = None,
        exclude: Callable[[Path], bool] | None = None,
    ) -> Artifact:
        """Upload source code files under a root path as an artifact.

        Args:
            root_path: Root directory to scan. Defaults to the current working directory.
            name: Optional artifact name.
            include: Optional whitelist callable. It receives each resolved
                absolute file path and returns True to include it.
            exclude: Optional blacklist callable. It receives each resolved
                absolute file path and returns True to exclude it.

        Returns:
            Artifact containing matched files from the root path.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
            RuntimeError: If the run has already been finished.
            ValueError: If ``root_path`` is not a directory.
        """
        self._require_active()
        root = Path.cwd() if root_path is None else Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"root_path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"root_path must point to a directory: {root}")

        resolved_root = root.resolve()
        artifact = Artifact(name or "source-code", "code")
        include_match = include or (lambda path: path.suffix == ".py")
        paths = sorted((path for path in resolved_root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())

        for path in paths:
            if include_match(path) and (exclude is None or not exclude(path)):
                artifact.add_file(path, name=path.relative_to(resolved_root).as_posix())

        self.log_artifact(artifact)
        return artifact

    def log_model(self, checkpoint: PathOrBytes, *, name: str | None = None) -> Artifact:
        """Upload a model checkpoint as an artifact.

        Args:
            checkpoint: Model checkpoint as a file path, directory path, or byte buffer.
            name: Optional artifact name.

        Returns:
            Artifact containing the model checkpoint.

        Raises:
            FileNotFoundError: If a path checkpoint does not exist.
            RuntimeError: If the run has already been finished.
            TypeError: If ``checkpoint`` is not path-like or bytes-like.
            ValueError: If a path checkpoint is neither a file nor directory.
        """
        self._require_active()
        artifact = Artifact(name or "model-checkpoint", "model")
        if isinstance(checkpoint, (bytes, bytearray, memoryview)):
            artifact.add_bytes(checkpoint, name="checkpoint.bin")
        elif isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"checkpoint path does not exist: {path}")
            if path.is_file():
                artifact.add_file(path)
            elif path.is_dir():
                artifact.add_dir(path)
            else:
                raise ValueError(f"checkpoint path must point to a file or directory: {path}")
        else:
            raise TypeError("checkpoint must be a path string, Path, or bytes-like object")

        self.log_artifact(artifact)
        return artifact

    def log_artifact(self, artifact: Artifact) -> None:
        """Upload an artifact's attached entries to the backend.

        Args:
            artifact: Artifact to upload.

        Raises:
            RuntimeError: If the run has already been finished.
            TypeError: If ``artifact`` is not an ``underfit.Artifact``.
        """
        self._require_active()
        if not isinstance(artifact, Artifact):
            raise TypeError("artifact must be an underfit.Artifact")

        for entry in artifact.upload_manifest():
            self.backend.upload_artifact_entry(artifact.name, entry)

    def finish(self) -> None:
        """Finalize the run."""
        if self._finished:
            return
        self.backend.finish()
        self._finished = True

    def read_scalars(self) -> list[dict[str, Any]]:
        """Return scalar records available from the active backend."""
        return self.backend.read_scalars()

    def read_logs(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        """Return log records available from the active backend."""
        return self.backend.read_logs(worker_id)

    def read_artifact_entries(self, artifact_name: str | None = None) -> list[dict[str, Any]]:
        """Return artifact entries available from the active backend."""
        return self.backend.read_artifact_entries(artifact_name)
