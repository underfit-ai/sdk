"""Run model for Underfit."""

from __future__ import annotations

import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Union
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from underfit.artifact import Artifact
from underfit.backends.base import Backend
from underfit.media import Audio, Html, Image, Video

PathLike = Union[str, Path]
PathOrBytes = Union[str, Path, bytes, bytearray, memoryview]


def _git_output(repo_path: Path, args: list[str], *, ok_returncodes: tuple[int, ...] = (0,)) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, check=False)  # noqa: S603,S607
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed") from exc

    if result.returncode not in ok_returncodes:
        command = " ".join(["git", *args])
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"{command} failed with exit code {result.returncode}")
    return result.stdout


def _git_text(repo_path: Path, args: list[str], *, ok_returncodes: tuple[int, ...] = (0,)) -> str:
    return _git_output(repo_path, args, ok_returncodes=ok_returncodes).decode("utf-8").strip()


def _join_bytes(chunks: list[bytes]) -> bytes:
    output = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        if output and not output.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(chunk)
    return bytes(output)


class Run:
    """Represent an Underfit run."""

    def __init__(self, project: str, name: str, backend: Backend, config: dict[str, Any] | None = None) -> None:
        """Initialize a run.

        Args:
            project: Project name.
            name: Run name.
            backend: Backend used to store run data.
            config: Run configuration dictionary.
        """
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
                if value and all(isinstance(v, (Html, Image, Video, Audio)) for v in value):
                    media_batches.append((key, list(value)))
                else:
                    raise TypeError(f"Lists passed to underfit.Run.log must contain only media objects: {key}")
            else:
                raise TypeError(f"Unsupported value for underfit.Run.log: {key}")

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
        """Upload source code under a root path as a zip artifact.

        Args:
            root_path: Root directory to scan. Defaults to the current working directory.
            name: Optional artifact name.
            include: Optional whitelist callable. It receives each resolved
                absolute file path and returns True to include it.
            exclude: Optional blacklist callable. It receives each resolved
                absolute file path and returns True to exclude it.

        Returns:
            Artifact containing a zip archive of matched files from the root path.

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
        matched_paths = [path for path in paths if include_match(path) and (exclude is None or not exclude(path))]

        if matched_paths:
            buffer = BytesIO()
            with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
                for path in matched_paths:
                    entry_name = path.relative_to(resolved_root).as_posix()
                    info = ZipInfo(entry_name)
                    info.compress_type = ZIP_DEFLATED
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    archive.writestr(info, path.read_bytes())
            artifact.add_bytes(buffer.getvalue(), name=f"{artifact.name}.zip")

        self.log_artifact(artifact)
        return artifact

    def log_git(self, repo_path: PathLike | None = None, *, name: str | None = None) -> Artifact:
        """Upload the current git state as an artifact.

        Args:
            repo_path: Git repository path. Defaults to the current working directory.
            name: Optional artifact name.

        Returns:
            Artifact containing the working tree patch and git metadata.

        Raises:
            FileNotFoundError: If ``repo_path`` does not exist.
            RuntimeError: If git is not installed, the path is not in a git
                repository, or the run has already been finished.
            ValueError: If ``repo_path`` is not a directory.
        """
        self._require_active()
        repo = Path.cwd() if repo_path is None else Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"repo_path does not exist: {repo}")
        if not repo.is_dir():
            raise ValueError(f"repo_path must point to a directory: {repo}")

        repo_root = Path(_git_text(repo, ["rev-parse", "--show-toplevel"]))
        status = _git_text(repo_root, ["status", "--porcelain=v2", "--branch"], ok_returncodes=(0, 128))
        head = None
        branch = None
        untracked_files: list[str] = []

        for line in status.splitlines():
            if line.startswith("# branch.oid "):
                value = line.removeprefix("# branch.oid ")
                head = None if value == "(initial)" else value
            elif line.startswith("# branch.head "):
                value = line.removeprefix("# branch.head ")
                branch = None if value == "(detached)" else value
            elif line.startswith("? "):
                untracked_files.append(line[2:])

        patch = (
            _git_output(repo_root, ["diff", "--binary", "HEAD"])
            if head is not None
            else _join_bytes([
                _git_output(repo_root, ["diff", "--binary", "--cached"]),
                _git_output(repo_root, ["diff", "--binary"]),
            ])
        )
        artifact = Artifact(name or "git-state", "git", metadata={
            "commit": head,
            "branch": branch,
            "is_dirty": bool(patch) or bool(untracked_files),
            "untracked_files": untracked_files,
        })
        artifact.add_bytes(patch, name="working-tree.patch")
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

        self.backend.log_artifact(artifact)

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
